import argparse
import logging
import os
import sys
import json
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.agents.dubins_robot import DubinsRobot
from src.controllers.cbf_smoke import CBFSmokeController
from src.env.smoke_env import SmokeEnv
from src.utils.time_tracker import TimeTracker
from projects.single_agent.step_05_cbf_sweep_evaluation.schema import SweepConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_sweep")

def load_config(config_path: str) -> SweepConfig:
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f) or {}
    return SweepConfig.model_validate(yaml_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_sweep.yaml", help="Path to config file")
    parser.add_argument("--sweep_idx", type=int, default=None, help="Index of the sweep_values to run. If None, runs all.")
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)
    
    if cfg.test_mode:
        log.info("Running in TEST MODE. Limiting sweep values and enabling rendering.")
        cfg.sweep_values = [cfg.sweep_values[0]] if cfg.sweep_values else []
        cfg.env.render = "human"
    
    if args.sweep_idx is not None:
        if args.sweep_idx < 0 or args.sweep_idx >= len(cfg.sweep_values):
            raise ValueError(f"Invalid sweep_idx {args.sweep_idx}. Max is {len(cfg.sweep_values)-1}")
        values_to_run = [cfg.sweep_values[args.sweep_idx]]
        indices = [args.sweep_idx]
    else:
        values_to_run = cfg.sweep_values
        indices = list(range(len(cfg.sweep_values)))

    log.info(f"Running sweep over {cfg.sweep_param} with values: {values_to_run}")

    for idx, val in zip(indices, values_to_run):
        log.info(f"--- Starting Evaluation: {cfg.sweep_param} = {val} ---")
        
        # Inject the swept parameter into cbf_base
        setattr(cfg.cbf_base, cfg.sweep_param, val)
        
        # Determine output directory to segregate datasets
        base_path = os.path.join("outputs", cfg.project_name, cfg.sub_project_name)
        val_str = f"{val:.4f}".replace(".", "_")
        current_save_path = os.path.join(base_path, cfg.sweep_param, f"val_{val_str}")

        # Assert to prevent overwriting existing data
        if os.path.exists(current_save_path) and len(os.listdir(current_save_path)) > 0:
            assert False, f"Output directory {current_save_path} already exists and is not empty. Aborting to prevent overwrite."

        if cfg.test_mode:
            log.info(f"TEST MODE: Data would be saved to {current_save_path}, but saving is disabled.")
            cfg.env.save_transitions = False
        else:
            cfg.env.save_transitions_path = current_save_path
            os.makedirs(current_save_path, exist_ok=True)
            
            # Save a copy of the specific config used for this run
            with open(os.path.join(current_save_path, "run_config.json"), "w") as f:
                f.write(cfg.model_dump_json(indent=2))

        # Setup Env
        env = SmokeEnv(
            env_cfg=cfg.env,
            robot_cfg=cfg.robot,
            sensor_cfg=cfg.sensor,
            simulator_cfg=cfg.simulator,
        )

        assert cfg.env.goal_locations is not None, "goal_locations must be defined in the env configuration"
        goal_loc = cfg.env.goal_locations[0]
        controller = CBFSmokeController(
            config=cfg.cbf_base,
            env_config=cfg.env,
            robot_config=cfg.robot,
            goal=np.array(goal_loc),
            num_agents=1
        )
        tracker = TimeTracker()

        obs, info = env.reset()
        max_steps = cfg.env.max_steps

        for step in tqdm(range(max_steps), desc=f"Evaluating {val}"):
            agent_id = list(obs.keys())[0] if obs else None
            agent_obs = obs.get(agent_id) if agent_id else None
            if agent_obs is None:
                break

            loc = agent_obs["location"]
            angle = agent_obs["angle"]
            state_np = np.array([loc[0], loc[1], float(np.ravel(angle)[0])])

            smoke_density = agent_obs["smoke_density"]
            smoke_positions = agent_obs["smoke_density_location"]

            with tracker.track("cbf_update"):
                controller.update_h_discrete(smoke_density.flatten(), smoke_positions, state_np)
            with tracker.track("cbf_control"):
                cmd = controller.get_command(state_np)
                action = {agent_id: cmd}

            with tracker.track("env_step"):
                obs, rewards, terminations, truncations, infos = env.step(action)
                
            if cfg.env.render and cfg.env.render != "none":
                with tracker.track("render"):
                    env.render(controller=controller)

            if terminations.get(agent_id, False) or truncations.get(agent_id, False):
                log.info(f"Episode finished at step {step}. Info: {infos.get(agent_id)}")
                break
        
        env.close()
        
        if not cfg.test_mode:
            # Save TimeTracker metrics
            timing_file = os.path.join(current_save_path, "timing_metrics.json")
            with open(timing_file, "w") as f:
                json.dump(tracker.summary(), f, indent=4)
            log.info(f"Finished {cfg.sweep_param}={val}. Saved to {current_save_path}")
        else:
            log.info(f"Finished test run for {cfg.sweep_param}={val}.")

if __name__ == "__main__":
    main()
