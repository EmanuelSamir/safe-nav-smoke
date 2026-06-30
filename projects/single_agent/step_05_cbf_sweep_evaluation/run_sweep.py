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
        
        # Modify output directory to segregate datasets
        base_path = getattr(cfg.env, "save_transitions_path", "outputs/cbf_sweep")
        val_str = f"{val:.4f}".replace(".", "_")
        current_save_path = os.path.join(base_path, cfg.sweep_param, f"val_{val_str}")
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

        controller = CBFSmokeController(cfg.cbf_base)
        tracker = TimeTracker()

        obs, info = env.reset()
        max_steps = cfg.env.max_steps if cfg.env.max_steps else 1000

        for step in tqdm(range(max_steps), desc=f"Evaluating {val}"):
            agent_obs = obs.get("agent_0")
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
                action = {"agent_0": cmd}

            with tracker.track("env_step"):
                obs, rewards, terminations, truncations, infos = env.step(action)
                
            if cfg.env.render and cfg.env.render != "none":
                with tracker.track("render"):
                    env.render(controller=controller)

            if terminations.get("agent_0", False) or truncations.get("agent_0", False):
                log.info(f"Episode finished at step {step}. Info: {infos.get('agent_0')}")
                break
        
        env.close()
        
        # Save TimeTracker metrics
        timing_file = os.path.join(current_save_path, "timing_metrics.json")
        with open(timing_file, "w") as f:
            json.dump(tracker.summary(), f, indent=4)

        log.info(f"Finished {cfg.sweep_param}={val}. Saved to {current_save_path}")

if __name__ == "__main__":
    main()
