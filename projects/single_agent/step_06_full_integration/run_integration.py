import argparse
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict

import json
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.agents.dubins_robot import DubinsRobot
from src.controllers.base_multi_agent import AgentMPPI, BaseMultiAgentController
from src.controllers.cbf_smoke import CBFSmokeController
from src.controllers.schemas import BaseMultiAgentConfig
from src.env.smoke_env import SmokeEnv
from src.utils.time_tracker import TimeTracker

from projects.single_agent.step_06_full_integration.schema import IntegrationConfig

# Optional imports for FNO mode
try:
    from src.models.lightning_fno import FNOLightningModule
    from src.models.shared.schemas import FNOTrainingConfig
    from src.wrappers.smoke_forecast_wrapper import _cvar
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_integration")


def load_config(config_path: str) -> IntegrationConfig:
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f) or {}
    return IntegrationConfig.model_validate(yaml_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_no_risk.yaml", help="Path to config file")
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)
    log.info(f"Loaded configuration for experiment_mode: {cfg.experiment_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Environment first (crucial for PlaybackConfig)
    log.info("Initializing SmokeEnv (Playback)...")
    env = SmokeEnv(
        env_cfg=cfg.env,
        robot_cfg=cfg.robot,
        sensor_cfg=cfg.sensor,
        simulator_cfg=cfg.simulator,
    )

    # 2. Load Predictive Model if FNO mode
    fno_model = None
    if cfg.experiment_mode == "fno":
        log.info("Loading FNO model...")
        with open(cfg.fno_config, "r") as f:
            fno_cfg_data = yaml.safe_load(f)
        fno_train_cfg = FNOTrainingConfig.model_validate(fno_cfg_data)

        # H and W come directly from the Playback simulator
        H = env.smoke_simulator.H
        W = env.smoke_simulator.W
        x_size = env.smoke_simulator.cfg.x_size
        y_size = env.smoke_simulator.cfg.y_size

        fno_model = FNOLightningModule.load_from_checkpoint(
            cfg.fno_checkpoint, t_cfg=fno_train_cfg, H=H, W=W, x_size=x_size, y_size=y_size
        )
        fno_model.to(device)
        fno_model.eval()
        log.info("FNO model loaded successfully.")

    # 3. Setup Controller
    goal_location = np.array(cfg.env.goal_locations[0])
    
    if cfg.experiment_mode == "cbf":
        log.info("Initializing CBFSmokeController...")
        controller = CBFSmokeController(
            config=cfg.cbf,
            env_config=cfg.env,
            robot_config=cfg.robot,
            goal=goal_location,
            num_agents=1
        )
    else:
        log.info("Initializing BaseMultiAgentController (MPPI)...")
        # BaseMultiAgentController needs BaseMultiAgentConfig
        mppi_cfg_wrapped = BaseMultiAgentConfig(
            mppi=cfg.mppi,
            dt=cfg.env.clock
        )
        controller = BaseMultiAgentController(
            num_agents=1,
            robot_config=cfg.robot,
            config=mppi_cfg_wrapped,
            goal_thresh=cfg.env.goal_radius,
            dtype=torch.float32
        )
        controller.set_goals({"agent_0": goal_location})

    # 4. Run Simulation Loop
    log.info("Starting simulation loop...")
    obs, _ = env.reset()

    # Buffers for FNO context window
    h_ctx = fno_train_cfg.model.h_ctx if cfg.experiment_mode == "fno" else 1
    smoke_history = deque(maxlen=h_ctx)
    max_steps = cfg.env.max_steps if cfg.env.max_steps else 1000

    # Initialize TimeTracker
    tracker = TimeTracker()

    for step in tqdm(range(max_steps), desc="Simulation Steps"):
        agent_obs = obs.get("agent_0")
        if agent_obs is None:
            break # Agent terminated or truncated

        # Extract current state
        loc = agent_obs["location"]
        angle = agent_obs["angle"]
        state_np = np.array([loc[0], loc[1], float(np.ravel(angle)[0])])

        smoke_density = agent_obs["smoke_density"]
        smoke_positions = agent_obs["smoke_density_location"]
        
        # Grid dimensions from playback
        H = env.smoke_simulator.H
        W = env.smoke_simulator.W
        
        # Dispatch logic based on mode
        if cfg.experiment_mode == "cbf":
            with tracker.track("cbf_update"):
                controller.update_h_discrete(smoke_density.flatten(), smoke_positions, state_np)
            with tracker.track("cbf_control"):
                cmd = controller.get_command(state_np)
                action = {"agent_0": cmd}

        else:
            # MPPI logic for maps
            maps_deque = deque(maxlen=cfg.mppi.horizon)
            
            if cfg.experiment_mode == "no_risk":
                # Do nothing, empty maps_deque means zero risk cost
                pass
                
            elif cfg.experiment_mode == "persistent":
                # Duplicate current smoke density across the entire horizon
                for _ in range(cfg.mppi.horizon):
                    maps_deque.append((smoke_positions, smoke_density.flatten()))
                    
            elif cfg.experiment_mode == "fno":
                # Reshape smoke to (H, W) and append to history
                smoke_grid = smoke_density.reshape(H, W)
                smoke_history.append(torch.tensor(smoke_grid, dtype=torch.float32, device=device))
                
                # If we have enough context frames, run autoregressive prediction
                if len(smoke_history) == h_ctx:
                    with tracker.track("fno_inference"):
                        with torch.no_grad():
                            # Stack context: (1, h_ctx, H, W)
                            ctx_w = torch.stack(list(smoke_history)).unsqueeze(0)
                            
                            preds = fno_model.model.autoregressive_forecast(
                                ctx_w, seed_t_start=0, horizon=cfg.mppi.horizon, num_samples=1, mode="mean"
                            )
                            
                            # Populate maps_deque with forecasted frames
                            for p in preds:
                                pred_mean = p["mean"][0] # (H, W)
                                pred_std = p["std"][0]
                                cvar_grid = _cvar(pred_mean, pred_std, alpha=cfg.fno_cvar_alpha)
                                maps_deque.append((smoke_positions, cvar_grid.flatten()))
                else:
                    # Fallback to persistent if context is not yet full
                    for _ in range(cfg.mppi.horizon):
                        maps_deque.append((smoke_positions, smoke_density.flatten()))

            controller.set_maps(maps_deque)
            # get_commands returns Dict[str, torch.Tensor]
            with tracker.track("mppi_control"):
                cmd_dict = controller.get_commands(obs)
                action = cmd_dict

        # Step the environment
        with tracker.track("env_step"):
            obs, rewards, terminations, truncations, infos = env.step(action)
        
        # Render
        if cfg.env.render and cfg.env.render != "none":
            with tracker.track("render"):
                env.render(controller=controller if cfg.experiment_mode != "cbf" else None)

        if terminations.get("agent_0", False) or truncations.get("agent_0", False):
            log.info(f"Episode finished at step {step}. Info: {infos.get('agent_0')}")
            break

    env.close()
    
    # Save TimeTracker metrics to output directory
    save_dir = getattr(cfg.env, "save_transitions_path", "outputs/integration/default")
    os.makedirs(save_dir, exist_ok=True)
    timing_file = os.path.join(save_dir, "timing_metrics.json")
    with open(timing_file, "w") as f:
        json.dump(tracker.summary(), f, indent=4)
        
    tracker.pretty_print()
    log.info(f"Simulation completed. Transitions and timing metrics saved to {save_dir}")

if __name__ == "__main__":
    main()
