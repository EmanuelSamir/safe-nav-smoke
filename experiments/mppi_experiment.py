"""
MPPIExperiment — covers two experiment modes:
  - no-risk    (risk_aware=False): pure MPPI, no risk cost at all
  - persistent (risk_aware=True):  MPPI with the last observed smoke frame
                                   repeated for the full horizon (static assumption)

Rendering
---------
  env.render = "human"     → live matplotlib window updated every step
  env.render = "rgb_array" → no live window, frames saved to video.mp4 at teardown

Run with live view:
  python run_experiment.py experiment=no_risk env.render=human
"""
import logging
from collections import deque
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from experiments.base_experiment import BaseExperiment
from controllers.mppi_ctrl import MPPICtrl, MPPICtrlParams
from envs.smoke_env import EnvParams, SmokeEnv, SmokeParams
from agents.basic_robot import RobotParams
from envs.simulator.sensor import GlobalSensorParams
from envs.simulator.smoke import BlobParams

log = logging.getLogger(__name__)


class MPPIExperiment(BaseExperiment):
    def __init__(self, cfg: DictConfig, risk_aware: bool = False):
        super().__init__(cfg)
        self.risk_aware = risk_aware

    # ------------------------------------------------------------------
    def setup(self):
        # --- Env params -----------------------------------------------
        self.env_params = EnvParams()
        self.env_params.max_steps     = self.cfg.env.max_steps
        self.env_params.clock         = self.cfg.env.clock
        self.env_params.render        = self.cfg.env.render
        self.env_params.goal_location = np.array(self.cfg.env.goal_location)
        self.env_params.goal_radius   = self.cfg.env.goal_radius

        # Playback path (overrides world size when set)
        playback_path = self.cfg.env.get("playback_path", None)
        self.env_params.playback_path = str(playback_path) if playback_path else None

        # World size defaults — will be overridden by Playback if path is set
        self.env_params.world_x_size = self.cfg.env.world_x_size
        self.env_params.world_y_size = self.cfg.env.world_y_size

        # Always GlobalSensor so we get a full map observation
        self.sensor_params = GlobalSensorParams(
            world_x_size=self.env_params.world_x_size,
            world_y_size=self.env_params.world_y_size,
        )
        self.env_params.sensor_params = self.sensor_params

        # --- Robot params -----------------------------------------------
        self.robot_params = RobotParams()
        self.robot_params.world_x_size = self.env_params.world_x_size
        self.robot_params.world_y_size = self.env_params.world_y_size
        self.robot_params.action_min   = np.array(self.cfg.agent.action_min)
        self.robot_params.action_max   = np.array(self.cfg.agent.action_max)
        self.robot_params.dt           = self.cfg.agent.dt

        # Smoke blobs only needed when not using playback
        # Dummy smoke params — ignored by SmokeEnv when playback_path is set
        self.smoke_params = SmokeParams(
            x_size=self.env_params.world_x_size,
            y_size=self.env_params.world_y_size,
            smoke_blob_params=[BlobParams(x_pos=10, y_pos=10, intensity=1.0, spread_rate=2.0)],
            resolution=1.0,
        )

        # --- Environment -----------------------------------------------
        self.env = SmokeEnv(self.env_params, self.robot_params, self.smoke_params)

        # After env init: playback overrides world_x/y_size — rebuild sensor with correct size
        x_size = self.env.env_params.world_x_size
        y_size = self.env.env_params.world_y_size
        self.sensor_params = GlobalSensorParams(world_x_size=x_size, world_y_size=y_size)
        self.env.env_params.sensor_params = self.sensor_params
        self.env.sensor = type(self.env.sensor)(self.sensor_params)  # rebuild sensor
        self.robot_params.state_max[0] = x_size
        self.robot_params.state_max[1] = y_size

        initial_loc   = self.get_initial_location()
        episode_idx   = self.cfg.experiment.get("episode_idx", None)
        self.state, _ = self.env.reset(
            initial_state={
                "location": initial_loc, "angle": 0.0, "smoke_density": 0.0
            },
            seed=episode_idx
        )
        self.goal_location = self.get_goal_location()
        self.env.env_params.goal_location = self.goal_location

        # --- Controller ------------------------------------------------
        horizon = self.cfg.experiment.time_horizon
        mppi_params = MPPICtrlParams(
            horizon=horizon,
            num_samples=self.cfg.experiment.get("samples", 50),
        )
        self.controller = MPPICtrl(
            self.robot_params,
            self.cfg.agent.name,
            mppi_params=mppi_params,
            goal_thresh=self.env_params.goal_radius,
        )
        self.controller.set_goal(self.goal_location.tolist())

        # Setup standard renderer
        self.setup_renderer(has_predictions=False)

        log.info(f"MPPIExperiment ready | risk_aware={self.risk_aware} | "
                 f"world=({self.env.env_params.world_x_size}x{self.env.env_params.world_y_size}) | "
                 f"render={self.cfg.env.render}")

    # ------------------------------------------------------------------
    def run_episode(self):
        finished = False
        horizon  = self.cfg.experiment.time_horizon

        for t in tqdm(range(self.env.env_params.max_steps + 1)):
            if finished:
                break

            # 1. Persistent: reuse current smoke map for all H steps ----
            if self.risk_aware:
                # squeeze (H*W, 1) → (H*W,) as Playback returns (N, 1)
                smoke_map    = self.state["smoke_density"].squeeze().astype(np.float32)
                smoke_coords = self.state["smoke_density_location"]
                with self.time_tracker.track("set_maps"):
                    self.controller.set_maps(
                        deque([(smoke_coords, smoke_map)] * horizon, maxlen=horizon)
                    )
            # No-risk: no set_maps call → _compute_risk_cost returns zeros

            # 2. Control (MPPI) -----------------------------------------
            with self.time_tracker.track("control"):
                self.controller.set_state(np.array([
                    self.state["location"][0],
                    self.state["location"][1],
                    self.state["angle"],
                ]))
                nominal_action = self.controller.get_command()
                action_input   = nominal_action.numpy()

            # 3. Environment step ----------------------------------------
            with self.time_tracker.track("env"):
                next_state, reward, terminated, truncated, _ = self.env.step(
                    np.array(action_input)
                )

            # 4. Logging & metrics ---------------------------------------
            finished = self.check_termination(self.state, reward, terminated, truncated)
            self.log_common_metrics(self.state, action_input, t)

            # 5. Rendering -----------------------------------------------
            self.render_step({
                "state": self.state,
                "env": self.env,
                "nom_controller": self.controller
            }, t)

            self.state = next_state

        return {"status": "completed"}
