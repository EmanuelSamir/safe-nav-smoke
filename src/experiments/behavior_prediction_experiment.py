"""
BehaviorPredictionExperiment — MPPI with online model-based map prediction.

The SmokeForecastWrapper provides H future risk maps (CVaR) using the
selected model (gp | rnp | rnp_multistep | fno | fno_3d).
MPPI uses these maps to compute risk-aware trajectories.

Rendering
---------
  env.render = "human"     → live matplotlib window updated every step
  env.render = "rgb_array" → no live window, frames saved to video.mp4 at teardown

Run examples:
  python run_experiment.py experiment=behavior_prediction experiment.model_type=gp env.render=human
  python run_experiment.py experiment=behavior_prediction experiment.model_type=fno \
      experiment.checkpoint_path=outputs/.../best_model.pt
"""
import logging
from collections import deque

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.experiments.base_experiment import BaseExperiment
from src.mppi_control_dyn import MPPIControlDyn, MPPIControlParams
from src.smoke_forecast_wrapper import SmokeForecastWrapper
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv, DynamicSmokeParams
from agents.basic_robot import RobotParams
from simulator.sensor import GlobalSensorParams
from simulator.dynamic_smoke import SmokeBlobParams

log = logging.getLogger(__name__)


class BehaviorPredictionExperiment(BaseExperiment):

    # ------------------------------------------------------------------
    def setup(self):
        # --- Env params -----------------------------------------------
        self.env_params = EnvParams()
        self.env_params.max_steps     = self.cfg.env.max_steps
        self.env_params.clock         = self.cfg.env.clock
        self.env_params.render        = self.cfg.env.render
        self.env_params.goal_location = np.array(self.cfg.env.goal_location)
        self.env_params.goal_radius   = self.cfg.env.goal_radius

        # PlaybackSmoke path (overrides world size when set)
        playback_path = self.cfg.env.get("playback_path", None)
        self.env_params.playback_path = str(playback_path) if playback_path else None

        # World size defaults — will be overridden by PlaybackSmoke if path is set
        self.env_params.world_x_size = self.cfg.env.world_x_size
        self.env_params.world_y_size = self.cfg.env.world_y_size

        # Always GlobalSensor — models trained on dense global maps
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
        # Dummy smoke params — ignored by DynamicSmokeEnv when playback_path is set
        self.smoke_params = DynamicSmokeParams(
            x_size=self.env_params.world_x_size,
            y_size=self.env_params.world_y_size,
            smoke_blob_params=[SmokeBlobParams(x_pos=10, y_pos=10, intensity=1.0, spread_rate=2.0)],
            resolution=1.0,
        )

        # --- Environment -----------------------------------------------
        self.env = DynamicSmokeEnv(self.env_params, self.robot_params, self.smoke_params)

        # After env init: playback overrides world_x/y_size — rebuild sensor with correct size
        x_size = self.env.env_params.world_x_size
        y_size = self.env.env_params.world_y_size
        
        # GP is very slow on a 15,000 point full grid ~ O(N^3). Downsample if 'gp'
        resolution = 5.0
        if self.cfg.experiment.model_type == "gp":
            resolution = self.cfg.experiment.get("gp_sensor_resolution", 1.0)
            
        self.sensor_params = GlobalSensorParams(
            world_x_size=x_size, 
            world_y_size=y_size,
            density_reading_per_unit_length=resolution,
        )
        self.env.env_params.sensor_params = self.sensor_params
        self.env.sensor = type(self.env.sensor)(self.sensor_params)  # rebuild sensor
        self.robot_params.state_max[0] = x_size
        self.robot_params.state_max[1] = y_size

        # Update wrapper with corrected world size
        self.wrapper = SmokeForecastWrapper(
            model_type  = self.cfg.experiment.model_type,
            x_size      = x_size,
            y_size      = y_size,
            checkpoint  = self.cfg.experiment.get("checkpoint_path", None) or None,
            cvar_alpha  = self.cfg.experiment.get("cvar_alpha", 0.95),
            gamma       = self.cfg.experiment.get("gamma", 0.75),
            beta        = self.cfg.experiment.get("beta", 0.20),
            device      = self.cfg.experiment.get("device", "cpu"),
        )

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

        # --- Forecast wrapper ------------------------------------------
        model_type  = self.cfg.experiment.model_type
        checkpoint  = self.cfg.experiment.get("checkpoint_path", None) or None
        device      = self.cfg.experiment.get("device", "cpu")

        self.wrapper = SmokeForecastWrapper(
            model_type  = model_type,
            x_size      = x_size,
            y_size      = y_size,
            checkpoint  = checkpoint,
            cvar_alpha  = self.cfg.experiment.get("cvar_alpha", 0.95),
            gamma       = self.cfg.experiment.get("gamma", 0.75),
            beta        = self.cfg.experiment.get("beta", 0.20),
            device      = device,
        )
        log.info(f"Forecast model: {model_type} | checkpoint: {checkpoint} | "
                 f"world=({x_size}x{y_size}) | render={self.cfg.env.render}")

        # --- Controller ------------------------------------------------
        horizon = self.cfg.experiment.time_horizon
        mppi_params = MPPIControlParams(
            horizon=horizon,
            num_samples=self.cfg.experiment.get("samples", 50),
        )
        self.controller = MPPIControlDyn(
            self.robot_params,
            self.cfg.agent.name,
            mppi_params=mppi_params,
            goal_thresh=self.env_params.goal_radius,
        )
        self.controller.set_goal(self.goal_location.tolist())

        # 4. Setup renderer
        self.setup_renderer(has_predictions=True)

    # ------------------------------------------------------------------
    def run_episode(self):
        finished = False
        horizon  = self.cfg.experiment.time_horizon

        for t in tqdm(range(self.env.env_params.max_steps + 1)):
            if finished:
                break

            smoke_map    = self.state["smoke_density"].squeeze().astype(np.float32)  # (H*W,)
            smoke_coords = self.state["smoke_density_location"]

            # 1. Update wrapper with current observation ---------------
            with self.time_tracker.track("model_update"):
                self.wrapper.update(smoke_map, smoke_coords, t * self.env_params.clock)

            # 2. Predict H future risk maps ----------------------------
            with self.time_tracker.track("prediction"):
                predicted_maps = self.wrapper.predict_risk_maps(
                    smoke_map, smoke_coords,
                    t * self.env_params.clock,
                    horizon,
                )

            # 3. Control (MPPI) ----------------------------------------
            with self.time_tracker.track("control"):
                self.controller.set_state(np.array([
                    self.state["location"][0],
                    self.state["location"][1],
                    self.state["angle"],
                ]))
                self.controller.set_maps(deque(predicted_maps, maxlen=horizon))
                nominal_action = self.controller.get_command()
                action_input   = nominal_action.numpy()

            # 4. Environment step --------------------------------------
            with self.time_tracker.track("env"):
                next_state, reward, terminated, truncated, _ = self.env.step(
                    np.array(action_input)
                )

            # 5. Logging & metrics -------------------------------------
            finished = self.check_termination(self.state, reward, terminated, truncated)
            self.log_common_metrics(self.state, action_input, t)

            # 6. Rendering ---------------------------------------------
            self.render_step({
                "state": self.state,
                "env": self.env,
                "nom_controller": self.controller,
                "predicted_maps": predicted_maps,
                "builder": self.controller.cost_function.builder if hasattr(self.controller, "cost_function") else None
            }, t)

            self.state = next_state

        return {"status": "completed"}
