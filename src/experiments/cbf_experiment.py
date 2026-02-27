import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.experiments.base_experiment import BaseExperiment
from src.cbf_controller import CBFController
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv, DynamicSmokeParams
from agents.basic_robot import RobotParams
from simulator.sensor import GlobalSensorParams
from simulator.dynamic_smoke import SmokeBlobParams

class CBFExperiment(BaseExperiment):
    def setup(self):
        # 1. Env Params
        self.env_params = EnvParams()
        self.env_params.max_steps     = self.cfg.env.max_steps
        self.env_params.clock         = self.cfg.env.clock
        self.env_params.render        = self.cfg.env.render
        self.env_params.goal_location = np.array(self.cfg.env.goal_location)
        self.env_params.goal_radius   = self.cfg.env.goal_radius

        # PlaybackSmoke path (overrides world size when set)
        playback_path = self.cfg.env.get("playback_path", None)
        self.env_params.playback_path = str(playback_path) if playback_path else None

        self.env_params.world_x_size = self.cfg.env.world_x_size
        self.env_params.world_y_size = self.cfg.env.world_y_size

        # Always GlobalSensor for CBF — needs full map to build distance map
        self.sensor_params = GlobalSensorParams(
            world_x_size=self.env_params.world_x_size,
            world_y_size=self.env_params.world_y_size,
        )
        self.env_params.sensor_params = self.sensor_params

        # Smoke Params (fallback when no playback)
        # Dummy smoke params — ignored by DynamicSmokeEnv when playback_path is set
        self.smoke_params = DynamicSmokeParams(
            x_size=self.env_params.world_x_size,
            y_size=self.env_params.world_y_size,
            smoke_blob_params=[SmokeBlobParams(x_pos=10, y_pos=10, intensity=1.0, spread_rate=2.0)],
            resolution=1.0,
        )

        # Robot Params
        self.robot_params = RobotParams()
        self.robot_params.world_x_size = self.env_params.world_x_size
        self.robot_params.world_y_size = self.env_params.world_y_size
        self.robot_params.action_min = np.array(self.cfg.agent.action_min)
        self.robot_params.action_max = np.array(self.cfg.agent.action_max)
        self.robot_params.dt = self.cfg.agent.dt

        # 2. Initialize Environment
        self.env = DynamicSmokeEnv(self.env_params, self.robot_params, self.smoke_params)

        # Sync world size after env init (playback overrides it)
        self.sensor_params.world_x_size = self.env.env_params.world_x_size
        self.sensor_params.world_y_size = self.env.env_params.world_y_size
        self.robot_params.state_max[0]  = self.env.env_params.world_x_size
        self.robot_params.state_max[1]  = self.env.env_params.world_y_size
        
        initial_loc = self.get_initial_location()
        episode_idx = self.cfg.experiment.get("episode_idx", None)
        self.state, _ = self.env.reset(
            initial_state={
                "location": initial_loc, 
                "angle": 0.0, 
                "smoke_density": 0.0
            },
            seed=episode_idx
        )
        
        self.goal_location = self.get_goal_location()
        self.env.env_params.goal_location = self.goal_location

        # 3. CBF Controller
        self.controller = CBFController(
            self.env.env_params,
            self.robot_params,
            self.goal_location,
            smoke_threshold=self.cfg.experiment.smoke_threshold
        )

        # Setup standard renderer
        self.setup_renderer(has_predictions=False)

    def run_episode(self):
        finished = False
        
        for t in tqdm(range(self.env_params.max_steps + 1)):
            if finished:
                break

            # 1. Update Control Logic (CBF)
            with self.time_tracker.track("cbf"):
                pos_x, pos_y = self.state["location"]
                angle = self.state["angle"]
                x_state = np.array([pos_x, pos_y, angle])

                # Update Barrier Function (h) based on smoke density
                self.controller.update_h_discrete(
                    self.state["smoke_density"].flatten(), 
                    self.state["smoke_density_location"], 
                    x_state
                )

                # Get dynamics and jacobian from robot
                f = self.env.robot.open_loop_dynamics(x_state)
                g = self.env.robot.control_jacobian(x_state)

                # Compute Safe Action
                action_input = self.controller.get_command(x_state, f, g)

            # 2. Environment Step
            with self.time_tracker.track("env"):
                next_state, reward, terminated, truncated, _ = self.env.step(np.array(action_input))
            
            # 3. Logging & Metrics
            finished = self.check_termination(self.state, reward, terminated, truncated)
            self.log_common_metrics(self.state, action_input, t)

            # 4. Rendering
            self.render_step({
                "state": self.state,
                "env": self.env,
                "nom_controller": None
            }, t)

            self.state = next_state

        return {"status": "completed"}
