import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from collections import deque

from src.experiments.base_experiment import BaseExperiment
from src.mppi_control_dyn import MPPIControlDyn, MPPIControlParams
from src.risk_map_builder import RiskMapBuilder, RiskMapParams, LocalRegion, GlobalRegion
from src.models.gaussian_process import GaussianProcess
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv, DynamicSmokeParams
from agents.basic_robot import RobotParams
from simulator.sensor import DownwardsSensorParams, GlobalSensorParams
from simulator.dynamic_smoke import SmokeBlobParams

class MPPIExperiment(BaseExperiment):
    def setup(self):
        # 1. Configure Parameters from Hydra
        # Convert DictConfig to specific Params objects
        # TODO: Ideally, constructors should accept dictionaries or kwargs
        
        # Env Params
        self.env_params = EnvParams()
        self.env_params.world_x_size = self.cfg.env.world_x_size
        self.env_params.world_y_size = self.cfg.env.world_y_size
        self.env_params.max_steps = self.cfg.env.max_steps
        self.env_params.clock = self.cfg.env.clock
        self.env_params.render = self.cfg.env.render
        self.env_params.goal_location = np.array(self.cfg.env.goal_location)
        self.env_params.goal_radius = self.cfg.env.goal_radius

        # Sensor Params
        if self.cfg.env.sensor.type == 'downwards':
            self.sensor_params = DownwardsSensorParams(
                world_x_size=self.env_params.world_x_size,
                world_y_size=self.env_params.world_y_size,
                points_in_range=self.cfg.env.sensor.points_in_range,
                fov_size_degrees=self.cfg.env.sensor.fov_size_degrees
            )
        else:
            self.sensor_params = GlobalSensorParams(
                world_x_size=self.env_params.world_x_size,
                world_y_size=self.env_params.world_y_size
            )
        self.env_params.sensor_params = self.sensor_params

        # Smoke Params
        blobs = [SmokeBlobParams(x_pos=b.x, y_pos=b.y, intensity=b.intensity, spread_rate=b.spread) 
                 for b in self.cfg.env.smoke.blobs]
        
        self.smoke_params = DynamicSmokeParams(
            x_size=self.env_params.world_x_size, 
            y_size=self.env_params.world_y_size, 
            smoke_blob_params=blobs, 
            resolution=self.cfg.env.smoke.resolution
        )

        # Robot Params
        self.robot_params = RobotParams() # TODO: Load from cfg.agent
        self.robot_params.world_x_size = self.env_params.world_x_size
        self.robot_params.world_y_size = self.env_params.world_y_size
        # Overwrite defaults with config
        self.robot_params.action_min = np.array(self.cfg.agent.action_min)
        self.robot_params.action_max = np.array(self.cfg.agent.action_max)
        self.robot_params.dt = self.cfg.agent.dt

        # 2. Initialize Environment
        self.env = DynamicSmokeEnv(self.env_params, self.robot_params, self.smoke_params)
        
        # Initial state
        initial_loc = self.get_initial_location()
        self.state, _ = self.env.reset(initial_state={
            "location": initial_loc, 
            "angle": 0.0, 
            "smoke_density": 0.0
        })

        # 3. Initialize Models (GP & RiskMap)
        # TODO: Move this to a model abstraction
        num_points = self.env.sensor.grid_pairs_positions.shape[0]
        history_size = self.cfg.experiment.time_horizon * num_points
        self.learner = GaussianProcess(online=False, history_size=history_size)

        max_inference_range = self.env_params.clock * self.cfg.experiment.time_horizon * self.robot_params.action_max[0]
        
        # Risk Map Builder
        # TODO: Configure inference region from hydra
        inference_region = LocalRegion(
            range_bound=(max_inference_range, max_inference_range), 
            resolution=0.5
        )
        self.builder = RiskMapBuilder(params=RiskMapParams(
            inference_region=inference_region, 
            map_rule_type='cvar' # TODO: Parameterize
        ))

        # 4. Initialize Controller (MPPI)
        mppi_params = MPPIControlParams(horizon=self.cfg.experiment.time_horizon)
        self.controller = MPPIControlDyn(
            self.robot_params, 
            self.cfg.agent.name, 
            0.6, # discrete_resolution TODO: Parameterize
            mppi_params=mppi_params, 
            goal_thresh=self.env_params.goal_radius
        )
        self.controller.set_goal(list(self.env_params.goal_location))

        # 5. Setup Renderer
        # Create options dictionary compatible with StandardRenderer
        render_opts = {
            'env_params': self.env_params,
            'robot_params': self.robot_params,
            'smoke_params': self.smoke_params,
            'render': self.cfg.env.render,
            'inference': 'local', # TODO
            'seq_filepath': str(self.output_dir / "video.mp4")
        }
        self.setup_renderer(render_opts)

    def run_episode(self):
        finished = False
        
        # Main loop
        for t in tqdm(range(self.env_params.max_steps + 1)):
            if finished:
                break

            # 1. Update Model (GP)
            with self.time_tracker.track("model_update"):
                x_input = np.concatenate([
                    self.state["smoke_density_location"], 
                    np.full((self.state["smoke_density_location"].shape[0], 1), t * self.env_params.clock)
                ], axis=1)
                self.learner.track_data(x_input, self.state["smoke_density"])
                self.learner.update()

            # 2. Prediction (Risk Map)
            predicted_maps = deque(maxlen=self.cfg.experiment.time_horizon)
            with self.time_tracker.track("prediction"):
                for i in range(self.cfg.experiment.time_horizon):
                    self.builder.build_map(
                        self.learner, 
                        self.state["location"][0], 
                        self.state["location"][1], 
                        (t + i + 1) * self.env_params.clock
                    )
                    coords = self.builder.map_assets['coords_map']
                    risk = np.clip(self.builder.map_assets['risk_map'], 0, 1)
                    predicted_maps.append((coords, risk))

            # 3. Control (MPPI)
            with self.time_tracker.track("control"):
                self.controller.set_state(np.array([
                    self.state["location"][0], 
                    self.state["location"][1], 
                    self.state["angle"]
                ]))
                self.controller.set_maps(predicted_maps)
                nominal_action = self.controller.get_command()
                action_input = nominal_action.numpy()

            # 4. Environment Step
            with self.time_tracker.track("env"):
                next_state, reward, terminated, truncated, _ = self.env.step(np.array(action_input))
            
            # 5. Logging & Metrics
            finished = self.check_termination(self.state, reward, terminated, truncated)
            self.log_common_metrics(self.state, action_input, t)
            
            # 6. Rendering
            if self.renderer:
                info = {
                    'env': self.env,
                    'builder': self.builder,
                    'nom_controller': self.controller,
                    'logger_metrics': self.metrics,
                    'state': self.state,
                    'action_input': action_input,
                    'nominal_action': nominal_action,
                    'predicted_maps': predicted_maps,
                    'path': self.metrics.get_values('traj_path'),
                }
                self.renderer.render(info=info)
                self.renderer.save_frame()

            self.state = next_state

        return {"status": "completed"}

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    # Hydra changes the default working directory, which sometimes breaks relative imports.
    # We will print the current directory for debug.
    print(f"Working directory: {os.getcwd()}")
    experiment = MPPIExperiment(cfg)
    experiment.run()

if __name__ == "__main__":
    main()
