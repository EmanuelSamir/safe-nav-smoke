import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.experiments.base_experiment import BaseExperiment
from src.cbf_controller import CBFController
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv, DynamicSmokeParams
from agents.basic_robot import RobotParams
from simulator.sensor import DownwardsSensorParams, GlobalSensorParams
from simulator.static_smoke import SmokeBlobParams

class CBFExperiment(BaseExperiment):
    def setup(self):
        # 1. Configurar Parámetros desde Hydra (Similar a MPPIExperiment)
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
        self.robot_params = RobotParams()
        self.robot_params.world_x_size = self.env_params.world_x_size
        self.robot_params.world_y_size = self.env_params.world_y_size
        self.robot_params.action_min = np.array(self.cfg.agent.action_min)
        self.robot_params.action_max = np.array(self.cfg.agent.action_max)
        self.robot_params.dt = self.cfg.agent.dt

        # 2. Inicializar Entorno
        self.env = DynamicSmokeEnv(self.env_params, self.robot_params, self.smoke_params)
        
        initial_loc = self.get_initial_location()
        self.state, _ = self.env.reset(initial_state={
            "location": initial_loc, 
            "angle": 0.0, 
            "smoke_density": 0.0
        })

        # 3. Inicializar Controlador (CBF)
        self.controller = CBFController(
            self.env_params, 
            self.robot_params, 
            self.env_params.goal_location, 
            smoke_threshold=self.cfg.experiment.smoke_threshold
        )

        # 4. Setup Renderer
        render_opts = {
            'env_params': self.env_params,
            'robot_params': self.robot_params,
            'smoke_params': self.smoke_params,
            'render': self.cfg.env.render,
            'inference': 'global', # CBF usually assumes global knowledge or built map
            'seq_filepath': str(self.output_dir / "video.mp4")
        }
        self.setup_renderer(render_opts)

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
            if self.renderer:
                info = {
                    'env': self.env,
                    'logger_metrics': self.metrics,
                    'state': self.state,
                    'action_input': action_input,
                    'path': self.metrics.get_values('traj_path'),
                    # CBF doesn't have predicted_maps in the same way, but we could visualize h
                }
                self.renderer.render(info=info)
                self.renderer.save_frame()

            self.state = next_state

        return {"status": "completed"}
