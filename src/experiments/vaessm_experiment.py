import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from collections import deque

from src.experiments.base_experiment import BaseExperiment
from src.models.vaessm_wrapper import VAESSMWrapper
from src.mppi_control_dyn import MPPIControlDyn, MPPIControlParams
from src.risk_map_builder import RiskMapBuilder, RiskMapParams, LocalRegion, GlobalRegion
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv, DynamicSmokeParams
from agents.basic_robot import RobotParams
from simulator.sensor import DownwardsSensorParams, GlobalSensorParams
from simulator.static_smoke import SmokeBlobParams

class VAESSMExperiment(BaseExperiment):
    def setup(self):
        # 1. Configurar Parámetros (Igual que los otros)
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

        # 3. Inicializar Modelo (VAESSM)
        # Detectar dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # En Mac con MPS
        if torch.backends.mps.is_available():
            device = "mps"
            
        self.learner = VAESSMWrapper(
            checkpoint_path=self.cfg.experiment.checkpoint_path,
            device=device
        )

        # 4. Risk Map Builder
        max_inference_range = self.env_params.clock * self.cfg.experiment.time_horizon * self.robot_params.action_max[0]
        
        # TODO: Parametrizar inference region type
        inference_region = LocalRegion(
            range_bound=(max_inference_range, max_inference_range), 
            resolution=0.5
        )
        self.builder = RiskMapBuilder(params=RiskMapParams(
            inference_region=inference_region, 
            map_rule_type=self.cfg.experiment.map_rule_type
        ))

        # 5. Inicializar Controlador (MPPI)
        mppi_params = MPPIControlParams(horizon=self.cfg.experiment.time_horizon)
        self.controller = MPPIControlDyn(
            self.robot_params, 
            self.cfg.agent.name, 
            0.6, 
            mppi_params=mppi_params, 
            goal_thresh=self.env_params.goal_radius
        )
        self.controller.set_goal(list(self.env_params.goal_location))

        # 6. Setup Renderer
        render_opts = {
            'env_params': self.env_params,
            'robot_params': self.robot_params,
            'smoke_params': self.smoke_params,
            'render': self.cfg.env.render,
            'inference': 'local', 
            'seq_filepath': str(self.output_dir / "video.mp4")
        }
        self.setup_renderer(render_opts)

        # Estado interno del bucle VAESSM
        self.prev_h = torch.zeros(1, self.learner.model.params.deter_dim).to(device)
        self.prev_z = torch.zeros(1, self.learner.model.params.stoch_dim).to(device)
        self.prev_action = torch.zeros(1, self.learner.model.params.action_dim).to(device)


    def run_episode(self):
        finished = False
        
        for t in tqdm(range(self.env_params.max_steps + 1)):
            if finished:
                break

            # 1. Learning Posterior (Update with current observation)
            with self.time_tracker.track("learning_posterior"):
                locs = torch.tensor(self.state["smoke_density_location"], dtype=torch.float32)
                robot_pose = torch.tensor([self.state["location"][0], self.state["location"][1]], dtype=torch.float32)
                norm_locs = locs - robot_pose

                # Forward pass to get posterior h, z
                h, z, _, _, _ = self.learner.forward(
                    self.prev_h, 
                    self.prev_z, 
                    self.prev_action, 
                    norm_locs, 
                    torch.tensor(self.state["smoke_density"], dtype=torch.float32)
                )

            # 2. Learning Prior (Predict future maps)
            predicted_maps = deque(maxlen=self.cfg.experiment.time_horizon)
            with self.time_tracker.track("learning_prior"):
                # Reset learner state to current posterior for prediction rollout
                self.learner.set_prev_states(self.prev_h, self.prev_z)
                
                for i in range(self.cfg.experiment.time_horizon):
                    # build_map calls learner.predict() which advances internal state
                    self.builder.build_map(
                        self.learner, 
                        self.state["location"][0], 
                        self.state["location"][1]
                    )
                    coords = self.builder.map_assets['coords_map']
                    risk = self.builder.map_assets['risk_map']
                    predicted_maps.append((coords, risk))

            # Update states for next iteration (using posterior from step 1)
            self.prev_h = h
            self.prev_z = z

            # 3. Control (MPPI)
            with self.time_tracker.track("mppi"):
                self.controller.set_state(np.array([
                    self.state["location"][0], 
                    self.state["location"][1], 
                    self.state["angle"]
                ]))
                
                if not self.cfg.experiment.disable_risk_map:
                    self.controller.set_maps(predicted_maps)
                
                nominal_action = self.controller.get_command()
                action_input = nominal_action.numpy()

            # 4. Environment Step
            with self.time_tracker.track("env"):
                next_state, reward, terminated, truncated, _ = self.env.step(np.array(action_input))
                
                # Calculate action vector for next step (global frame velocity)
                v_linear, w = action_input
                v_x = v_linear * np.cos(self.state["angle"])
                v_y = v_linear * np.sin(self.state["angle"])
                self.prev_action = torch.tensor([v_x, v_y], dtype=torch.float32).unsqueeze(0).to(self.learner.device)

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
