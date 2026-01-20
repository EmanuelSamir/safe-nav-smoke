from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
from src.mppi_control_dyn import MPPIControlDyn, MPPIControlParams
from src.risk_map_builder import *
from learning.gaussian_process import GaussianProcess
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv
from agents.basic_robot import RobotParams
from agents.dubins_robot import DubinsRobot
from simulator.sensor import DownwardsSensorParams, GlobalSensorParams
from simulator.static_smoke import SmokeBlobParams
from skimage.transform import resize
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import FancyArrow, Arrow, Circle, Polygon
from src.utils import *
from src.cbf_hj import CBFHJ
from functools import wraps
from time import time, sleep, perf_counter, strftime
import imageio.v2 as imageio
from io import BytesIO
from src.time_tracker import TimeTracker
import pandas as pd
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
import json
from dataclasses import is_dataclass, asdict

from learning.vaessm import *
from learning.base_model import BaseModel
# ===============================================================================
# IMPORTS CONSOLIDADOS - Fase 1 Refactoring
# ===============================================================================
from src.utils import LoggerMetrics, dataclass_json_dump
from src.visualization import Renderer
# ===============================================================================


# class LoggerMetrics:
#     def __init__(self):
#         self.values = defaultdict(list)

#     def add_value(self, key: str, value: float):
#         self.values[key].append(value)

#     def get_last_value(self, key: str):
#         if self.values.get(key) is None:
#             return 0.0
#         return self.values[key][-1]

#     def get_values(self, key: str = None):
#         if key is None:
#             return self.values
#         return self.values.get(key, [])

#     def dump_to_csv(self, filepath: str):
#         df = pd.DataFrame(self.values)
#         df.to_csv(filepath, index=False)
    
#     def reset(self):
#         self.values.clear()

# def dataclass_json_dump(obj, path):
#     def convert(x):

        # ----- Dataclass -----
        if is_dataclass(x):
            return convert(asdict(x))

        # ----- Numpy array -----
        if isinstance(x, np.ndarray):
            return x.tolist()

        # ----- Numpy scalar (np.float32, np.int64, etc.) -----
        if isinstance(x, (np.generic,)):
            return x.item()

        # ----- Dict -----
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}

        # ----- List / Tuple -----
        if isinstance(x, (list, tuple)):
            return [convert(i) for i in x]

        # ----- Tipos JSON válidos -----
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x

        # ----- Clases no dataclass -----
        # NO convertirlas, solo representarlas como string para evitar crash.
        return str(x)

    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=4)

# class Renderer:
#     def __init__(self, opts: dict):
#         self.opts = opts

#         self.frames = []

#         self.fig = None
#         self.axes = None

#         self.cmap = plt.colormaps["RdYlGn"]

#         self.reset()

#         self.cbar_map = {}

#     def render(self, info: dict):
#         if not self.opts.get('render'):
#             return
#         assert self.fig is not None and self.axes is not None, "Figure and axes must be initialized"

#         state = info.get('state')
#         location = info.get('state')['location']
#         angle = info.get('state')['angle']
#         smoke_density = info.get('state')['smoke_density']
#         smoke_density_location = info.get('state')['smoke_density_location']
#         predicted_maps = info.get('predicted_maps')
#         env = info.get('env')
#         builder = info.get('builder')

#         self.flush_decoratives(self.axes['pred'])

#         env._render_frame(fig=self.fig, ax=self.axes['env'])
#         builder.plot_map(risk_map=predicted_maps[0][1], x_robot_pos=location[0], y_robot_pos=location[1], fig=self.fig, ax=self.axes['pred'])

#         im = self.axes['env'].images[0]
#         if self.cbar_map.get('env') is None:
#             self.cbar_map['env'] = self.fig.colorbar(im, ax=self.axes['env'], orientation='horizontal', location='bottom', pad=0.05)
#             self.cbar_map['env'].set_label("Smoke Density", fontsize=12)

#         im = self.axes['pred'].images[0]
#         if self.cbar_map.get('pred') is None:
#             self.cbar_map['pred'] = self.fig.colorbar(im, ax=self.axes['pred'], orientation='horizontal', location='bottom', pad=0.05)
#             self.cbar_map['pred'].set_label("Predicted Risk Map", fontsize=12)

#         self.add_decoratives(self.axes['pred'], info)

#         self.plot_smoke(self.axes['risk'], info)
#         # self.plot_dist_error(self.axes['dist_to_goal'], info)
#         # self.plot_angle_err(self.axes['angle_err'], info)
#         self.plot_velocity(self.axes['velocity'], info)
#         self.plot_angular_velocity(self.axes['angular_velocity'], info)

#         plt.pause(self.opts.get('env_params').clock)
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()

#     def add_decoratives(self, ax, info: dict):
#         location = info.get('state')['location']
#         angle = info.get('state')['angle']
#         path = info.get('path')
#         env = info.get('env')
#         nom_controller = info.get('nom_controller')

#         # Draw the robot's pose as an arrow
#         ax.arrow(location[0], location[1], np.cos(angle)*0.1, np.sin(angle)*0.1, 
#                                 head_width=0.8, head_length=0.8, fc='blue', ec='blue', zorder=10)

#         if len(path) > 0:
#             ax.scatter(np.array(path)[:, 0], np.array(path)[:, 1], color='green', label='Nominal Path', marker='.', s=0.4, zorder=10)

#         # Plot the goal location
#         if self.opts.get('inference') == 'global':
#             circle = Circle((env.env_params.goal_location[0], env.env_params.goal_location[1]), radius=env.env_params.goal_radius, color='g', fill=True, alpha=0.8)
#             (xc, yc), r = circle.center, circle.radius

#             ax.add_patch(circle)
#             ax.text(xc, yc - r - env.env_params.goal_radius - 0.5, "goal", ha="center", va="bottom",
#                     fontsize=10, color="green", zorder=10)

#         square = env.sensor.projection_bounds(location[0], location[1])

#         bounded_square = np.array([clip_world(p[0], p[1], env.env_params.world_x_size, env.env_params.world_y_size) for p in square])
#         ax.add_patch(Polygon(bounded_square, facecolor='none', edgecolor='blue', linewidth=2,))

#         trajectories = nom_controller.get_sampled_trajectories()
#         if trajectories is not None:
#             omega = nom_controller.planner.omega.detach().cpu()
#             weighted_traj = (omega[:, None, None] * trajectories).sum(dim=0).numpy()
#             omega_norm = (omega - omega.min()) / (omega.max() - omega.min() + 1e-9)
#             sorted_indices = np.argsort(omega_norm)
#             # for i, traj in enumerate(trajectories[sorted_indices]):
#             #     ax.plot(traj[:, 0], traj[:, 1], color='purple', alpha=0.4, linewidth=1.0)
#             ax.plot(weighted_traj[:, 0], weighted_traj[:, 1], color="purple", linewidth=2)

#     def plot_smoke(self, ax, info):
#         smoke_on_robot = info.get('logger_metrics').get_values('smoke_on_robot')
#         smoke_on_robot_acc = info.get('logger_metrics').get_values('smoke_on_robot_acc')
#         steps = info.get('logger_metrics').get_values('steps')

#         if not len(ax.lines):
#             ax.set_xlim(0, self.opts.get('env_params').max_steps)
#             ax.plot(steps, smoke_on_robot, color='red', label='smoke')
#             # ax.plot(steps, smoke_on_robot_acc, color='blue', label='acc smoke', alpha=0.5)
#             # ax.set_title('Smoke on Robot')
#             # ax.set_xlabel('Time step')
#             ax.set_ylabel('Smoke at Robot Pos.')
#             ax.grid(True)
#             # ax.legend()

#         else:
#             ax.lines[0].set_data(steps, smoke_on_robot)
#             # ax.lines[1].set_data(steps, smoke_on_robot_acc)
#             # ax.set_ylim(np.min(smoke_on_robot), np.max(smoke_on_robot_acc))
#             ax.set_ylim(0, 1)
        
#         self.fig.tight_layout()

#     def plot_dist_error(self, ax, info):
#         dist_error = info.get('logger_metrics').get_values('dist_to_goal')
#         steps = info.get('logger_metrics').get_values('steps')
#         if not len(ax.lines):
#             ax.plot(steps, dist_error, color='green', label='dist error')
#             ax.set_xlim(0, self.opts.get('env_params').max_steps)
#             ax.set_ylim(0, np.max(dist_error))
#             ax.set_xlabel('Time step')
#             ax.set_ylabel('Distance error')
#             ax.grid(True)
        
#         else:
#             ax.lines[0].set_data(steps, dist_error)
#         self.fig.tight_layout()

#     def plot_velocity(self, ax, info):
#         velocity = info.get('logger_metrics').get_values('action_0')
#         steps = info.get('logger_metrics').get_values('steps')
#         if not len(ax.lines):
#             ax.plot(steps, velocity, color='orange', label='velocity')
#             # ax.set_xlabel('Time step')
#             ax.set_ylabel('Linear velocity')
#             ax.set_xlim(0, self.opts.get('env_params').max_steps)

#             ax.grid(True)
#         else:
#             ax.lines[0].set_data(steps, velocity)
#             ax.set_ylim(info.get('env').robot_params.action_min[0], info.get('env').robot_params.action_max[0])
#         self.fig.tight_layout()

#     def plot_angular_velocity(self, ax, info):
#         angular_velocity = info.get('logger_metrics').get_values('action_1')
#         steps = info.get('logger_metrics').get_values('steps')
#         if not len(ax.lines):
#             ax.plot(steps, angular_velocity, color='purple', label='angular velocity')
#             ax.set_xlabel('Time step')
#             ax.set_ylabel('Angular velocity')
#             ax.set_xlim(0, self.opts.get('env_params').max_steps)
#             ax.grid(True)
#         else:
#             ax.lines[0].set_data(steps, angular_velocity)
#             ax.set_ylim(info.get('env').robot_params.action_min[1], info.get('env').robot_params.action_max[1])
#         self.fig.tight_layout()

#     def plot_angle_err(self, ax, info):
#         angle_err = info.get('logger_metrics').get_values('angle_err')
#         steps = info.get('logger_metrics').get_values('steps')
#         if not len(ax.lines):
#             ax.set_xlim(0, self.opts.get('env_params').max_steps)
#             ax.set_ylim(0, np.pi)
#             ax.plot(steps, angle_err, color='purple', label='angle err')
#             # ax.set_xlabel('Time step')
#             ax.set_ylabel('Angle error')
#             ax.grid(True)
#         else:
#             ax.lines[0].set_data(steps, angle_err)

#         self.fig.tight_layout()

#     def flush_decoratives(self, ax: plt.Axes):
#         for patch in ax.patches:
#             if isinstance(patch, (FancyArrow, Arrow, Polygon)):  
#                 patch.remove()

#         for patch in (ax.collections + ax.lines):
#             patch.remove()

#     def save_frame(self):
#         if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
#             buf = BytesIO()
#             self.fig.savefig(buf, format='png')
#             buf.seek(0)
#             image = imageio.imread(buf)
#             self.frames.append(image)

#     def reset(self):
#         if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
#             if self.frames:
#                 time_clock_vis = 0.5
#                 imageio.mimsave(f'{self.opts["seq_filepath"]}', self.frames, duration=time_clock_vis, loop=0)
            
#                 fps = 1 / 0.1   # igual que tu duration=0.5 → 2 fps
#                 filepath = self.opts["seq_filepath"].replace(".gif", ".mp4")
#                 target_w, target_h = 1920, 1080   # 1080p exacto


#                 import cv2
#                 writer = imageio.get_writer(
#                     filepath,
#                     fps=10,                   # o tu fps = 1/duration
#                     codec='libx264',
#                     quality=10,
#                     pixelformat='yuv420p',
#                     macro_block_size=16      # requerido para compatibilidad total
#                 )

#                 for frame in self.frames:
#                     h, w = frame.shape[:2]

#                     # --- Fondo blanco 1080p ---
#                     canvas = np.ones((target_h, target_w, 4), dtype=np.uint8) * 255

#                     # --- Escalar manteniendo aspect ratio ---
#                     scale = min(target_w / w, target_h / h)
#                     new_w = int(w * scale)
#                     new_h = int(h * scale)

#                     resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

#                     # --- Centrar ---
#                     x0 = (target_w - new_w) // 2
#                     y0 = (target_h - new_h) // 2

#                     canvas[y0:y0+new_h, x0:x0+new_w] = resized

#                     writer.append_data(canvas)

#                 writer.close()
#             else:
#                 print(f"No frames to save for {self.opts['seq_filepath']}")

#         # Clean up and store
#         self.fig = plt.figure(figsize=(12, 5))
#         gs = self.fig.add_gridspec(3, 3, height_ratios=[0.4, 0.4, 0.4], width_ratios=[1.2, 0.6, 1.2])

#         self.axes = {}
#         self.axes['env'] = self.fig.add_subplot(gs[0:, 0])
#         self.axes['risk'] = self.fig.add_subplot(gs[0, 1])
#         # self.axes['angle_err'] = seilf.fig.add_subplot(gs[1, 1])
#         # self.axes['dist_to_goal'] = self.fig.add_subplot(gs[2, 1])

#         self.axes['velocity'] = self.fig.add_subplot(gs[1, 1])
#         self.axes['angular_velocity'] = self.fig.add_subplot(gs[2, 1])
#         self.axes['pred'] = self.fig.add_subplot(gs[0:, 2])

#         for ax in [self.axes['env'], self.axes['pred']]:
#             ax.set_xlim(0, self.opts.get('env_params').world_x_size)
#             ax.set_ylim(0, self.opts.get('env_params').world_y_size)
#             ax.autoscale(False)

#         for ax in [self.axes['risk'], self.axes['velocity'], self.axes['angular_velocity']]:
#             ax.set_xlim(0, self.opts.get('env_params').max_steps)
#             ax.autoscale(tight=True)

#         for ax in self.axes.values():
#             ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#         self.fig.tight_layout()

# class VAESSMWrapper(BaseModel):
#     def __init__(self, checkpoint_path: str):
#         self.model = ScalarFieldVAESSM(params=VAESSMParams())
#         state_dict = torch.load(checkpoint_path)
#         self.model.load_state_dict(state_dict)
#         self.model.eval()
#         self.prev_h = None
#         self.prev_z = None

#     def forward(self, prev_h, prev_z, action_step, pos_obs, value_obs = None):
#         if value_obs is None:
#             # print("Enters to do prediction")
#             obs_step = ObsVAESSM(
#                 xs=torch.tensor(pos_obs[:, 0]).unsqueeze(0),
#                 ys=torch.tensor(pos_obs[:, 1]).unsqueeze(0),
#             )
#             h, z, prior_dist, post_dist, pred_dist = self.model(
#                 prev_h=prev_h,
#                 prev_z=prev_z,
#                 prev_action=action_step,
#                 dones=torch.tensor([False]),
#                 query_obs=obs_step
#             )

#         else:
#             # print("Enters to do one step inference")
#             obs_step = ObsVAESSM(
#                 xs=torch.tensor(pos_obs[:, 0]).unsqueeze(0),
#                 ys=torch.tensor(pos_obs[:, 1]).unsqueeze(0),
#                 values=torch.tensor(value_obs).squeeze().unsqueeze(0),
#             )
#             h, z, prior_dist, post_dist, pred_dist = self.model(
#                 prev_h=prev_h,
#                 prev_z=prev_z,
#                 prev_action=action_step,
#                 dones=torch.tensor([False]),
#                 obs=obs_step,
#             )

#         return h, z, prior_dist, post_dist, pred_dist

#     def set_prev_states(self, h, z):
#         self.prev_h = h
#         self.prev_z = z

#     def predict(self, pos_obs):
#         action_vec = torch.zeros(1, self.model.params.action_dim)

#         h, z, prior_dist, post_dist, pred_dist = self.forward(self.prev_h, self.prev_z, action_vec, pos_obs)
#         mean = pred_dist.mean.detach().numpy()
#         std = pred_dist.stddev.detach().numpy()
#         self.set_prev_states(h, z)
#         return mean, std

#     def update(self):
#         pass

def main(opts: dict = {}):
    env_params = EnvParams.load_from_yaml("envs/env_cfg.yaml")
    discrete_resolution = 0.9
    sim_resolution = 0.4
    time_horizon = 10

    robot_type = "dubins2d"
    cfg_file = f"agents/{robot_type}_cfg.yaml"

    goal_location_samples = [
        [25, 7],
        [25, 13],
    ]
    
    initial_location_samples = [
        [1, 4],
        [1, 10],
        [1, 16],
    ]

    initial_location = initial_location_samples[np.random.randint(0, len(initial_location_samples))]


    # ==== SMOKE SIMULATOR SETUP ====
    smoke_blob_params = [
        SmokeBlobParams(x_pos=9, y_pos=16, intensity=1.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=9, y_pos=10, intensity=1.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=9, y_pos=3, intensity=1.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=17, y_pos=13, intensity=1.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=17, y_pos=7, intensity=1.0, spread_rate=1.0),
    ]

    if opts.get('sensor') == 'downwards':
        sensor_params = DownwardsSensorParams(world_x_size=env_params.world_x_size,world_y_size=env_params.world_y_size)
    elif opts.get('sensor') == 'global':
        sensor_params = GlobalSensorParams(world_x_size=env_params.world_x_size,world_y_size=env_params.world_y_size)
    else:
        raise ValueError(f"Invalid sensor: {opts.get('sensor')}")

    smoke_params = DynamicSmokeParams(x_size=env_params.world_x_size, y_size=env_params.world_y_size, smoke_blob_params=smoke_blob_params, resolution=sim_resolution)

    # ==== ROBOT SETUP ====
    robot_params = RobotParams.load_from_yaml(cfg_file)
    robot_params.world_x_size = env_params.world_x_size
    robot_params.world_y_size = env_params.world_y_size

    # ==== ENVIRONMENT SETUP ====
    env_params.goal_location = goal_location_samples[np.random.randint(0, len(goal_location_samples))]
    env_params.render = opts.get("render")
    env_params.sensor_params = sensor_params
    env = DynamicSmokeEnv(env_params, robot_params, smoke_params)

    state, _ = env.reset(initial_state={"location": np.array(initial_location), "angle": 0.0, "smoke_density": 0.0})

    # ==== learner SETUP ====
    num_points_in_inference_region = env.sensor.grid_pairs_positions.shape[0]
    learner = VAESSMWrapper(checkpoint_path="/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/vaessm_3rd.pt")

    max_inference_range = env.clock * time_horizon * robot_params.action_max[0]

    # ==== failure map builder SETUP ====
    if opts.get('inference') == 'global':
        inference_region = GlobalRegion(world_size=(env_params.world_x_size, env_params.world_y_size), resolution=discrete_resolution)
    elif opts.get('inference') == 'local':
        inference_region = LocalRegion(range_bound=(max_inference_range, max_inference_range), resolution=0.5)
    else:
        raise ValueError(f"Invalid inference region: {opts.get('inference')}")

    builder = RiskMapBuilder(params=RiskMapParams(inference_region=inference_region, map_rule_type=opts.get('map_rule_type')))

    # ==== MPPI CONTROL SETUP ====
    mppi_params = MPPIControlParams(horizon=time_horizon)
    nom_controller = MPPIControlDyn(robot_params, robot_type, discrete_resolution, mppi_params=mppi_params, goal_thresh=env_params.goal_radius)
    nom_controller.set_state(np.array([state["location"][0], state["location"][1], state["angle"]]))
    nom_controller.set_goal(list(env_params.goal_location))

    opts['env_params'] = env_params
    opts['robot_params'] = robot_params
    opts['smoke_params'] = smoke_params

    renderer = Renderer(opts=opts)
    
    logger_metrics = LoggerMetrics()

    # Placeholders
    summary = {"reached_goal": False, "crashed": False, "truncated": False}

    time_tracker = TimeTracker()

    finished = False

    prev_h = torch.zeros(1, learner.model.params.deter_dim)
    prev_z = torch.zeros(1, learner.model.params.stoch_dim)
    prev_action = torch.zeros(1, learner.model.params.action_dim)

    # Create a line plot for the gradient values
    for t in tqdm(range(0,env_params.max_steps+1)):
        if finished:
            break

        with time_tracker.track("learning_posterior"):
            locs = torch.tensor(state["smoke_density_location"], dtype=torch.float32)
            robot_pose = torch.tensor([state["location"][0], state["location"][1]], dtype=torch.float32)
            norm_locs = locs - robot_pose

            print("norm_locs min", norm_locs.min())
            print("norm_locs max", norm_locs.max())
            h, z, _, _, _ = learner.forward(prev_h, prev_z, prev_action, norm_locs, torch.tensor(state["smoke_density"], dtype=torch.float32))
        
        with time_tracker.track("learning_prior"):
            learner.set_prev_states(prev_h, prev_z)
            predicted_maps = deque(maxlen=time_horizon)
            for i in range(time_horizon):
                builder.build_map(learner, state["location"][0], state["location"][1])
                coords, flatten_risk_map = builder.map_assets['coords_map'], builder.map_assets['risk_map']
                # flatten_risk_map = np.clip(flatten_risk_map, 0, 1)
                predicted_maps.append((coords, flatten_risk_map))
                
        prev_h = h
        prev_z = z

        with time_tracker.track("mppi"):
            nom_controller.set_state(np.array([state["location"][0], state["location"][1], state["angle"]]))
            if not opts.get('disable_risk_map', False):
                nom_controller.set_maps(predicted_maps)
            nominal_action = nom_controller.get_command()
            action_input = nominal_action.numpy()

        with time_tracker.track("env"):
            state, reward, terminated, truncated, env_info = env.step(np.array(action_input))
            v_linear, w = action_input
            v_x = v_linear * np.cos(state["angle"])
            v_y = v_linear * np.sin(state["angle"])
            prev_action = torch.tensor([v_x, v_y], dtype=torch.float32).unsqueeze(0)

        if terminated or truncated:
            finished = True
            if terminated and reward == 1.0:
                logger_metrics.add_value('status', 'reached_goal')
            elif terminated and reward == 0.0:
                logger_metrics.add_value('status', 'crashed')
            else:
                logger_metrics.add_value('status', 'truncated')
        else:
            logger_metrics.add_value('status', 'running')

        smoke_on_robot = env.get_smoke_density_in_robot()[0]
        dist_to_goal = np.linalg.norm(state["location"] - env_params.goal_location)

        desired_angle = np.arctan2(
            env_params.goal_location[1] - state["location"][1],
            env_params.goal_location[0] - state["location"][0]
        )

        angle_err = np.abs(np.arctan2(
            np.sin(state["angle"] - desired_angle),
            np.cos(state["angle"] - desired_angle)
        ))

        logger_metrics.add_value('smoke_on_robot', smoke_on_robot)
        last_smoke_acc = logger_metrics.get_last_value('smoke_on_robot_acc')

        smoke_on_robot_acc = last_smoke_acc + smoke_on_robot
        logger_metrics.add_value('traj_path', state["location"])
        logger_metrics.add_value('smoke_on_robot_acc', smoke_on_robot_acc)

        logger_metrics.add_value('dist_to_goal', dist_to_goal)
        logger_metrics.add_value('angle_err', angle_err)
        logger_metrics.add_value('time', t*env_params.clock)
        logger_metrics.add_value('steps', t)

        for i, action_item in enumerate(action_input):
            logger_metrics.add_value(f'action_{i}', action_item)

        for i, state_item in enumerate(list(state["location"]) + [state["angle"]]):
            logger_metrics.add_value(f'state_{i}', state_item)

        info = {
            'env': env,
            'builder': builder,
            'nom_controller': nom_controller,
            'logger_metrics': logger_metrics,
            'state': state,
            'action_input': action_input,
            'nominal_action': nominal_action,
            'predicted_maps': predicted_maps,
            'path': logger_metrics.get_values('traj_path'),
        }

        renderer.render(info=info)
        renderer.save_frame()
    
    dataclass_json_dump(opts, f'{opts["run_fp"]}/options.json')

    renderer.reset()
    logger_metrics.dump_to_csv(f'{opts["run_fp"]}/metrics.csv')

    time_df = pd.DataFrame(time_tracker.as_dict())
    time_df.to_csv(f'{opts["run_fp"]}/time.csv', index=False)

    env.close()



if __name__ == "__main__":

    all_opts_experiments = []

    # Test: local and downwards. cvar Performance 50 steps
    # Test: local and global. cvar Time and performance 20 steps
    # Test: global and global. cvar Time 5 steps

    # Test: local and downwards. mean Performance 50 steps
    # Test: Discounted cost?

    data_fp = f'misc/'

    render = "human"

    all_opts_experiments.append({
        'inference': 'local',
        'sensor': 'downwards',
        'map_rule_type': 'mean',
        'render': render,
        'samples': 20,
        'disable_risk_map': False
    })

    for i, opts in enumerate(all_opts_experiments):
        experiment_name = f'vaessm_{i+1}'
        experiment_name = f'{experiment_name}_{strftime("%m%d_%H%M")}'
        experiment_fp = f'{data_fp}/{experiment_name}'
        os.makedirs(experiment_fp, exist_ok=True)

        for j in tqdm(range(opts['samples'])):
            print('--------------------------------')
            print(f'Running sample {j + 1} of {opts['samples']}')
            run_fp = f'{experiment_fp}/run_{j + 1}'
            os.makedirs(run_fp, exist_ok=True)

            opts['experiment_fp'] = experiment_fp
            opts['run_fp'] = run_fp
            opts['seq_filepath'] = f'{run_fp}/result.gif'

            metrics = main(opts=opts)
