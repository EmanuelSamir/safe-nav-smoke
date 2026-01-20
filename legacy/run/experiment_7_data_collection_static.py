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
from src.cbf_controller import CBFController
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
from src.utils import LoggerMetrics, dataclass_json_dump
from src.visualization import StandardRenderer as SimpleRenderer
from envs.replay_buffer import GenericReplayBuffer

from collections import deque, OrderedDict

def main(opts: dict = {}):
    num_collection_data = 1e5 / 5

    env_params = EnvParams.load_from_yaml("envs/env_cfg.yaml")
    discrete_resolution = 0.6
    sim_resolution = 0.4
    time_horizon = 10

    robot_type = "dubins2d"
    cfg_file = f"agents/{robot_type}_cfg.yaml"

    sensor_params = DownwardsSensorParams(
        world_x_size=env_params.world_x_size,
        world_y_size=env_params.world_y_size,
        points_in_range=30,
        fov_size_degrees=50
    )
    num_smoke_blobs = 6

    # ==== ROBOT SETUP ====
    robot_params = RobotParams.load_from_yaml(cfg_file)
    robot_params.world_x_size = env_params.world_x_size
    robot_params.world_y_size = env_params.world_y_size

    # ==== ENVIRONMENT SETUP ====
    env_params.goal_location = None
    env_params.smoke_density_threshold = None
    env_params.render = opts.get("render")
    env_params.sensor_params = sensor_params

    # ==== DATA COLLECTION ====
    replay_buffer = GenericReplayBuffer(buffer_size=num_collection_data, data_keys = ['actions', 'state', 'smoke_values', 'smoke_value_positions', 'done'])

    datasize = 0
    with tqdm(total=num_collection_data) as p:
        while datasize < num_collection_data:
            initial_location = np.array([env_params.world_x_size * np.random.rand(), env_params.world_y_size * np.random.rand()])
            
            smoke_blob_params = [
                SmokeBlobParams(
                    x_pos= env_params.world_x_size * np.random.rand(), 
                    y_pos= env_params.world_y_size * np.random.rand(), 
                    intensity=1.0, spread_rate=1.0)
                for _ in range(num_smoke_blobs)
            ]

            smoke_params = DynamicSmokeParams(
                x_size=env_params.world_x_size, y_size=env_params.world_y_size, 
                smoke_blob_params=smoke_blob_params, resolution=sim_resolution
                )

            env = DynamicSmokeEnv(env_params, robot_params, smoke_params)

            state, _ = env.reset(initial_state={"location": np.array(initial_location), "angle": 0.0, "smoke_density": 0.0})

            finished = False

            waypoint = np.array([env_params.world_x_size * np.random.rand(), env_params.world_y_size * np.random.rand()])
            cbf = CBFController(env_params, robot_params, waypoint)

            while not finished:
                while np.linalg.norm(waypoint - state["location"]) < env_params.goal_radius:
                    waypoint = np.array([env_params.world_x_size * np.random.rand(), env_params.world_y_size * np.random.rand()])
                    cbf.goal = waypoint

                pose = np.array([state["location"][0], state["location"][1], state["angle"]])
                action_input = cbf.nominal_control(pose)

                # Trick for data balance in zero specially that is where I need more data

                if np.random.rand() < 0.3:
                    # Multiplicamos por un número aleatorio pequeño (entre 0 y 0.5)
                    action_weight_random = np.random.uniform(0, 0.5)
                    action_input = action_input * action_weight_random

                # action_weight_temporal = np.abs(np.sin(2 * np.pi / 5.0 *  (env.current_step * env_params.clock)))
                # action_input = action_input * action_weight_temporal

                state, reward, terminated, truncated, env_info = env.step(np.array(action_input))
                replay_buffer.add(
                    state = pose,
                    actions = action_input,
                    smoke_values = state['smoke_density'],
                    smoke_value_positions = state['smoke_density_location'],
                    done = truncated
                    )

                datasize += 1
                p.update(1)

                if truncated or datasize >= num_collection_data:
                    finished = True

            env.close()

    replay_buffer.save_to_file(f'tmp.npz')


    replay_buffer.save_to_file(f'{opts["experiment_fp"]}/replay_buffer.npz')



if __name__ == "__main__":

    all_opts_experiments = []

    data_fp = f'misc/'

    render = "rgb_array"

    all_opts_experiments.append({
        'sensor': 'downwards',
        'render': render,
        'samples': 100,
        'smoke_threshold': 0.25,
    })

    for i, opts in enumerate(all_opts_experiments):
        experiment_name = f'collection_data_static_{i+1}'
        experiment_name = f'{experiment_name}_{strftime("%m%d_%H%M")}'
        experiment_fp = f'{data_fp}/{experiment_name}'
        os.makedirs(experiment_fp, exist_ok=True)

        opts['experiment_fp'] = experiment_fp

        main(opts=opts)
