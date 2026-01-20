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

from collections import deque, OrderedDict
from envs.replay_buffer import GenericReplayBuffer

# Script to store values in replay for GP inference
# t -> t+1, t+2 ... , t+1 -> t+2, t+3 ... , t+2 -> t+3, t+4 ...

def main(opts: dict = {}):
    env_params = EnvParams.load_from_yaml("envs/env_cfg.yaml")
    discrete_resolution = 0.6
    sim_resolution = 0.4
    time_horizon = 10

    robot_type = "dubins2d"
    cfg_file = f"agents/{robot_type}_cfg.yaml"

    sensor_params = DownwardsSensorParams(
        world_x_size=env_params.world_x_size,
        world_y_size=env_params.world_y_size,
        points_in_range=8,
        fov_size_degrees=15
    )

    # ==== DATA LOADING ====
    num_collection_data = 1e5
    replay_buffer = GenericReplayBuffer(buffer_size=num_collection_data, data_keys=[])
    replay_buffer.load_from_file(opts["replay_buffer_fp"])

    # ==== DATA COLLECTION OF NEW DATA ====
    keys_to_store = ['index', 'usable', 'smoke_value_positions']
    for t in range(time_horizon):
        keys_to_store.extend([f'smoke_values_pred_{t}', f'smoke_values_std_{t}'])

    captures = GenericReplayBuffer(buffer_size=num_collection_data, data_keys=keys_to_store)
    
    # ==== GP SETUP ====
    num_points_in_inference_region = sensor_params.points_in_range**2
    k = time_horizon * num_points_in_inference_region
    learner = GaussianProcess(online=False, history_size=k)

    datasize = replay_buffer.current_size

    learner.update()
    clock = 0.1

    usable = False
    usable_count = 0
    for t in tqdm(range(datasize)):
        t_env = t % 150
        data = replay_buffer.get_from_index(t)

        if data["done"]:
            usable = False
            usable_count = 0

        if usable_count < time_horizon:
            usable_count += 1
        else:
            usable = True

        x_input = np.concatenate([data["smoke_value_positions"], np.full((data["smoke_value_positions"].shape[0], 1), t_env*clock)], axis=1)
        learner.track_data(x_input, data["smoke_values"])
        
        data_to_capture = {
            'index': t_env,
            'usable': usable,
            'smoke_value_positions': data['smoke_value_positions'],
        }
        if usable:
            learner.update()

            for t_add in range(time_horizon):
                if t_env + t_add > 150 or t + t_add >= datasize:
                    data_to_capture[f'smoke_values_pred_{t_add}'] = []
                    data_to_capture[f'smoke_values_std_{t_add}'] = []
                    continue

                data = replay_buffer.get_from_index(t + t_add)

                x_input = np.concatenate([data["smoke_value_positions"], np.full((data["smoke_value_positions"].shape[0], 1), (t_env + t_add)*clock)], axis=1)
                pred, std = learner.predict(x_input)
                data_to_capture[f'smoke_values_pred_{t_add}'] = pred
                data_to_capture[f'smoke_values_std_{t_add}'] = std

        else: 
            for t_add in range(time_horizon):
                data_to_capture[f'smoke_values_pred_{t_add}'] = []
                data_to_capture[f'smoke_values_std_{t_add}'] = []
        
        captures.add(**data_to_capture)

    captures.save_to_file('tmp.npz')
    captures.save_to_file(os.path.join(opts["experiment_fp"], 'captures.npz'))

if __name__ == "__main__":

    all_opts_experiments = []

    data_fp = f'misc/'

    render = "rgb_array"

    all_opts_experiments.append({
        'sensor': 'downwards',
        'render': render,
        'samples': 100,
        'smoke_threshold': 0.25,
        'replay_buffer_fp': '/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/misc/collection_data_1_1210_1530/replay_buffer.npz'
    })

    for i, opts in enumerate(all_opts_experiments):
        experiment_name = f'collection_data_{i+1}'
        experiment_name = f'{experiment_name}_{strftime("%m%d_%H%M")}'
        experiment_fp = os.path.join(data_fp, experiment_name)
        os.makedirs(experiment_fp, exist_ok=True)

        opts['experiment_fp'] = experiment_fp

        main(opts=opts)
