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

def main(opts: dict = {}):
    env_params = EnvParams.load_from_yaml("envs/env_cfg.yaml")
    discrete_resolution = 0.6
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
        sensor_params = DownwardsSensorParams(
            world_x_size=env_params.world_x_size,
            world_y_size=env_params.world_y_size,
            points_in_range=8,
            fov_size_degrees=15
        )
    elif opts.get('sensor') == 'global':
        sensor_params = GlobalSensorParams(
            world_x_size=env_params.world_x_size,
            world_y_size=env_params.world_y_size
        )
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

    # ==== CBF CONTROL SETUP ====
    cbf = CBFController(env_params, robot_params, env_params.goal_location, opts.get('smoke_threshold'))

    opts['env_params'] = env_params
    opts['robot_params'] = robot_params
    opts['smoke_params'] = smoke_params

    renderer = SimpleRenderer(opts=opts)
    
    logger_metrics = LoggerMetrics()

    # Placeholders
    summary = {"reached_goal": False, "crashed": False, "truncated": False}

    time_tracker = TimeTracker()

    finished = False

    # Create a line plot for the gradient values
    for t in tqdm(range(0,env_params.max_steps+1)):
        if finished:
            break

        with time_tracker.track("cbf"):
            pos_x, pos_y, angle = state["location"][0], state["location"][1], state["angle"]
            bounds = env.sensor.projection_bounds(pos_x, pos_y)

            x_state = np.array([pos_x, pos_y, angle])
            cbf.update_h_discrete(state["smoke_density"].flatten(), state["smoke_density_location"], x_state)
            f = env.robot.open_loop_dynamics(x_state)
            g = env.robot.control_jacobian(x_state)
            action_input = cbf.get_command(x_state, f, g)
        
        with time_tracker.track("env"):
            state, reward, terminated, truncated, env_info = env.step(np.array(action_input))

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
            'logger_metrics': logger_metrics,
            'state': state,
            'action_input': action_input,
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

    data_fp = f'misc/'

    render = "rgb_array"

    all_opts_experiments.append({
        'sensor': 'downwards',
        'render': render,
        'samples': 100,
        'smoke_threshold': 0.25,
    })

    # all_opts_experiments.append({
    #     'sensor': 'downwards',
    #     'render': render,
    #     'samples': 100,
    #     'smoke_threshold': 0.5,
    # })

    # all_opts_experiments.append({
    #     'sensor': 'downwards',
    #     'render': render,
    #     'samples': 100,
    #     'smoke_threshold': 0.75,
    # })

    for i, opts in enumerate(all_opts_experiments):
        experiment_name = f'exp_cbf_{i+1}'
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
