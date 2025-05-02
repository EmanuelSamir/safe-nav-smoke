from learning.base_model import BaseModel
from learning.gaussian_process import GaussianProcess, plot_static_map
from itertools import product
import matplotlib.pyplot as plt
from src.failure_map_builder import FailureMapBuilder, FailureMapParams
from envs.smoke_env import EnvParams, RobotParams, SmokeEnv
from reachability.warm_start_solver import WarmStartSolver, WarmStartSolverConfig
from simulator.static_smoke import SmokeBlobParams
import numpy as np

from src.mppi import Navigator, dubins_dynamics_tensor
from matplotlib.patches import FancyArrow, Arrow
from src.utils import *

from time import time
import imageio.v2 as imageio
from io import BytesIO

import pandas as pd
from tqdm import tqdm
import os

def main(opts: dict = {}):
    env_params = EnvParams()
    env_params.world_x_size = 20
    env_params.world_y_size = 20
    env_params.max_steps = 200
    env_params.render = False
    env_params.goal_location = (18, 18)

    n_blobs = 5
    margin = 4
    random_poses = np.zeros((n_blobs, 2))
    random_poses[:, 0] = np.random.uniform(margin, env_params.world_x_size - margin, n_blobs)
    random_poses[:, 1] = np.random.uniform(margin, env_params.world_y_size - margin, n_blobs)

    robot_params = RobotParams()
    # smoke_blob_params = [
    #     SmokeBlobParams(x_pos=5, y_pos=5, intensity=1.0, spread_rate=2.0),
    #     SmokeBlobParams(x_pos=15, y_pos=12, intensity=1.0, spread_rate=2.0),
    #     SmokeBlobParams(x_pos=5, y_pos=12, intensity=1.0, spread_rate=2.0),
    #     SmokeBlobParams(x_pos=12, y_pos=7, intensity=1.0, spread_rate=2.0),
    # ]
    smoke_blob_params = [
        SmokeBlobParams(x_pos=random_poses[i, 0], y_pos=random_poses[i, 1], intensity=1.0, spread_rate=2.0) for i in range(n_blobs)
    ]

    env = SmokeEnv(env_params, robot_params, smoke_blob_params)

    state, _ = env.reset(initial_state=np.array([2.0, 2.0, 0.0, 0.0]))

    learner = GaussianProcess()


    builder = FailureMapBuilder(
        params=FailureMapParams(
            x_size=env_params.world_x_size, 
            y_size=env_params.world_y_size, 
            resolution=0.2, 
            map_rule_type='cvar',
            map_rule_threshold=0.7
            )
        )


    cell_y_size, cell_x_size = get_index_bounds(env_params.world_x_size, env_params.world_y_size, builder.params.resolution)
    # Order is according to model input order and not the image order
    domain_cells = np.array([cell_x_size, cell_y_size, 20])
    domain = [[0, 0, 0], [env_params.world_x_size, env_params.world_y_size, 2*np.pi]]

    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins3d",
            domain_cells=domain_cells,
            domain=domain,
            mode="brt",
            accuracy="very_high",
            converged_values=None,
            until_convergent=False,
            print_progress=False,
        )
    )


    nom_controller = Navigator()
    nom_controller.set_odom(state[:2],state[2])
    nom_controller.set_map(builder.failure_map, builder.params.resolution)
    nom_controller.set_goal(list(env_params.goal_location))

    # TODO: Make this part of the dynamics
    NOMINAL_ACTION_V = 3.0
    nominal_action_w = nom_controller.get_command().item()
    nominal_action = np.array([NOMINAL_ACTION_V, nominal_action_w])

    update_interval = 5
    values = None

    learner_gt = GaussianProcess(online=True)

    for i in range(200):
        X_sample = np.concatenate([np.random.uniform(0, env_params.world_y_size, 1), np.random.uniform(0, env_params.world_x_size, 1)])
        y_observe = env.smoke_simulator.get_smoke_density(X_sample[1], X_sample[0])
        learner_gt.track_data(X_sample, y_observe)
    learner_gt.update()

    cell_y_size, cell_x_size = get_index_bounds(builder.params.x_size, builder.params.y_size, builder.params.resolution)
    x = np.linspace(0, env_params.world_x_size, cell_x_size)
    y = np.linspace(0, env_params.world_y_size, cell_y_size)
    xy = np.array(list(product(y, x)))


    y_pred, std = learner_gt.predict(xy)
    continuous_map = y_pred.reshape(cell_y_size, cell_x_size)
    map_assets = {}
    map_assets['continuous_map'] = continuous_map
    map_assets['std_map'] = std.reshape(cell_y_size, cell_x_size)
    gt_failure_map = builder.rule_based_map(map_assets)

    if opts.get('render'):
        f = plt.figure(figsize=(12, 6))
        gs = f.add_gridspec(1, 3)
        ax_env = f.add_subplot(gs[0])
        ax_map = f.add_subplot(gs[1])
        ax_fail = f.add_subplot(gs[2])
        ax_fail.contour(x, y, gt_failure_map, levels=[0.5], colors='red')

        plt.tight_layout()
        plt.draw()

    traj_nonfail = []
    traj_fail = []

    frames = []

    soft_action_enabled = True
    tau_hard = 0.05
    tau_soft = 0.50
    gamma_spread = 1.0

    # Create a line plot for the gradient values
    for t in range(1,env_params.max_steps):
        learner.track_data(state[0:2][::-1], state[3])

        learner.update()
        builder.build_map(learner)
        if t % update_interval == 0 and not opts.get("safety_disabled"):
            if np.all(builder.failure_map == 1):
                values = None
            else:
                values = solver.solve(builder.failure_map.T, target_time=-2.5, dt=0.1, epsilon=0.0001)

        nominal_action = nom_controller.get_command()
        nominal_action = np.array([NOMINAL_ACTION_V, nominal_action.item()])
        safe_action = nominal_action
        if values is not None and not opts.get("safety_disabled"):
            if soft_action_enabled:
                _, value, _ = solver.check_if_safe(state[0:3], values)
                sigmoid = lambda x: 1 / (1 + np.exp((gamma_spread * (x - tau_soft))))
                lambda_value = 1.0 if value < tau_hard else sigmoid(value)
                safe_action = solver.compute_safe_action(state[0:3], action_bounds=np.array([[0.0, 5.0], [-4.0, 4.0]]), values=values)
                safe_action = lambda_value * safe_action + (1 - lambda_value) * nominal_action
            else:
                safe_action, _, _ = solver.compute_safe_control(state[0:3], nominal_action, action_bounds=np.array([[0.0, 5.0], [-4.0, 4.0]]), values=values)
        else:   
            safe_action = nominal_action
        
        state, reward, terminated, truncated, info = env.step(safe_action)

        if state[3] > builder.params.map_rule_threshold:
            traj_fail.append(state[0:2])
        else:
            traj_nonfail.append(state[0:2])

        if terminated:
            break

        nom_controller.set_odom(state[:2], state[2])

        if opts.get("safety_disabled"):
            nom_controller.set_map(~gt_failure_map, builder.params.resolution)
        else:
            nom_controller.set_map(~builder.failure_map, builder.params.resolution)

        if opts.get('render'):
            # Real time plotting
            env._render_frame(fig=f, ax=ax_env)

            builder.plot_failure_map(fig=f, ax=ax_fail)

            plot_static_map(learner, world_x_size=env_params.world_x_size, world_y_size=env_params.world_y_size, resolution=builder.params.resolution, plot_type='cvar', fig=f, ax=ax_map)

            for arrow in ax_fail.patches:
                if isinstance(arrow, (FancyArrow, Arrow)):  
                    arrow.remove()

            for coll in ax_fail.collections:
                coll.remove()

            if values is not None and not opts.get("safety_disabled"):
                is_safe, _, _ = solver.check_if_safe(state[:3], values)
                color_robot = 'g' if is_safe else 'r'
            else:
                color_robot = 'g'

            # Plot the agent's location as a blue arrow
            ax_fail.arrow(state[0], state[1], np.cos(state[2])*0.1, np.sin(state[2])*0.1, 
                                        head_width=1., head_length=1., fc=color_robot, ec=color_robot)
            
            if len(traj_nonfail) > 0:
                ax_fail.scatter(np.array(traj_nonfail)[:, 0], np.array(traj_nonfail)[:, 1], color='green', label='Non-fail', marker='.', s=0.2)
            if len(traj_fail) > 0:
                ax_fail.scatter(np.array(traj_fail)[:, 0], np.array(traj_fail)[:, 1], color='red', label='Fail', marker='.', s=0.2)

            if values is None:
                continue

            state_ind = solver._state_to_grid(state[:3])
            z = values[:,:,state_ind[2]].T
            z_mask = z > solver.config.superlevel_set_epsilon

            contour = ax_fail.contour(x, y, z, levels=10, cmap='viridis')
            ax_fail.clabel(contour, fmt="%2.1f", colors="black", fontsize=5)

            # ax_fail.contour(x, y, z_mask, levels=[0.5], colors='orange')
            # ax_fail.contour(x, y, gt_failure_map, levels=[0.5], colors='red')

            plt.tight_layout()
            plt.draw()

        if opts.get('seq_filename') and opts.get('render'):
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = imageio.imread(buf)
            frames.append(image)

    if opts.get('seq_filename') and opts.get('render'):
        if frames:
            imageio.mimsave(f'misc/{opts["seq_filename"]}.gif', frames, duration=env_params.clock)
        else:
            print(f"No frames to save for {opts['seq_filename']}")

    env.close()


    report = {
        'total_time': len(traj_nonfail) * env_params.clock + len(traj_fail) * env_params.clock,
        'time in fail': len(traj_fail) * env_params.clock,
        'time in non-fail': len(traj_nonfail) * env_params.clock,
        'terminated': terminated,
        'seq_filename': opts.get('seq_filename'),
    }

    return report


if __name__ == "__main__":

    samples = 25
    for i in tqdm(range(samples)):
        res = main(opts={'render': True})#, "safety_disabled": False, 'seq_filename': f'mppi_{i}'})
        df = pd.DataFrame(res, index=[0])
        
        # Verificar si el archivo existe
        file_path = 'misc/results_proposal.csv'
        if os.path.exists(file_path):
            # Leer el archivo existente
            existing_df = pd.read_csv(file_path)
            
            # Asegurar que las columnas coincidan
            for col in df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            for col in existing_df.columns:
                if col not in df.columns:
                    df[col] = None
            
            # Reordenar las columnas para que coincidan
            df = df[existing_df.columns]
            
            # Agregar los nuevos resultados
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Guardar el DataFrame actualizado
        df.to_csv(file_path, index=False)
    