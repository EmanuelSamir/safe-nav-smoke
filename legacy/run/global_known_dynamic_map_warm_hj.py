from learning.base_model import BaseModel
from learning.gaussian_process import GaussianProcess, plot_static_map, plot_dynamic_map
from itertools import product
import matplotlib.pyplot as plt
from src.risk_map_builder import RiskMapBuilder, RiskMapParams
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv
from agents.basic_robot import RobotParams
from simulator.dynamic_smoke import DynamicSmoke, DynamicSmokeParams, DownwardsSensorParams
from reachability.warm_start_solver import WarmStartSolver, WarmStartSolverConfig
from simulator.static_smoke import SmokeBlobParams
from skimage.transform import resize
import numpy as np

from src.mppi_control import NominalControl
from matplotlib.patches import FancyArrow, Arrow, Circle
from src.utils import *

from functools import wraps
from time import time, sleep, perf_counter
import imageio.v2 as imageio
from io import BytesIO

import pandas as pd
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore")

def main(opts: dict = {}):
    env_params = EnvParams.load_from_yaml("envs/env_cfg.yaml")

    n_blobs = 10
    random_poses = np.zeros((n_blobs, 2))
    for i in range(n_blobs):
        while True:
            random_poses[i, 0] = np.random.uniform(0, env_params.world_x_size)
            random_poses[i, 1] = np.random.uniform(0, env_params.world_y_size)
            if np.linalg.norm(random_poses[i, :] - env_params.goal_location) > 4.0 and np.linalg.norm(random_poses[i, :] - np.array([2.0, 2.0])) > 4.0:
                break

    smoke_blob_params = [
        SmokeBlobParams(x_pos=random_poses[i, 0], y_pos=random_poses[i, 1], intensity=1.0, spread_rate=2.0) for i in range(n_blobs)
    ]

    robot_params = RobotParams.load_from_yaml("agents/dubins_cfg.yaml")
    robot_params.world_x_size = env_params.world_x_size
    robot_params.world_y_size = env_params.world_y_size

    sensor_params = DownwardsSensorParams(world_x_size=env_params.world_x_size, world_y_size=env_params.world_y_size)
    smoke_params = DynamicSmokeParams(x_size=env_params.world_x_size, y_size=env_params.world_y_size, smoke_blob_params=smoke_blob_params, resolution=0.4, fov_sensor_params=sensor_params)

    env_params.render = opts.get("render")

    env = DynamicSmokeEnv(env_params, robot_params, smoke_params)

    state, _ = env.reset(initial_state={"location": np.array([2.0, 2.0]), "angle": 0.0, "smoke_density": 0.0})


    cell_y_size, cell_x_size = get_index_bounds(env_params.world_x_size, env_params.world_y_size, 0.6)
    
    # Order is according to model input order and not the image order
    domain_cells = np.array([cell_x_size, cell_y_size, 20])
    domain = [[0, 0, 0], [env_params.world_x_size, env_params.world_y_size, 2*np.pi]]

    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins3d",
            domain_cells=domain_cells,
            domain=domain,
            mode="brt",
            accuracy="low",
            converged_values=None,
            until_convergent=False,
            print_progress=False,
        )
    )

    nom_controller = NominalControl(robot_params, "dubins2d", 0.6, goal_thresh=env_params.goal_radius)
    nom_controller.set_state(np.array([state["location"][0], state["location"][1], state["angle"]]))

    resized_map = resize(env.smoke_simulator.get_smoke_map(), (cell_y_size, cell_x_size), anti_aliasing=True)
    failure_map = (resized_map < env.env_params.smoke_density_threshold).astype(float)
    nom_controller.set_map(failure_map)
    nom_controller.set_goal(list(env_params.goal_location))

    # TODO: Make this part of the dynamics
    nominal_action = nom_controller.get_command()

    hj_update_interval = 1
    gp_update_interval = 1
    values = None

    cell_y_size, cell_x_size = get_index_bounds(env_params.world_x_size, env_params.world_y_size, 0.6)
    x_space = np.linspace(0, env_params.world_x_size, cell_x_size)
    y_space = np.linspace(0, env_params.world_y_size, cell_y_size)

    if opts.get('render'):
        f = plt.figure(figsize=(8, 7))
        gs = f.add_gridspec(2, 2)
        ax_env = f.add_subplot(gs[0,0])
        ax_pred = f.add_subplot(gs[0,1])
        ax_fail = f.add_subplot(gs[1,0])
        ax_hj = f.add_subplot(gs[1,1])
        # ax_fail.contour(x, y, gt_failure_map, levels=[0.5], colors='red')

        plt.tight_layout()
        plt.draw()

    traj_nonfail = []
    traj_fail = []

    frames = []

    times_track = {
        "gaussian_process_training": [],
        "gaussian_process_inference": [],
        "hj": [],
        "mppi": [],
        "sim": [],
        "least-restrictive-control": [],
    }

    # Create a line plot for the gradient values
    for t in tqdm(range(1,env_params.max_steps)):

        s = time()
        
        # print(f"Time taken to track data: {time() - s}")
        times_track["gaussian_process_training"].append(time() - s)

        s = time()
        # print(f"Time taken to build map: {time() - s}")
        times_track["gaussian_process_inference"].append(time() - s)

        s = time()
        if t % hj_update_interval == 0 and not opts.get("safety_disabled"):
            smoke_map = resize(env.smoke_simulator.get_smoke_map(), (y_space.shape[0], x_space.shape[0]), anti_aliasing=True)
            failure_mask = (smoke_map < env.env_params.smoke_density_threshold).astype(float)
            nom_controller.set_map(failure_mask)
            # values = solver.solve(failure_mask.T, target_time=-2.5, dt=0.1, epsilon=0.001)
        # print(f"Time taken to solve: {time() - s}")
        times_track["hj"].append(time() - s)

        s = time()
        nominal_action = nom_controller.get_command()
        # print(f"Time taken to get command: {time() - s}")
        times_track["mppi"].append(time() - s)
        # nominal_action = np.array([robot_params.action_max[0], nominal_action.item()])
        safe_action = nominal_action
        s = time()
        # if values is not None and not opts.get("safety_disabled"):
        #     safe_action, _, _ = solver.compute_safe_control(np.array([state["location"][0], state["location"][1], state["angle"]]), nominal_action, action_bounds=np.array([[0.0, 5.0], [-4.0, 4.0]]), values=values)
        # else:   
        #     safe_action = nominal_action
        times_track["least-restrictive-control"].append(time() - s)
        # print(f"Time taken to compute safe action: {time() - s}")

        # Env update
        s = time()
        state, reward, terminated, truncated, info = env.step(safe_action)
        times_track["sim"].append(time() - s)

        # print(f"Time taken to env step: {time() - s}")

        if terminated:
            if reward == 1.0:
                traj_nonfail.append(state["location"])
            else:
                traj_fail.append(state["location"])
        else:
            traj_nonfail.append(state["location"])

        if terminated:
            break

        nom_controller.set_state(np.array([state["location"][0], state["location"][1], state["angle"]]))

        if opts.get('render'):
            # Real time plotting
            s = time()
            env._render_frame(fig=f, ax=ax_env)
            # print(f"Time taken to render env: {time() - s}")

            s = time()
            ax_fail.imshow(failure_mask, cmap='gray', origin='lower', extent=[0, env_params.world_x_size, 0, env_params.world_y_size], zorder=-10)
            # print(f"Time taken to plot failure map: {time() - s}")

            s = time() 
            # plot_dynamic_map(learner, world_x_size=env_params.world_x_size, world_y_size=env_params.world_y_size, time=t*env_params.clock, resolution=builder.params.resolution, plot_type='mean', fig=f, ax=ax_pred)
            # print(f"Time taken to plot dynamic map: {time() - s}")

            if not ax_hj.images:
                ax_hj.imshow(np.ones_like(env.smoke_simulator.get_smoke_map()), origin='lower', cmap='gray', vmin=0, vmax=1.)
                ax_hj.set_xlim(0, env_params.world_x_size)
                ax_hj.set_ylim(0, env_params.world_y_size)
                ax_hj.set_title('Safe Map (HJ Reachability)')
                ax_hj.set_xlabel('X')
                ax_hj.set_ylabel('Y')

            def add_artifacts_in_map(ax, pose, robot_color, failure_mask, traj_fail, traj_nonfail):
                # Remove all previous pose patches
                for arrow in ax.patches:
                    if isinstance(arrow, (FancyArrow, Arrow)):  
                        arrow.remove()
                    
                for coll in ax.collections:
                    coll.remove()

                ax.arrow(pose[0], pose[1], np.cos(pose[2])*0.1, np.sin(pose[2])*0.1, 
                                        head_width=1., head_length=1., fc=robot_color, ec=robot_color)

                if len(traj_nonfail) > 0:
                    ax.scatter(np.array(traj_nonfail)[:, 0], np.array(traj_nonfail)[:, 1], color='green', label='Non-fail', marker='.', s=0.2, zorder=10)
                if len(traj_fail) > 0:
                    ax.scatter(np.array(traj_fail)[:, 0], np.array(traj_fail)[:, 1], color='red', label='Fail', marker='.', s=0.2, zorder=10)

                ax.contour(x_space, y_space, failure_mask, levels=[0.5], colors='red', zorder=10)

                circle = Circle((env_params.goal_location[0], env_params.goal_location[1]), 
                radius=env_params.goal_radius, color='g', fill=True, alpha=0.8)
                ax.add_patch(circle)
                x0, y0 = circle.center
                r = circle.radius
                ax.text(x0, y0 - r - 1.5, "goal", ha="center", va="bottom",
                        fontsize=10, color="green", zorder=10)

            smoke_map = resize(env.smoke_simulator.get_smoke_map(), (y_space.shape[0], x_space.shape[0]), anti_aliasing=True)
            failure_mask = (smoke_map > env.env_params.smoke_density_threshold).astype(float)

            if values is not None and not opts.get("safety_disabled"):
                # Change the color of the robot if the robot uses safe control. TODO: Remove when using dual shield.
                is_safe, _, _ = solver.check_if_safe(np.array([state["location"][0], state["location"][1], state["angle"]]), values)
                robot_color = 'g' if is_safe else 'r'
            else:
                robot_color = 'g'

            s = time()
            pose = np.array([state["location"][0], state["location"][1], state["angle"]])
            add_artifacts_in_map(ax_pred, pose, robot_color, failure_mask, traj_fail, traj_nonfail)
            add_artifacts_in_map(ax_fail, pose, robot_color, failure_mask, traj_fail, traj_nonfail)
            add_artifacts_in_map(ax_hj, pose, robot_color, failure_mask, traj_fail, traj_nonfail)
            # print(f"Time taken to add artifacts: {time() - s}")

            s = time()  
            if values is not None:
                # Plot the unsafe contourn
                state_ind = solver._state_to_grid(np.array([state["location"][0], state["location"][1], state["angle"]]))
                z = values[:,:,state_ind[2]].T
                z_mask = z > solver.config.superlevel_set_epsilon

                levels_ = np.linspace(0, 1.0, 5)
                contour = ax_fail.contour(x_space, y_space, z, levels=levels_, cmap='viridis')
                ax_fail.clabel(contour, fmt="%2.1f", colors="black", fontsize=5)

            # print(f"Time taken to plot contours: {time() - s}")

            # ax_fail.contour(x, y, z_mask, levels=[0.5], colors='orange')
            # ax_fail.contour(x, y, gt_failure_map, levels=[0.5], colors='red')

            # sleep(10)

            plt.tight_layout()
            plt.draw()

        if opts.get('seq_filename') and opts.get('render'):
            buf = BytesIO()
            f.savefig(buf, format='png')
            buf.seek(0)
            image = imageio.imread(buf)
            frames.append(image)

        # safe_set = values > 0.0#solver.config.superlevel_set_epsilon
        # safe_set_projected = np.all(safe_set, axis=2) #(x, y)

        # if opts.get("safety_disabled"):
        #     pass 
        #     # TODO Use Global knowledge of the world if disabled
        # else:
        #     nom_controller.set_map(safe_set_projected.T)#(failure_mask.T)

    if opts.get('seq_filename') and opts.get('render'):
        if frames:
            print(f"Saving frames to {opts['seq_filename']}.gif")
            imageio.mimsave(f'misc/{opts["seq_filename"]}.gif', frames, duration=env_params.clock)
        else:
            print(f"No frames to save for {opts['seq_filename']}")

    env.close()

    def stats_no_near_zero(data, tol=1e-3, return_counts=False):
        arr = np.asarray(data, dtype=float)
        mask = ~np.isclose(arr, 0.0, atol=tol)
        filtered = arr[mask]
        
        if filtered.size > 0:
            mean = np.mean(filtered)
            std = np.std(filtered, ddof=0) if filtered.size > 1 else 0.0
        else:
            mean, std = np.nan, np.nan
        
        if return_counts:
            return mean, std, np.sum(~mask), np.sum(mask)
        else:
            return mean, std
    
    report = { 
        "gp_train_mean": stats_no_near_zero(times_track["gaussian_process_training"])[0],
        "gp_train_std": stats_no_near_zero(times_track["gaussian_process_training"])[1],
        "gp_infer_mean": stats_no_near_zero(times_track["gaussian_process_inference"])[0],
        "gp_infer_std": stats_no_near_zero(times_track["gaussian_process_inference"])[1],
        "hj_mean": stats_no_near_zero(times_track["hj"])[0],
        "hj_std": stats_no_near_zero(times_track["hj"])[1],
        "mppi_mean": stats_no_near_zero(times_track["mppi"])[0],
        "mppi_std": stats_no_near_zero(times_track["mppi"])[1],
        "sim_mean": stats_no_near_zero(times_track["sim"])[0],
        "sim_std": stats_no_near_zero(times_track["sim"])[1],
        "lrc_mean": stats_no_near_zero(times_track["least-restrictive-control"])[0],
        "lrc_std": stats_no_near_zero(times_track["least-restrictive-control"])[1],
        "time_in_safe": stats_no_near_zero(times_track["least-restrictive-control"], return_counts = True)[3] * env_params.clock,
        "total_time": len(times_track["least-restrictive-control"])*env_params.clock,
        "seq_filename": opts.get('seq_filename'),
    }

    # report = {
    #     'total_time': len(traj_nonfail) * env_params.clock + len(traj_fail) * env_params.clock,
    #     'time in fail': len(traj_fail) * env_params.clock,
    #     'time in non-fail': len(traj_nonfail) * env_params.clock,
    #     'terminated': terminated,
    #     'seq_filename': opts.get('seq_filename'),
    #     'reached_goal': reward,
    # }

    return report

if __name__ == "__main__":

    samples = 200

    for i in range(samples):
        time_start = time()
        print('--------------------------------')
        print(f'Running sample {i + 1} of {samples}')
        res = main(opts={'render': "human", 'seq_filename': f'result_attempt_7_global_known_{i+1}'})#, "safety_disabled": False, 'seq_filename': f'mppi_{i}'})
        df = pd.DataFrame(res, index=[0])
        time_end = time()
        # print(f'Time taken: {time_end - time_start} seconds')
        print('--------------------------------')
        # Verificar si el archivo existe
        file_path = 'misc/results_proposal_7_global_known.csv'
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