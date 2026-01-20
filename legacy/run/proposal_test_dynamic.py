from learning.gaussian_process import GaussianProcess
from itertools import product
import matplotlib.pyplot as plt
from src.risk_map_builder import RiskMapBuilder, RiskMapParams
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv
from agents.basic_robot import RobotParams
from agents.dubins_robot_fixed_velocity import DubinsRobotFixedVelocity
from agents.dubins_robot import DubinsRobot
from agents.unicycle_robot import UnicycleRobot
from simulator.dynamic_smoke import DynamicSmoke, DynamicSmokeParams, DownwardsSensorParams
from reachability.warm_start_solver import WarmStartSolver, WarmStartSolverConfig
from simulator.static_smoke import SmokeBlobParams
from skimage.transform import resize
import numpy as np
from src.mppi_control import NominalControl
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
import warnings
warnings.filterwarnings("ignore")

def main(opts: dict = {}):
    env_params = EnvParams.load_from_yaml("envs/env_cfg.yaml")
    discrete_resolution = 0.6
    sim_resolution = 0.4
    robot_type = "dubins2d_fixed_velocity"
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
    sensor_params = DownwardsSensorParams(world_x_size=env_params.world_x_size,world_y_size=env_params.world_y_size)
    smoke_params = DynamicSmokeParams(x_size=env_params.world_x_size, y_size=env_params.world_y_size, smoke_blob_params=smoke_blob_params, resolution=sim_resolution, fov_sensor_params=sensor_params)

    # ==== ROBOT SETUP ====
    robot_params = RobotParams.load_from_yaml(cfg_file)
    robot_params.world_x_size = env_params.world_x_size
    robot_params.world_y_size = env_params.world_y_size

    # ==== ENVIRONMENT SETUP ====
    env_params.goal_location = goal_location_samples[np.random.randint(0, len(goal_location_samples))]
    env_params.render = opts.get("render")
    env = DynamicSmokeEnv(env_params, robot_params, smoke_params)

    state, _ = env.reset(initial_state={"location": np.array(initial_location), "angle": 0.0, "smoke_density": 0.0})

    # ==== learner SETUP ====
    learner = GaussianProcess(online=True)

    # ==== failure map builder SETUP ====
    builder = RiskMapBuilder(
        params=RiskMapParams(
            x_size=env_params.world_x_size, 
            y_size=env_params.world_y_size, 
            resolution=discrete_resolution, 
            map_rule_type='cvar',
            map_rule_threshold=env.env_params.smoke_density_threshold,
            )
        )

    # ==== HJ REACHABILITY SOLVER SETUP ====
    cell_y_size, cell_x_size = get_index_bounds(env_params.world_x_size, env_params.world_y_size, builder.params.resolution)
    
    # Order is according to model input order and not the image order
    domain_cells = np.array([cell_x_size, cell_y_size, 20])
    domain = [[0, 0, 0], [env_params.world_x_size, env_params.world_y_size, 2*np.pi]]

    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins2d_fixed_velocity",
            domain_cells=domain_cells,
            domain=domain,
            mode="brt",
            accuracy="low",
            converged_values=None,
            until_convergent=False,
            print_progress=False,
        )
    )

    # ==== Control Barrier Function SETUP ====
    if robot_type == "dubins2d_fixed_velocity":
        robot = DubinsRobotFixedVelocity(robot_params)
    elif robot_type == "dubins2d":
        robot = DubinsRobot(robot_params)
    elif robot_type == "unicycle2d":
        robot = UnicycleRobot(robot_params)
    else:
        raise NotImplementedError(f"Robot type {robot_type} not implemented")

    cbf_hj = CBFHJ(R=0.1*np.eye(robot_params.action_dim),
                   rho=3.0,
                   u_min=robot_params.action_min,
                   u_max=robot_params.action_max,
                   alpha_type="linear",
                   b_margin=solver.config.superlevel_set_epsilon,
                   x_dim= robot_params.state_dim,
                   u_dim=robot_params.action_dim,
                   k=2.5)

    # ==== MPPI CONTROL SETUP ====
    nom_controller = NominalControl(robot_params, robot_type, builder.params.resolution)
    nom_controller.set_state(np.array([state["location"][0], state["location"][1], state["angle"]]))
    nom_controller.set_map(builder.failure_map)
    nom_controller.set_goal(list(env_params.goal_location))

    # ==== UPDATE INTERVALS SETUP ====
    hj_update_interval = 1
    gp_update_interval = 1
    values = None

    cell_y_size, cell_x_size = get_index_bounds(builder.params.x_size, builder.params.y_size, builder.params.resolution)
    x_space = np.linspace(0, env_params.world_x_size, cell_x_size)
    y_space = np.linspace(0, env_params.world_y_size, cell_y_size)

    # ==== PLOTTING SETUP ====
    if opts.get('render'):
        f = plt.figure(figsize=(8, 7))
        gs = f.add_gridspec(2, 2)
        ax_env = f.add_subplot(gs[0,0])
        ax_pred = f.add_subplot(gs[0,1])
        ax_fail = f.add_subplot(gs[1,0])
        ax_hj = f.add_subplot(gs[1,1])

        for ax in [ax_env, ax_pred, ax_fail, ax_hj]:
            ax.set_xlim(0, env_params.world_x_size)
            ax.set_ylim(0, env_params.world_y_size)
            ax.autoscale(False)

        plt.tight_layout()
        plt.draw()


    # Tracking variables
    nominal_path, safe_control_path = [], []
    h_history, xi_history, u_diff_history, u_smoothness_history = [], [], [], []
    frames = []


    # Placeholders
    xi, value, prev_action_input = None, None, None
    summary = {"reached_goal": False, "crashed": False, "truncated": False}

    time_tracker = TimeTracker()

    # Create a line plot for the gradient values
    for t in tqdm(range(1,env_params.max_steps)):

        with time_tracker.track("gaussian_process_training"):
            x_input = np.concatenate([state["smoke_density_location"], np.full((state["smoke_density_location"].shape[0], 1), t*env_params.clock)], axis=1)
            learner.track_data(x_input, state["smoke_density"])
            learner.update()
        
        with time_tracker.track("gaussian_process_inference"):
            builder.build_map(learner, (t + 3)*env_params.clock)

        with time_tracker.track("hj"):
            if t % hj_update_interval == 0: # and not opts.get("safety_disabled")
                if np.all(builder.failure_map == 1):
                    values = None
                else:
                    values = solver.solve(builder.failure_map.T, target_time=-5, dt=0.1, epsilon=0.001)

        with time_tracker.track("mppi"):
            nom_controller.set_state(np.array([state["location"][0], state["location"][1], state["angle"]]))
            nom_controller.set_map(builder.failure_map)
            nominal_action = nom_controller.get_command()
            # nominal_action = np.array([0.0])
            action_input = nominal_action

        with time_tracker.track("safe_control"):
            _, value, value_grad = solver.compute_least_restrictive_control(np.array([state["location"][0], state["location"][1], state["angle"]]), values=values)
            if value is not None:
                f_open_loop = robot.open_loop_dynamics(np.array([state["location"][0], state["location"][1], state["angle"]]))
                g_control_jacobian = robot.control_jacobian(np.array([state["location"][0], state["location"][1], state["angle"]]))
                u, xi, status = cbf_hj.compute_control(nominal_action, value, value_grad, f_open_loop, g_control_jacobian)
                if not opts.get("safety_disabled"):
                    action_input = u

        with time_tracker.track("render"):
            if opts.get('render'):
                # Real time plotting
                env._render_frame(fig=f, ax=ax_env)

                builder.plot_failure_map(fig=f, ax=ax_fail)

                pred_map = builder.map_assets['continuous_map']
                if ax_pred.images:
                    ax_pred.images[0].set_array(pred_map)
                else:
                    pred_im = ax_pred.imshow(pred_map, cmap='gray', vmin=0, vmax=1.0, extent=[0, env_params.world_x_size, 0, env_params.world_y_size], origin='lower', zorder=-10)
                    ax_pred.set_title('Predicted Map')
                    ax_pred.set_xlabel('X')
                    ax_pred.set_ylabel('Y')
                    # # Add colorbar
                    # f.colorbar(pred_im, ax=ax_pred)

                if not ax_hj.images:
                    ax_hj.imshow(np.ones_like(env.smoke_simulator.get_smoke_map()), origin='lower', cmap='gray', vmin=0, vmax=1.)
                    ax_hj.set_xlim(0, env_params.world_x_size)
                    ax_hj.set_ylim(0, env_params.world_y_size)
                    ax_hj.set_title('Safe Map (HJ Reachability)')
                    ax_hj.set_xlabel('X')
                    ax_hj.set_ylabel('Y')

                def add_artifacts_in_map(ax, pose, robot_color, gt_failure_map, nominal_path, safe_control_path):
                    # Remove all previous pose patches
                    for patch in (ax.patches):
                        if isinstance(patch, (FancyArrow, Arrow, Polygon)):  
                            patch.remove()
                    for patch in (ax.collections + ax.lines):
                        patch.remove()

                    # Draw the robot's pose as an arrow
                    ax.arrow(pose[0], pose[1], np.cos(pose[2])*0.1, np.sin(pose[2])*0.1, 
                                            head_width=0.5, head_length=0.5, fc=robot_color, ec=robot_color)

                    if len(nominal_path) > 0:
                        ax.scatter(np.array(nominal_path)[:, 0], np.array(nominal_path)[:, 1], color='green', label='Nominal Path', marker='.', s=0.4, zorder=10)
                    if len(safe_control_path) > 0:
                        ax.scatter(np.array(safe_control_path)[:, 0], np.array(safe_control_path)[:, 1], color='red', label='Safe Path', marker='.', s=0.4, zorder=10)

                    # Plot the ground truth failure map
                    ax.contour(x_space, y_space, gt_failure_map, levels=[0.5], colors='red', zorder=10)

                    # Plot the goal location
                    circle = Circle((env_params.goal_location[0], env_params.goal_location[1]), radius=env_params.goal_radius, color='g', fill=True, alpha=0.8)
                    (x0, y0), r = circle.center, circle.radius

                    ax.add_patch(circle)
                    ax.text(x0, y0 - r - env_params.goal_radius - 0.5, "goal", ha="center", va="bottom",
                            fontsize=10, color="green", zorder=10)

                    square = env.smoke_simulator.sensor.projection_bounds(pose[0], pose[1])

                    bounded_square = np.array([clip_world(p[0], p[1], env_params.world_x_size, env_params.world_y_size) for p in square])
                    ax.add_patch(Polygon(bounded_square, facecolor='none', edgecolor='blue', linewidth=2,))

                    trajectories = nom_controller.get_sampled_trajectories()
                    if trajectories is not None:
                        omega = nom_controller.planner.omega.detach().cpu()
                        weighted_traj = (omega[:, None, None] * trajectories).sum(dim=0).numpy()
                        for traj in trajectories:
                            ax.plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.3, linewidth=0.5)
                        ax.plot(weighted_traj[:, 0], weighted_traj[:, 1], color="blue", linewidth=2)

                smoke_map = resize(env.smoke_simulator.get_smoke_map(), (y_space.shape[0], x_space.shape[0]), anti_aliasing=True)
                gt_failure_map = (smoke_map > env.env_params.smoke_density_threshold).astype(float)

                if values is not None and not opts.get("safety_disabled"):
                    robot_color = 'g' if all(np.isclose(nominal_action, action_input)) else 'r'
                    if robot_color == 'g':
                        nominal_path.append(state["location"])
                    else:
                        safe_control_path.append(state["location"])
                else:
                    robot_color = 'g'
                    nominal_path.append(state["location"])

                pose = np.array([state["location"][0], state["location"][1], state["angle"]])
                add_artifacts_in_map(ax_pred, pose, robot_color, gt_failure_map, nominal_path, safe_control_path)
                add_artifacts_in_map(ax_fail, pose, robot_color, gt_failure_map, nominal_path, safe_control_path)
                add_artifacts_in_map(ax_hj, pose, robot_color, gt_failure_map, nominal_path, safe_control_path)

                if values is not None:
                    # Plot the unsafe contourn
                    state_ind = solver._state_to_grid(np.array([state["location"][0], state["location"][1], state["angle"]]))
                    z = values[:,:,state_ind[2]].T

                    safe_contour = ax_hj.contour(x_space, y_space, z, levels=[solver.config.superlevel_set_epsilon], colors='purple')
                    ax_hj.clabel(safe_contour, fmt="%2.1f", colors="black", fontsize=5)

                    fail_map = builder.map_assets['cvar_map'] if builder.params.map_rule_type == 'cvar' else builder.map_assets['continuous_map']
                    fail_contour = ax_hj.contour(x_space, y_space, fail_map, levels=[env.env_params.smoke_density_threshold], colors='orange')
                    ax_hj.clabel(fail_contour, fmt="%2.1f", colors="black", fontsize=5)

                plt.tight_layout()
                plt.draw()

        if opts.get('seq_filepath') and opts.get('render'):
            buf = BytesIO()
            f.savefig(buf, format='png')
            buf.seek(0)
            image = imageio.imread(buf)
            frames.append(image)

            if opts.get('render') == "rgb_array" and opts.get("writer"):
                if image.ndim == 2:  # grayscale fallback
                    image = np.expand_dims(image, axis=-1)
                elif image.shape[-1] == 4:
                    # RGBA → RGB (opcional)
                    image = image[:, :, :3]
                assert image.shape[2] == 3, "Image must be RGB"
                opts.get("writer").add_image(f"frames", image, global_step=t, dataformats="HWC")

        # Env update
        with time_tracker.track("sim"):
            state, reward, terminated, truncated, info = env.step(action_input)

        if terminated or truncated:
            if terminated and reward == 1.0:
                summary["reached_goal"] = True
            elif terminated and reward == 0.0:
                summary["crashed"] = True
            else:
                summary["truncated"] = True

        if opts.get('writer'):
            if all(v is not None for v in [value, xi, action_input, nominal_action, prev_action_input]):
                h = float(value - cbf_hj.b_margin)
                xi = float(xi)
                u_diff = float(np.linalg.norm(np.array(action_input) - np.array(nominal_action)))
                u_smoothness = float(np.linalg.norm(np.array(action_input) - np.array(prev_action_input)))
                h_history.append(h)
                xi_history.append(xi)
                u_diff_history.append(u_diff)
                u_smoothness_history.append(u_smoothness)
                u_smoothness = u_smoothness
                safety_active = float(np.array(nominal_action) != np.array(action_input))
                metrics_step = {
                    "cbf/h": h,
                    "cbf/h_active": h < 0,
                    "cbf/xi": xi,
                    "cbf/u_diff_norm": u_diff,
                    "cbf/u_smoothness": u_smoothness,
                    "perf/safety_active": safety_active,
                    "perf/dist_to_goal": np.linalg.norm(state["location"] - env_params.goal_location),
                }
                for k, v in metrics_step.items():
                    opts.get('writer').add_scalar(k, v, t)

        prev_action_input = action_input
        for name, times in time_tracker.times.items():
            opts.get('writer').add_scalar(f"time/{name}", times[-1], t)

        if terminated or truncated:
            break


    if opts.get('seq_filepath') and opts.get('render'):
        if frames:
            imageio.mimsave(f'{opts["seq_filepath"]}', frames, duration=env_params.clock)
        else:
            print(f"No frames to save for {opts['seq_filepath']}")

    # Episode metrics
    episode_metrics = compute_episode_metrics(h_history, xi_history, u_diff_history, u_smoothness_history, time_tracker, env_params.clock)
    episode_metrics.update(summary)

    env.close()

    return episode_metrics

def compute_episode_metrics(h_history, xi_history, u_diff_history, u_smoothness_history, tracker, dt):
    h_arr = np.array(h_history)
    xi_arr = np.array(xi_history)
    u_diff_arr = np.array(u_diff_history)
    u_smoothness_arr = np.array(u_smoothness_history)

    metrics = {
        "avg_h": np.mean(h_arr),
        "min_h": np.min(h_arr),
        "activation_ratio": np.mean(h_arr < 0),
        "avg_xi": np.mean(xi_arr),
        "max_u_diff": np.max(u_diff_arr),
        "mean_smoothness": np.mean(u_smoothness_arr) if len(u_smoothness_arr) > 0 else 0.0,
        "time_in_safe": np.sum(h_arr >= 0) * dt,
        "total_time": len(h_arr) * dt,
        "total_cbf_compute_time": sum(tracker.times.get("cbf_step", [])),
    }
    return metrics


if __name__ == "__main__":
    samples = 100
    experiment_name = 'proposal'
    experiment_name = f'{experiment_name}_{strftime("%m%d_%H%M")}'
    experiment_folder = f'misc/{experiment_name}'
    os.makedirs(experiment_folder, exist_ok=True)
    global_writer = SummaryWriter(f'{experiment_folder}/')

    results_df = pd.DataFrame()

    try: 
        for i in tqdm(range(samples)):
            print('--------------------------------')
            print(f'Running sample {i + 1} of {samples}')
            run_fp = f'{experiment_folder}/run_{i+1}'
            os.makedirs(run_fp, exist_ok=True)
            episode_writer = SummaryWriter(run_fp)
            seq_filepath = f'{run_fp}/result.gif'
            opts={'render': "rgb_array", 'seq_filepath': seq_filepath, "safety_disabled": False, 'writer': episode_writer}

            metrics = main(opts=opts)
            opts['writer'].flush()
            opts['writer'].close()

            for k, v in metrics.items():
                global_writer.add_scalar(k, v, i)

            results_df = pd.concat([results_df, pd.DataFrame(metrics, index=[0])], ignore_index=True)
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        results_df.to_csv(f'{experiment_folder}/results.csv', index=False)