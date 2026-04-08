import os
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# Add root directory to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controllers.mppi_control_dyn import MPPIControlDyn, MPPIControlParams
from envs.smoke_env import SmokeEnv
from utils.mapping import SpaceCarvingMapper


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/env/smoke_env.yaml")
    parser.add_argument("--test", action="store_true", help="Run in non-interactive mode for testing")
    parser.add_argument("--episode", type=int, default=None, help="Episode index to play back")
    args = parser.parse_args()

    # 1. Setup Environment
    config_path = args.config
    if not os.path.exists(config_path):
        cfg = {
            "world_x_size": 30.0,
            "world_y_size": 30.0,
            "max_steps": 250,
            "render": "none",
            "clock": 0.05,
            "goal_location": [25.0, 25.0],
            "goal_radius": 2.0,
            "sensor": {
                "type": "camera_1d",
                "fov_size_degrees": 90.0,
                "num_rays": 32,
                "max_range": 12.0,
                "step_size": 0.2,
            },
            "smoke": {
                "resolution": 0.5,
                "blobs": [
                    {"x": 10, "y": 20, "intensity": 1.0, "spread": 4.0},
                    {"x": 20, "y": 10, "intensity": 1.0, "spread": 5.0},
                ],
            },
            "robot": {
                "type": "dubins2d",
                "action_dim": 2,
                "state_dim": 3,
                "action_max": [4.0, 1.5],
                "action_min": [0.0, -1.5],
                "dt": 0.1,
            },
        }
        cfg = OmegaConf.create(cfg)
    else:
        cfg = OmegaConf.load(config_path)
        cfg.sensor.type = "camera_1d"
        cfg.render = "none"
        if not cfg.get("goal_location"):
            cfg.goal_location = [cfg.world_x_size - 5, cfg.world_y_size - 5]

    env = SmokeEnv(cfg=cfg)
    obs, _ = env.reset(seed=args.episode)

    # Ensure goal is set
    goal_pos = np.array(cfg.goal_location)

    # 2. Setup Mapper
    # Use environment's actual resolution
    res = (
        env.smoke_simulator.scalar_resolution
        if hasattr(env.smoke_simulator, "scalar_resolution")
        else 0.4
    )

    mapper = SpaceCarvingMapper(
        world_x=env.env_params.world_x_size, world_y=env.env_params.world_y_size, resolution=res
    )

    # Pre-compute mapper coordinates for MPPI format
    map_coords = np.stack([mapper.X.ravel(), mapper.Y.ravel()], axis=-1)

    # 3. Setup MPPI Controller
    # Tuned for more aggressive avoidance and longer horizon
    mppi_params = MPPIControlParams(num_samples=100, horizon=25, lambda_=0.5, device="cpu")

    controller = MPPIControlDyn(
        robot_params=env.robot_params,
        robot_type=env.robot_params.robot_type,
        goal_thresh=env.env_params.goal_radius,
        mppi_params=mppi_params,
        dt=env.robot_params.dt,
    )
    controller.set_goal(goal_pos)

    # 4. Setup Visualization
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # GT
    ax2 = fig.add_subplot(gs[0, 1])  # Est
    ax3 = fig.add_subplot(gs[1, :])  # Stats

    # Subplot 1: Ground Truth
    im1 = ax1.imshow(
        env.smoke_simulator.get_smoke_map(),
        cmap="gray",
        extent=env.smoke_simulator.get_smoke_extent(),
        origin="lower",
        vmin=0,
        vmax=1,
    )
    ax1.set_title("Ground Truth & MPPI Trajectories")
    (robot_marker,) = ax1.plot([], [], "bo", markersize=8, label="Robot")
    ax1.plot(goal_pos[0], goal_pos[1], "gx", markersize=12, label="Goal")
    ax1.legend()

    # Subplot 2: Estimated Map
    im2 = ax2.imshow(
        mapper.get_map(),
        cmap="gray",
        extent=env.smoke_simulator.get_smoke_extent(),
        origin="lower",
        vmin=0,
        vmax=1,
    )
    ax2.set_title("Estimated Map (Space Carving)")

    # Subplot 3: Stats
    imm_smoke_data = []
    acc_smoke_data = []
    time_steps = []
    (line_imm,) = ax3.plot([], [], label="Immediate Smoke Density", color="red", linewidth=2)
    (line_acc,) = ax3.plot([], [], label="Avg Sensor Reading (Local Risk)", color="blue", alpha=0.6)
    ax3.set_title("Smoke Exposure History")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Density")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.7)

    print("--- MPPI AUTONOMOUS NAVIGATION ---")
    print(f"Goal: {goal_pos} | Resolution: {res}m")

    step = 0
    avg_scan_smoke = 0
    try:
        while plt.fignum_exists(fig.number) or args.test:
            # 1. Perception
            odom = env.get_robot_odom()
            pos = odom["location"]
            angle = odom["angle"]

            # Update Mapper with 1D scan
            mapper.update(
                pose=[pos[0], pos[1], angle],
                readings=obs["smoke_density"],
                sensor_params=env.env_params.sensor_params,
            )

            # 2. Control Update
            est_map_flat = mapper.get_map().ravel()
            controller.set_maps(deque([(map_coords, est_map_flat)] * mppi_params.horizon))
            controller.set_state(np.array([pos[0], pos[1], angle]))

            action_torch = controller.get_command()
            action = action_torch.cpu().numpy()

            # 3. Environment Step
            obs, reward, terminated, truncated, info = env.step(action)

            # 4. Data Collection
            imm_smoke = env.get_smoke_density_in_robot()
            avg_scan_smoke += imm_smoke

            imm_smoke_data.append(imm_smoke)
            acc_smoke_data.append(avg_scan_smoke)
            time_steps.append(step)

            # 5. Visualization Update
            if step % 2 == 0:  # Update viz every 2 steps to save CPU
                im1.set_array(env.smoke_simulator.get_smoke_map())
                robot_marker.set_data([pos[0]], [pos[1]])

                im2.set_array(mapper.get_map())

                line_imm.set_data(time_steps, imm_smoke_data)
                line_acc.set_data(time_steps, acc_smoke_data)
                ax3.set_xlim(max(0, step - 100), step + 5)

                fig.canvas.draw_idle()
                plt.pause(0.001)

            step += 1
            if args.test and step > 30:
                print("Test successful. Autonomous loop verified.")
                break

            if terminated or truncated:
                dist_to_goal = np.linalg.norm(pos - goal_pos)
                if dist_to_goal < env.env_params.goal_radius + 0.5:
                    print(f"SUCCESS: Goal Reached in {step} steps!")
                else:
                    print(f"FAILURE: Episode ended at distance {dist_to_goal:.2f}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        plt.ioff()
        env.close()


if __name__ == "__main__":
    main()
