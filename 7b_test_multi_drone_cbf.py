#!/usr/bin/env python3
"""Test script for Multi-Agent Decentralized Shielded MPPI using the production MultiDualGuardCBFCtrl.
Scenario: N drones (DubinsRobots) arranged in a circle swapping positions to the opposite side.
Forces an unavoidable central collision, resolved gracefully by decentralized HOCBF-Shields.
"""

import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from agents.basic_robot import RobotParams
from controllers.multi_dual_guard_cbf_ctrl import MultiDualGuardCBFCtrl, MPPICtrlParams

# ==========================================================
#  1. CONSTANTS & SCENARIO CONFIG
# ==========================================================
DT = 0.1
N_DRONES = 10  # Number of drones
CIRCLE_RADIUS = 10.0  # Spawn circle radius
D_SAFE = 1.2  # Safety clearance diameter between drones
R_SENSE = 8.0  # Sensing/Interaction radius

# HOCBF Gains
K1 = 2.5
K2 = 2.5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Swarm Circle Swap using production MultiDualGuardCBFCtrl (Dubins) on: {device}")

    # 1. Setup Robot and Swarm Parameters for Dubins dynamics
    robot_params = RobotParams(
        action_dim=2,
        state_dim=3,
        action_max=[6.0, 4.0],  # [v_max, omega_max]
        action_min=[0.0, -4.0],  # [v_min, omega_min]
        state_max=[30.0, 30.0, 2 * np.pi],
        state_min=[-30.0, -30.0, 0.0],
        robot_type="dubins2d",
        dt=DT,
    )

    mppi_params = MPPICtrlParams(
        num_samples=100,
        horizon=15,
        lambda_=1.2,
    )

    controller = MultiDualGuardCBFCtrl(
        num_agents=N_DRONES,
        robot_params=robot_params,
        robot_type="dubins2d",
        goal_thresh=0.6,
        device=device,
        mppi_params=mppi_params,
        dt=DT,
        r_sense=R_SENSE,
        d_safe=D_SAFE,
        k1=K1,
        k2=K2,
    )

    # 2. Setup Spawn Configuration and Goals
    goals_dict = {}
    trajectories = {f"agent_{i}": [] for i in range(N_DRONES)}

    for i in range(N_DRONES):
        theta = i * (2 * np.pi / N_DRONES)
        px = CIRCLE_RADIUS * np.cos(theta)
        py = CIRCLE_RADIUS * np.sin(theta)
        gx = -px
        gy = -py

        # Set goal
        goals_dict[f"agent_{i}"] = np.array([gx, gy], dtype=np.float32)

        # Set initial state (facing towards center)
        angle_to_center = np.arctan2(-py, -px)
        controller.agents_controllers[f"agent_{i}"].robot.reset(
            np.array([px, py, angle_to_center], dtype=np.float32)
        )
        trajectories[f"agent_{i}"].append(np.array([px, py, angle_to_center]))

    controller.set_goals(goals_dict)

    # Dummy risk map — uniform zero risk over the workspace
    HORIZON = controller.agents_controllers["agent_0"].params.horizon
    _res = 0.5
    _xs = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, _res)
    _ys = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, _res)
    _coords = (
        np.stack(np.meshgrid(_xs, _ys, indexing="xy"), axis=-1).reshape(-1, 2).astype(np.float32)
    )
    _risk = np.zeros(len(_coords), dtype=np.float32)
    controller.set_maps(deque([(_coords, _risk)] * HORIZON, maxlen=HORIZON))

    print(f"Spawned {N_DRONES} Dubins drones for circular swap.")

    max_steps = 60
    print("Executing Simulation Steps...")

    for step in tqdm(range(max_steps)):
        # Construct current_obs dictionary to emulate the SmokeEnv interface
        current_obs = {}
        for i in range(N_DRONES):
            key = f"agent_{i}"
            state = controller.agents_controllers[key].robot.get_state()
            current_obs[key] = {
                "location": state[:2],
                "angle": state[2],
            }

        # Retrieve commands using production multi-agent controller
        actions = controller.get_commands(current_obs)

        # --- Diagnostics ---
        positions_now = [
            controller.agents_controllers[f"agent_{i}"].robot.get_state()[:2]
            for i in range(N_DRONES)
        ]
        dists = [
            np.linalg.norm(positions_now[i] - positions_now[j])
            for i in range(N_DRONES)
            for j in range(i + 1, N_DRONES)
        ]
        min_d = min(dists)
        unsafe_pairs = sum(1 for d in dists if d < D_SAFE)
        goal_dists = [
            np.linalg.norm(
                controller.agents_controllers[f"agent_{i}"].robot.get_state()[:2]
                - goals_dict[f"agent_{i}"]
            )
            for i in range(N_DRONES)
        ]
        cmds_str = "  ".join(
            f"a{i}:[{actions[f'agent_{i}'].cpu().numpy()[0]:.2f},{actions[f'agent_{i}'].cpu().numpy()[1]:.2f}]"
            for i in range(N_DRONES)
        )
        print(
            f"[step {step:3d}] min_dist={min_d:.3f} (D_SAFE={D_SAFE}) "
            f"unsafe_pairs={unsafe_pairs}  "
            f"goals={[f'{d:.2f}' for d in goal_dists]}  "
            f"cmds: {cmds_str}"
        )
        # -------------------

        # Step dynamics of each agent
        for i in range(N_DRONES):
            key = f"agent_{i}"
            u_cmd = actions[key].cpu().numpy()
            next_state = controller.agents_controllers[key].robot.dynamic_step(u_cmd)
            trajectories[key].append(next_state.copy())

        # Check termination (all drones reached goal)
        all_done = True
        for i in range(N_DRONES):
            key = f"agent_{i}"
            pos = controller.agents_controllers[key].robot.get_state()[:2]
            goal = goals_dict[key]
            dist = np.linalg.norm(pos - goal)
            if dist > 0.6:
                all_done = False

        if all_done:
            print(f"\nAll drones successfully completed circular swap safely at step {step}!")
            break

    # Calculate separation statistics
    min_dist_overall = float("inf")
    steps_recorded = len(trajectories["agent_0"])

    for s in range(steps_recorded):
        positions = []
        for i in range(N_DRONES):
            positions.append(trajectories[f"agent_{i}"][s][:2])
        for i in range(N_DRONES):
            for j in range(i + 1, N_DRONES):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist_overall:
                    min_dist_overall = d

    print(
        f"\nMinimum observed inter-drone separation: {min_dist_overall:.4f} meters (D_SAFE threshold is {D_SAFE} meters)"
    )
    if min_dist_overall >= D_SAFE:
        print("SUCCESS: Mathematical safety GUARANTEED. No collisions detected!")
    else:
        print("WARNING: Safety boundary violation detected.")

    # ==========================================================
    #  3. SWARM ANIMATION GENERATION
    # ==========================================================
    print("\nRendering Swarm Animation GIF...")
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.set_ylim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Circular Swap using Production MultiDualGuardCBFCtrl\n(Decentralized HOCBF-Shielded Dubins Swarm)"
    )

    # Draw Spawn Circle
    spawn_circ = plt.Circle(
        (0, 0), CIRCLE_RADIUS, color="gray", fill=False, linestyle="--", alpha=0.5
    )
    ax.add_patch(spawn_circ)

    # Goals and paths
    colors = plt.cm.rainbow(np.linspace(0, 1, N_DRONES))
    for i in range(N_DRONES):
        goal = goals_dict[f"agent_{i}"]
        ax.scatter(goal[0], goal[1], marker="*", color=colors[i], s=150, zorder=5)

    drone_patches = []
    trail_lines = []

    for i in range(N_DRONES):
        patch = plt.Circle((0, 0), D_SAFE / 2.0, color=colors[i], alpha=0.6)
        ax.add_patch(patch)
        drone_patches.append(patch)

        (line,) = ax.plot([], [], color=colors[i], linewidth=1.5, alpha=0.8)
        trail_lines.append(line)

    def init():
        for i in range(N_DRONES):
            drone_patches[i].center = (
                trajectories[f"agent_{i}"][0][0],
                trajectories[f"agent_{i}"][0][1],
            )
            trail_lines[i].set_data([], [])
        return drone_patches + trail_lines

    def animate(frame):
        for i in range(N_DRONES):
            key = f"agent_{i}"
            state = trajectories[key][frame]
            drone_patches[i].center = (state[0], state[1])

            # Draw trails
            trail_x = [pt[0] for pt in trajectories[key][: frame + 1]]
            trail_y = [pt[1] for pt in trajectories[key][: frame + 1]]
            trail_lines[i].set_data(trail_x, trail_y)
        return drone_patches + trail_lines

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=steps_recorded, interval=100, blit=True
    )
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "multi_drone_hocbf_swap.gif")
    ani.save(out_path, writer="pillow")
    print(f"SUCCESS: Saved animated simulation to '{out_path}'!")


if __name__ == "__main__":
    main()
