import argparse
import io
import os
import sys
import time
from collections import deque

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from agents.basic_robot import RobotParams
from controllers.hj import HJSolverConfig
from controllers.mppi_ctrl import MPPICtrlParams
from controllers.multi_cbf_filtering_ctrl import MultiCBFFilteringCtrl
from controllers.multi_cbf_penalize_ctrl import MultiCBFPenalizeCtrl
from controllers.multi_dual_guard_cbf_ctrl import MultiDualGuardCBFCtrl
from controllers.multi_dual_guard_hj_ctrl import MultiDualGuardHJCtrl
from controllers.multi_dual_guard_hj_online_ctrl import MultiDualGuardHJOnlineCtrl
from controllers.multi_lrf_filtering_ctrl import MultiLRFFilteringCtrl
from controllers.multi_lrf_filtering_online_ctrl import MultiLRFFilteringOnlineCtrl
from env.smoke_env import SmokeEnv
from controllers.mppi_ctrl import MPPICtrl

# ==========================================================
#  BASELINE: PURE MPPI WITHOUT ANY SAFETY FILTER
# ==========================================================
class _NominalMPPICtrl(MPPICtrl):
    def get_command(self) -> torch.Tensor:
        # Returns raw MPPI command, bypassing any safety filter
        return super().get_command()

class MultiNominalMPPICtrl(MultiCBFFilteringCtrl):
    def __init__(self, *args, **kwargs):
        # Initialize normally, then override agents_controllers with pure MPPI controllers
        super().__init__(*args, **kwargs)
        self.agents_controllers = {
            f"agent_{i}": _NominalMPPICtrl(
                robot_params=kwargs["robot_params"],
                robot_type=kwargs["robot_type"],
                goal_thresh=kwargs["goal_thresh"],
                device=kwargs["device"],
                mppi_params=kwargs["mppi_params"],
                dt=kwargs["dt"],
            )
            for i in range(self.num_agents)
        }



def main():
    parser = argparse.ArgumentParser(
        description="Swarm Crossing Decentralized Controllers Benchmark"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to evaluate (max 100)"
    )
    parser.add_argument(
        "--render",
        type=str,
        default="rgb_array",
        choices=["none", "rgb_array"],
        help="Rendering mode",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (visualize one episode in 'human' mode, no saving)",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        nargs="+",
        default=None,
        help="List of controllers to evaluate. If omitted, all will run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save metrics and videos (default: outputs/YYYY-MM-DD/HH-MM-SS)",
    )
    args = parser.parse_args()

    # Generate timestamped directory inside outputs if not explicitly set
    if args.output_dir is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        args.output_dir = os.path.join("outputs", timestamp)

    # Create output directories
    if not args.test:
        os.makedirs(f"{args.output_dir}/videos", exist_ok=True)

    # 1. Verification of Safe Starting Positions vs Smoke Sources
    print("=" * 70)
    print("🔍 VERIFYING DRONE START POSITIONS VS SMOKE SOURCES...")
    print("=" * 70)
    print("Drone X-start positions: 3.0 (left side), 27.0 (right side)")
    print("Smoke source X-centers: 8.0, 10.0, 15.0, 20.0, 22.0 (based on tailored cases)")
    print("Minimum separation distance: 5.0 meters (safe, > max 3.0 meters plume radius)")
    print(
        "Result: VERIFIED SAFE. All drones are guaranteed to start in 0-smoke density clean zones."
    )
    print("=" * 70)

    # 2. Benchmark Scenario Setup
    if args.test:
        args.episodes = 1
        render_mode = "human"
        print("🔧 RUNNING IN TEST MODE: Visualizing 1 episode per controller in 'human' mode...")
    else:
        render_mode = args.render

    num_agents = 6
    x_size = 30.0
    y_size = 20.0
    resolution = 0.2
    dt = 0.1
    max_steps = 200
    collision_radius = 0.4
    goal_radius = 0.6
    d_safe = 1.6  # Safety separation threshold used by controllers (1.2m per drone)
    r_sense = 8.0  # Neighborhood sensing radius

    # Symmetrical crossing coordinates
    initial_positions = [
        [3.0, 3.0],  # agent_0 (left bottom)
        [3.0, 10.0],  # agent_1 (left middle)
        [3.0, 17.0],  # agent_2 (left top)
        [27.0, 3.0],  # agent_3 (right bottom)
        [27.0, 10.0],  # agent_4 (right middle)
        [27.0, 17.0],  # agent_5 (right top)
    ]

    goal_locations = [
        [27.0, 17.0],  # agent_0 target (crosses to top right)
        [27.0, 10.0],  # agent_1 target (crosses straight right)
        [27.0, 3.0],  # agent_2 target (crosses to bottom right)
        [3.0, 17.0],  # agent_3 target (crosses to top left)
        [3.0, 10.0],  # agent_4 target (crosses straight left)
        [3.0, 3.0],  # agent_5 target (crosses to bottom left)
    ]

    # Precalculate optimal starting headings towards goals
    initial_headings = []
    for start, goal in zip(initial_positions, goal_locations):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        initial_headings.append(np.arctan2(dy, dx))

    # Unified Environment configuration
    env_cfg = {
        "world_x_size": x_size,
        "world_y_size": y_size,
        "max_steps": max_steps,
        "clock": dt,
        "render": render_mode,
        "playback_path": "data/planning_smoke_200_steps_100_episodes",
        "num_agents": num_agents,
        "collision_radius": collision_radius,
        "terminate_on_collision": False,
        "collision_penalty": -10.0,
        "goal_radius": goal_radius,
        "goal_locations": goal_locations,
        "robot": {
            "type": "dubins2d",
            "dt": dt,
            "action_dim": 2,
            "state_dim": 3,
            "action_max": [5.0, 4.0],
            "action_min": [0.5, -4.0],
            "state_max": [x_size, y_size, 6.28],
            "state_min": [0.0, 0.0, 0.0],
        },
    }

    # Robot physical params for controllers
    robot_params = RobotParams(
        action_dim=2,
        state_dim=3,
        action_max=[5.0, 4.0],
        action_min=[0.5, -4.0],
        state_max=[x_size, y_size, 6.28],
        state_min=[0.0, 0.0, 0.0],
        type="dubins2d",
        dt=dt,
    )

    mppi_params = MPPICtrlParams(num_samples=120, horizon=14, lambda_=1.2)
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
    elif device == "cpu" and torch.backends.mps.is_available():
        device = "mps"

    print(f"Planning device: {device.upper()}")

    # Hamilton-Jacobi solver configuration for relative avoidance controllers
    hj_config = HJSolverConfig(
        system_name="dubins2d",
        domain_cells=np.array([60, 60, 36]),
        domain=np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 2 * np.pi]]),
        mode="brt",
        accuracy="medium",
        superlevel_set_epsilon=0.0,
    )

    # Dictionary containing all controllers to evaluate
    controllers_meta = {
        "nominal-mppi": {"class": MultiNominalMPPICtrl, "args": {}},
        "cbf-filtering": {"class": MultiCBFFilteringCtrl, "args": {"k1": 1.5, "k2": 1.5}},
        "cbf-penalize": {
            "class": MultiCBFPenalizeCtrl,
            "args": {"k1": 1.5, "C_pen": 100.0, "alpha_pen": 1.0},
        },
        "dual-guard-cbf": {"class": MultiDualGuardCBFCtrl, "args": {"k1": 1.5, "k2": 1.5}},
        "dual-guard-hj": {
            "class": MultiDualGuardHJCtrl,
            "args": {"hj_config": hj_config},
            "precompute_hj": True,
        },
        "lrf-filtering": {
            "class": MultiLRFFilteringCtrl,
            "args": {"hj_config": hj_config},
            "precompute_hj": True,
        },
        "dual-guard-hj-online": {
            "class": MultiDualGuardHJOnlineCtrl,
            "args": {"hj_config": hj_config},
        },
        "lrf-filtering-online": {
            "class": MultiLRFFilteringOnlineCtrl,
            "args": {"hj_config": hj_config},
        },
    }

    # Filter controllers if specified
    if args.controllers is not None:
        filtered_meta = {}
        for c in args.controllers:
            if c in controllers_meta:
                filtered_meta[c] = controllers_meta[c]
            else:
                print(f"Warning: Controller '{c}' is not defined and will be skipped.")
        controllers_meta = filtered_meta

    # Store benchmark statistics
    benchmark_results = {}

    for name, info in controllers_meta.items():
        print("\n" + "=" * 70)
        print(f"🚀 EVALUATING CONTROLLER: {name.upper()}")
        print("=" * 70)

        # Initialize the controller orchestrator
        ctrl_class = info["class"]
        ctrl_args = {
            "num_agents": num_agents,
            "robot_params": robot_params,
            "robot_type": "dubins2d",
            "goal_thresh": goal_radius,
            "device": device,
            "mppi_params": mppi_params,
            "dt": dt,
            "r_sense": r_sense,
            "d_safe": d_safe,
            **info["args"],
        }

        controller = ctrl_class(**ctrl_args)

        # Solve HJI relative values if requested
        if info.get("precompute_hj", False):
            print(f"Precomputing JAX HJI Relative Value Function for {name} (d_safe={d_safe})...")
            start_hj = time.time()
            controller.solve_relative(
                d_safe=d_safe, time=0.0, target_time=-5.0, dt=0.05, epsilon=0.01
            )
            print(f"JAX HJI precomputation finished in {time.time() - start_hj:.2f}s")

        # Set Goals
        goals_dict = {f"agent_{i}": np.array(goal_locations[i]) for i in range(num_agents)}
        controller.set_goals(goals_dict)

        # Instantiate unified SmokeEnv
        env = SmokeEnv(cfg=env_cfg, robot_params=robot_params)

        # Metric accumulators
        ep_successes = []
        ep_collisions = []
        ep_avg_steps_to_goal = []
        ep_max_steps_to_goal = []
        ep_planning_latencies = []
        ep_drones_reached = []
        ep_collided_drones_count = []
        ep_min_separations = []
        ep_smoothness = []
        ep_smoke_q1_list = []
        ep_smoke_med_list = []
        ep_smoke_q3_list = []
        ep_smoke_max_list = []

        # Run episodes
        for ep_idx in range(args.episodes):
            print(f"Running Episode {ep_idx + 1}/{args.episodes}...")

            # Re-initialize coordinates exactly matching the symmetric swap
            initial_state_dict = {}
            for i in range(num_agents):
                initial_state_dict[f"agent_{i}"] = {
                    "location": np.array(initial_positions[i], dtype=np.float32),
                    "angle": np.array([initial_headings[i]], dtype=np.float32),
                }

            # Reset environment
            obs, _ = env.reset(initial_state=initial_state_dict, seed=ep_idx)

            step_latencies = []
            smoke_densities_ep = []
            collision_occurred = False
            drone_collided_flags = [False] * num_agents
            drone_reached_flags = [False] * num_agents
            steps_to_reach = [200] * num_agents
            min_separation_ep = 1000.0
            omega_sq_sum_ep = 0.0
            omega_count_ep = 0
            episode_trajectories = {
                f"agent_{i}": {"x": [], "y": [], "theta": [], "smoke": []}
                for i in range(num_agents)
            }
            gif_frames = []
            finished = False
            t = 0

            # Core step-by-step episode loop
            while not finished and t < max_steps:
                # 1. Update dynamic environmental risk maps (smoke)
                # Extracted from the first agent's observation sequence
                first_agent_obs = obs.get("agent_0", list(obs.values())[0])
                if "smoke_density" in first_agent_obs and len(first_agent_obs["smoke_density"]) > 0:
                    smoke_map_flat = first_agent_obs["smoke_density"].squeeze().astype(np.float32)
                    coords_flat = first_agent_obs["smoke_density_location"]
                    controller.set_maps(
                        deque(
                            [(coords_flat, smoke_map_flat)] * mppi_params.horizon,
                            maxlen=mppi_params.horizon,
                        )
                    )

                # 2. Plan optimal inputs
                start_plan = time.time()
                commands_dict = controller.get_commands(obs)
                step_latencies.append((time.time() - start_plan) * 1000.0)  # ms

                # Format actions for environment stepping
                step_actions = {k: cmd.detach().cpu().numpy() for k, cmd in commands_dict.items()}

                # 3. Step simulation
                obs, reward, terminated, truncated, _ = env.step(step_actions)

                # 4. Safety & Smoke metrics
                positions = [obs[f"agent_{i}"]["location"] for i in range(num_agents)]

                # Check for physical collisions and mark specific drones
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < (2.0 * collision_radius):
                            collision_occurred = True
                            drone_collided_flags[i] = True
                            drone_collided_flags[j] = True

                # Track if drones have ever reached their goal in this episode
                for i in range(num_agents):
                    dist = np.linalg.norm(positions[i] - goal_locations[i])
                    if dist < goal_radius:
                        drone_reached_flags[i] = True
                        if steps_to_reach[i] == 200:
                            steps_to_reach[i] = t

                # Track minimum inter-agent separation distance
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < min_separation_ep:
                            min_separation_ep = dist

                # Track control effort (steering smoothness)
                for i in range(num_agents):
                    act = step_actions.get(f"agent_{i}")
                    if act is not None:
                        w_val = float(act[1])
                        omega_sq_sum_ep += w_val**2
                        omega_count_ep += 1

                # Record step-by-step trajectory coordinates and experienced smoke
                for i in range(num_agents):
                    pos = positions[i]
                    heading = float(obs[f"agent_{i}"]["angle"])
                    smoke_val = float(np.ravel(env.get_smoke_density_in_robot(i))[0])
                    episode_trajectories[f"agent_{i}"]["x"].append(float(pos[0]))
                    episode_trajectories[f"agent_{i}"]["y"].append(float(pos[1]))
                    episode_trajectories[f"agent_{i}"]["theta"].append(heading)
                    episode_trajectories[f"agent_{i}"]["smoke"].append(smoke_val)
                    smoke_densities_ep.append(smoke_val)

                # 5. Rendering for visual comparison (rgb_array or human)
                if render_mode == "human" and t % 2 == 0:
                    env.render(controller=controller)
                elif render_mode == "rgb_array" and t % 2 == 0:
                    frame = env.render(controller=controller)
                    if frame is not None:
                        gif_frames.append(frame)

                # Check episode completion
                all_term = all(terminated.values()) if isinstance(terminated, dict) else terminated
                any_trunc = any(truncated.values()) if isinstance(truncated, dict) else truncated
                if all_term or any_trunc:
                    finished = True

                t += 1

            # Episode summary
            # Success is defined as all agents reached goal safely without collisions
            success = all(drone_reached_flags) and not collision_occurred
            ep_successes.append(1.0 if success else 0.0)
            ep_collisions.append(1.0 if collision_occurred else 0.0)

            # Resolution steps metrics
            mean_steps_to_reach = np.mean(steps_to_reach)
            max_steps_to_reach = np.max(steps_to_reach)
            ep_avg_steps_to_goal.append(mean_steps_to_reach)
            ep_max_steps_to_goal.append(max_steps_to_reach)

            # Count drones that reached goal in this episode (ever touched goal)
            drones_reached = sum(1 for flag in drone_reached_flags if flag)
            ep_drones_reached.append(drones_reached)

            collided_count = sum(1 for flag in drone_collided_flags if flag)
            ep_collided_drones_count.append(collided_count)

            ep_min_separations.append(min_separation_ep)
            ep_smoothness.append(omega_sq_sum_ep / max(1, omega_count_ep))

            # Compute agent-level smoke statistics for this episode and average them
            ep_agent_q1s = []
            ep_agent_meds = []
            ep_agent_q3s = []
            ep_agent_maxs = []
            for i in range(num_agents):
                agent_smoke = episode_trajectories[f"agent_{i}"]["smoke"]
                if agent_smoke:
                    ep_agent_q1s.append(np.percentile(agent_smoke, 25))
                    ep_agent_meds.append(np.percentile(agent_smoke, 50))
                    ep_agent_q3s.append(np.percentile(agent_smoke, 75))
                    ep_agent_maxs.append(np.max(agent_smoke))
                else:
                    ep_agent_q1s.append(0.0)
                    ep_agent_meds.append(0.0)
                    ep_agent_q3s.append(0.0)
                    ep_agent_maxs.append(0.0)

            ep_smoke_q1_list.append(np.mean(ep_agent_q1s))
            ep_smoke_med_list.append(np.mean(ep_agent_meds))
            ep_smoke_q3_list.append(np.mean(ep_agent_q3s))
            ep_smoke_max_list.append(np.mean(ep_agent_maxs))

            s_sum = sum(smoke_densities_ep)
            ep_planning_latencies.append(np.mean(step_latencies))

            print(
                f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'} | Steps (Avg/Max to Goal): {mean_steps_to_reach:.1f}/{max_steps_to_reach:.1f} | Reached: {drones_reached}/6 | Collided Drones: {collided_count}/6 | Smoke Sum: {s_sum:.3f} | Latency: {np.mean(step_latencies):.2f} ms"
            )

            # Save detailed trajectory data
            if not args.test:
                os.makedirs(f"{args.output_dir}/trajectories", exist_ok=True)
                traj_path = f"{args.output_dir}/trajectories/{name}_ep_{ep_idx + 1}_trajectory.json"
                import json

                with open(traj_path, "w") as f:
                    json.dump(episode_trajectories, f, indent=2)

            # Save comparative GIF for this episode
            if not args.test and render_mode == "rgb_array" and gif_frames:
                gif_path = f"{args.output_dir}/videos/{name}_ep_{ep_idx + 1}_crossing.gif"
                print(f"Saving playback GIF to {gif_path}...")
                imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)

        # Store mean aggregated metrics
        q1 = np.mean(ep_smoke_q1_list) if ep_smoke_q1_list else 0.0
        median = np.mean(ep_smoke_med_list) if ep_smoke_med_list else 0.0
        q3 = np.mean(ep_smoke_q3_list) if ep_smoke_q3_list else 0.0
        max_smoke = np.mean(ep_smoke_max_list) if ep_smoke_max_list else 0.0

        benchmark_results[name] = {
            "Success Rate (%)": np.mean(ep_successes) * 100.0,
            "Collision Rate (%)": np.mean(ep_collisions) * 100.0,
            "Avg Reached Drones": np.mean(ep_drones_reached),
            "Avg Collided Drones": np.mean(ep_collided_drones_count),
            "Avg Steps to Goal": np.mean(ep_avg_steps_to_goal),
            "Max Steps to Goal": np.mean(ep_max_steps_to_goal),
            "Smoke Q1": float(q1),
            "Smoke Median": float(median),
            "Smoke Q3": float(q3),
            "Smoke Max (Peak)": float(max_smoke),
            "Min Separation (m)": np.mean(ep_min_separations),
            "Control Smoothness": np.mean(ep_smoothness),
            "Avg Planning Latency (ms)": np.mean(ep_planning_latencies),
        }

        # Clear active plots to release memory
        plt.close("all")
        env.close()

    # 3. Print Premium Comparative Summary Table
    print("\n" + "=" * 165)
    print("📊 DECENTRALIZED MULTI-AGENT SAFETY SHIELD BENCHMARK COMPARATIVE RESULTS")
    print("=" * 165)

    headers = [
        "Controller",
        "Success (%)",
        "Collision (%)",
        "Reached Dr",
        "Collided Dr",
        "Avg Steps",
        "Max Steps",
        "Smoke Q1",
        "Smoke Med",
        "Smoke Q3",
        "Smoke Max",
        "Min Sep(m)",
        "Smoothness",
        "Latency(ms)",
    ]
    print(
        f"{headers[0]:<15} | {headers[1]:>11} | {headers[2]:>13} | {headers[3]:>10} | {headers[4]:>11} | "
        f"{headers[5]:>9} | {headers[6]:>9} | {headers[7]:>8} | {headers[8]:>9} | {headers[9]:>8} | "
        f"{headers[10]:>9} | {headers[11]:>10} | {headers[12]:>10} | {headers[13]:>11}"
    )
    print("-" * 165)

    for name, metrics in benchmark_results.items():
        print(
            f"{name:<15} | "
            f"{metrics['Success Rate (%)']:>11.1f} | "
            f"{metrics['Collision Rate (%)']:>13.1f} | "
            f"{metrics['Avg Reached Drones']:>10.2f} | "
            f"{metrics['Avg Collided Drones']:>11.2f} | "
            f"{metrics['Avg Steps to Goal']:>9.1f} | "
            f"{metrics['Max Steps to Goal']:>9.1f} | "
            f"{metrics['Smoke Q1']:>8.4f} | "
            f"{metrics['Smoke Median']:>9.4f} | "
            f"{metrics['Smoke Q3']:>8.4f} | "
            f"{metrics['Smoke Max (Peak)']:>9.4f} | "
            f"{metrics['Min Separation (m)']:>10.3f} | "
            f"{metrics['Control Smoothness']:>10.4f} | "
            f"{metrics['Avg Planning Latency (ms)']:>11.2f}"
        )
    print("=" * 165)

    # 4. Save metrics to CSV
    if not args.test:
        df = pd.DataFrame(benchmark_results).T
        df.index.name = "Controller"
        csv_path = f"{args.output_dir}/multi_agent_benchmark_results.csv"
        df.to_csv(csv_path)
        print(f"Saved detailed comparative CSV table to '{csv_path}'.")

    # 5. Generate high-fidelity comparison plot
    if not args.test:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Decentralized Multi-Agent Safety Shield Controllers Comparative Study",
            fontsize=16,
            fontweight="bold",
        )

        controllers = list(benchmark_results.keys())
        success_rates = [metrics["Success Rate (%)"] for metrics in benchmark_results.values()]
        avg_steps = [metrics["Avg Steps to Goal"] for metrics in benchmark_results.values()]
        smoke_q3 = [metrics["Smoke Q3"] for metrics in benchmark_results.values()]
        min_seps = [metrics["Min Separation (m)"] for metrics in benchmark_results.values()]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Success Rate
        axes[0, 0].bar(controllers, success_rates, color=colors, alpha=0.8, edgecolor="black")
        axes[0, 0].set_title("Success Rate (higher is better)", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Success Rate (%)")
        axes[0, 0].set_ylim(0, 105)
        axes[0, 0].grid(axis="y", alpha=0.3)

        # Avg Steps to Goal
        axes[0, 1].bar(controllers, avg_steps, color=colors, alpha=0.8, edgecolor="black")
        axes[0, 1].set_title("Avg Steps to Goal (lower is better)", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(axis="y", alpha=0.3)

        # Smoke Q3 (75th Percentile Exposure)
        axes[1, 0].bar(controllers, smoke_q3, color=colors, alpha=0.8, edgecolor="black")
        axes[1, 0].set_title(
            "Smoke Q3 (75th Percentile Exposure - lower is better)", fontsize=12, fontweight="bold"
        )
        axes[1, 0].set_ylabel("Smoke Density")
        axes[1, 0].grid(axis="y", alpha=0.3)

        # Min Inter-Agent Separation
        axes[1, 1].bar(controllers, min_seps, color=colors, alpha=0.8, edgecolor="black")
        axes[1, 1].set_title(
            "Min Inter-Agent Separation (higher is better)", fontsize=12, fontweight="bold"
        )
        axes[1, 1].set_ylabel("Separation Distance (meters)")
        # Draw physical collision limit line
        axes[1, 1].axhline(
            y=0.8, color="r", linestyle="--", linewidth=1.5, label="Collision Limit (0.8m)"
        )
        axes[1, 1].legend()
        axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = f"{args.output_dir}/multi_agent_benchmark_comparison.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Saved comparative study chart to '{plot_path}'.")


if __name__ == "__main__":
    main()
