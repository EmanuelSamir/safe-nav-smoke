import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def main():
    parser = argparse.ArgumentParser(description="Reconstruct benchmark results from saved JSON trajectories")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/outputs/2026-06-17/01-43-15",
        help="Path to the timestamped output directory"
    )
    args = parser.parse_args()

    trajectories_dir = os.path.join(args.output_dir, "trajectories")
    if not os.path.exists(trajectories_dir):
        print(f"Error: Directory {trajectories_dir} does not exist.")
        return

    # Symmetrical benchmark parameters
    goal_locations = [
        [27.0, 17.0],  # agent_0 target
        [27.0, 10.0],  # agent_1 target
        [27.0, 3.0],   # agent_2 target
        [3.0, 17.0],   # agent_3 target
        [3.0, 10.0],   # agent_4 target
        [3.0, 3.0],    # agent_5 target
    ]
    goal_radius = 0.6
    collision_radius = 0.4
    dt = 0.1
    num_agents = 6

    # Load all json files
    json_pattern = os.path.join(trajectories_dir, "*.json")
    files = glob.glob(json_pattern)
    if not files:
        print("No trajectory files found.")
        return

    # Group files by controller name and sort by episode number
    controller_data = {}
    for f in files:
        basename = os.path.basename(f)
        # Format: {controller}_ep_{ep_num}_trajectory.json
        parts = basename.rsplit("_ep_", 1)
        if len(parts) != 2:
            continue
        ctrl_name = parts[0]
        ep_parts = parts[1].split("_")
        try:
            ep_num = int(ep_parts[0])
        except ValueError:
            continue
        
        mtime = os.path.getmtime(f)
        if ctrl_name not in controller_data:
            controller_data[ctrl_name] = []
        controller_data[ctrl_name].append((ep_num, f, mtime))

    # Sort each group by episode number
    for ctrl in controller_data:
        controller_data[ctrl] = sorted(controller_data[ctrl], key=lambda x: x[0])

    benchmark_results = {}

    for name, episodes in controller_data.items():
        print(f"\nProcessing {name} (completed {len(episodes)} episodes)...")
        
        ep_successes = []
        ep_collisions = []
        ep_avg_steps_to_goal = []
        ep_max_steps_to_goal = []
        ep_drones_reached = []
        ep_collided_drones_count = []
        ep_min_separations = []
        ep_smoothness = []
        ep_smoke_q1_list = []
        ep_smoke_med_list = []
        ep_smoke_q3_list = []
        ep_smoke_max_list = []
        ep_planning_latencies = []

        # We will calculate latency as difference in file modification times
        # divided by the number of steps in the episode.
        # Note: the first episode of a controller will use a fallback or the time
        # difference from the previous controller's last file, but to be safe,
        # we can calculate it only if we have a previous file in the sequence.
        prev_mtime = None

        for idx, (ep_num, path, mtime) in enumerate(episodes):
            with open(path, "r") as f:
                data = json.load(f)

            # Extract steps length
            first_agent_traj = data.get("agent_0", {})
            steps = len(first_agent_traj.get("x", []))
            if steps == 0:
                continue

            # Reconstruction of Success & Collision
            collision_occurred = False
            drone_collided_flags = [False] * num_agents
            drone_reached_flags = [False] * num_agents
            steps_to_reach = [200] * num_agents
            min_separation_ep = 1000.0
            
            omega_sq_sum_ep = 0.0
            omega_count_ep = 0
            
            smoke_densities_ep = []

            for t in range(steps):
                positions = []
                for i in range(num_agents):
                    agent_key = f"agent_{i}"
                    x = data[agent_key]["x"][t]
                    y = data[agent_key]["y"][t]
                    positions.append(np.array([x, y]))
                
                # Check for collisions
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < (2.0 * collision_radius):
                            collision_occurred = True
                            drone_collided_flags[i] = True
                            drone_collided_flags[j] = True
                        if dist < min_separation_ep:
                            min_separation_ep = dist

                # Track if reached goal
                for i in range(num_agents):
                    dist = np.linalg.norm(positions[i] - goal_locations[i])
                    if dist < goal_radius:
                        drone_reached_flags[i] = True
                        if steps_to_reach[i] == 200:
                            steps_to_reach[i] = t

                # Smoothness reconstruction
                for i in range(num_agents):
                    agent_key = f"agent_{i}"
                    theta = data[agent_key]["theta"]
                    if t < steps - 1:
                        # heading wrap-around safe omega
                        dtheta = wrap_angle(theta[t+1] - theta[t])
                        w_val = dtheta / dt
                        omega_sq_sum_ep += w_val**2
                        omega_count_ep += 1

            # Smoke calculations
            ep_agent_q1s = []
            ep_agent_meds = []
            ep_agent_q3s = []
            ep_agent_maxs = []
            for i in range(num_agents):
                agent_smoke = data[f"agent_{i}"]["smoke"]
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

            # Estimate Latency from file timestamps (excluding episode 1 compile time outlier for cleaner steady-state)
            if prev_mtime is not None:
                elapsed_time_sec = mtime - prev_mtime
                # If the elapsed time is abnormally large (e.g. run was paused or controllers changed),
                # we don't include it in average latency.
                if elapsed_time_sec > 0 and elapsed_time_sec < (steps * 10): 
                    latency_per_step_ms = (elapsed_time_sec / steps) * 1000.0
                    ep_planning_latencies.append(latency_per_step_ms)
            
            prev_mtime = mtime

            success = all(drone_reached_flags) and not collision_occurred
            ep_successes.append(1.0 if success else 0.0)
            ep_collisions.append(1.0 if collision_occurred else 0.0)
            ep_avg_steps_to_goal.append(np.mean(steps_to_reach))
            ep_max_steps_to_goal.append(np.max(steps_to_reach))
            ep_drones_reached.append(sum(1 for f in drone_reached_flags if f))
            ep_collided_drones_count.append(sum(1 for f in drone_collided_flags if f))
            ep_min_separations.append(min_separation_ep)
            ep_smoothness.append(omega_sq_sum_ep / max(1, omega_count_ep))
            ep_smoke_q1_list.append(np.mean(ep_agent_q1s))
            ep_smoke_med_list.append(np.mean(ep_agent_meds))
            ep_smoke_q3_list.append(np.mean(ep_agent_q3s))
            ep_smoke_max_list.append(np.mean(ep_agent_maxs))

        # Handle latency fallback if only 1 episode or no consecutive episodes
        if not ep_planning_latencies:
            ep_planning_latencies = [0.0]

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

    # Print Table
    print("\n" + "=" * 165)
    print("📊 RECONSTRUCTED DECENTRALIZED MULTI-AGENT BENCHMARK RESULTS")
    print("=" * 165)
    headers = [
        "Controller", "Success (%)", "Collision (%)", "Reached Dr", "Collided Dr",
        "Avg Steps", "Max Steps", "Smoke Q1", "Smoke Med", "Smoke Q3", "Smoke Max",
        "Min Sep(m)", "Smoothness", "Latency(ms)*"
    ]
    print(
        f"{headers[0]:<25} | {headers[1]:>11} | {headers[2]:>13} | {headers[3]:>10} | {headers[4]:>11} | "
        f"{headers[5]:>9} | {headers[6]:>9} | {headers[7]:>8} | {headers[8]:>9} | {headers[9]:>8} | "
        f"{headers[10]:>9} | {headers[11]:>10} | {headers[12]:>10} | {headers[13]:>11}"
    )
    print("-" * 165)

    for name, metrics in benchmark_results.items():
        print(
            f"{name:<25} | "
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
    print("* Note: Latency(ms) is estimated from file modification timestamps (excludes JIT compile outliers).")

    # Save to CSV
    df = pd.DataFrame(benchmark_results).T
    df.index.name = "Controller"
    csv_path = os.path.join(args.output_dir, "reconstructed_results.csv")
    df.to_csv(csv_path)
    print(f"Saved reconstructed CSV table to '{csv_path}'.")

    # Save comparative study plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reconstructed Multi-Agent Safety Shield Controllers Comparative Study", fontsize=16, fontweight="bold")
    controllers = list(benchmark_results.keys())
    success_rates = [metrics["Success Rate (%)"] for metrics in benchmark_results.values()]
    avg_steps = [metrics["Avg Steps to Goal"] for metrics in benchmark_results.values()]
    smoke_q3 = [metrics["Smoke Q3"] for metrics in benchmark_results.values()]
    min_seps = [metrics["Min Separation (m)"] for metrics in benchmark_results.values()]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    axes[0, 0].bar(controllers, success_rates, color=colors[:len(controllers)], alpha=0.8, edgecolor="black")
    axes[0, 0].set_title("Success Rate (higher is better)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Success Rate (%)")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].bar(controllers, avg_steps, color=colors[:len(controllers)], alpha=0.8, edgecolor="black")
    axes[0, 1].set_title("Avg Steps to Goal (lower is better)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[1, 0].bar(controllers, smoke_q3, color=colors[:len(controllers)], alpha=0.8, edgecolor="black")
    axes[1, 0].set_title("Smoke Q3 (75th Percentile Exposure - lower is better)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Smoke Density")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(controllers, min_seps, color=colors[:len(controllers)], alpha=0.8, edgecolor="black")
    axes[1, 1].set_title("Min Inter-Agent Separation (higher is better)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Separation Distance (meters)")
    axes[1, 1].axhline(y=0.8, color="r", linestyle="--", linewidth=1.5, label="Collision Limit (0.8m)")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(args.output_dir, "reconstructed_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved reconstructed comparative study chart to '{plot_path}'.")

if __name__ == "__main__":
    main()
