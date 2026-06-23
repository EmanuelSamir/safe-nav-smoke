import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def find_latest_results_dir():
    """Scans outputs/benchmark to find the latest timestamped folder."""
    benchmark_root = os.path.join(project_root, "outputs", "benchmark")
    if not os.path.exists(benchmark_root):
        return None
        
    subdirs = []
    # outputs/benchmark is structured as outputs/benchmark/YYYY-MM-DD/HH-MM-SS
    for date_dir in os.listdir(benchmark_root):
        date_path = os.path.join(benchmark_root, date_dir)
        if os.path.isdir(date_path):
            for time_dir in os.listdir(date_path):
                time_path = os.path.join(date_path, time_dir)
                if os.path.isdir(time_path) and os.path.exists(os.path.join(time_path, "results.csv")):
                    subdirs.append(time_path)
                    
    if not subdirs:
        # Check direct subdirectories as well
        for direct_dir in os.listdir(benchmark_root):
            direct_path = os.path.join(benchmark_root, direct_dir)
            if os.path.isdir(direct_path) and os.path.exists(os.path.join(direct_path, "results.csv")):
                subdirs.append(direct_path)
                
    if not subdirs:
        return None
        
    # Return the one with the latest modification time
    return max(subdirs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument("--output_dir", type=str, default=None, help="Benchmark directory to analyze")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = find_latest_results_dir()
        if output_dir is None:
            print("Error: No benchmark results found. Run a benchmark first.")
            sys.exit(1)
        print(f"No output directory specified. Analyzing latest run: {output_dir}")
    else:
        output_dir = os.path.abspath(output_dir)
        
    csv_file = os.path.join(output_dir, "results.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        sys.exit(1)
        
    # Read the results
    df = pd.read_csv(csv_file)
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
        "Smoke Q1",
        "Smoke Med",
        "Smoke Q3",
        "Smoke Max",
        "Min Sep(m)",
        "Smoothness",
        "Latency(ms)",
    ]
    print(
        f"{headers[0]:<17} | {headers[1]:>11} | {headers[2]:>13} | {headers[3]:>10} | {headers[4]:>11} | "
        f"{headers[5]:>9} | {headers[6]:>8} | {headers[7]:>9} | {headers[8]:>8} | "
        f"{headers[9]:>9} | {headers[10]:>10} | {headers[11]:>10} | {headers[12]:>11}"
    )
    print("-" * 165)
    
    for _, row in df.iterrows():
        print(
            f"{row['Controller']:<17} | "
            f"{row['Success Rate (%)']:>11.1f} | "
            f"{row['Collision Rate (%)']:>13.1f} | "
            f"{row['Avg Reached Drones']:>10.2f} | "
            f"{row['Avg Collided Drones']:>11.2f} | "
            f"{row['Avg Steps to Goal']:>9.1f} | "
            f"{row['Smoke Q1']:>8.4f} | "
            f"{row['Smoke Median']:>9.4f} | "
            f"{row['Smoke Q3']:>8.4f} | "
            f"{row['Smoke Max (Peak)']:>9.4f} | "
            f"{row['Min Separation (m)']:>10.3f} | "
            f"{row['Control Smoothness']:>10.4f} | "
            f"{row['Avg Planning Latency (ms)']:>11.2f}"
        )
    print("=" * 165)
    
    # 5. Generate high-fidelity comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Decentralized Multi-Agent Safety Shield Controllers Comparative Study",
        fontsize=16,
        fontweight="bold",
    )
    
    controllers = df["Controller"].tolist()
    success_rates = df["Success Rate (%)"].tolist()
    avg_steps = df["Avg Steps to Goal"].tolist()
    smoke_q3 = df["Smoke Q3"].tolist()
    min_seps = df["Min Separation (m)"].tolist()
    
    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    colors = colors[:len(controllers)]
    
    # Success Rate Plot
    axes[0, 0].bar(controllers, success_rates, color=colors, alpha=0.8, edgecolor="black")
    axes[0, 0].set_title("Success Rate (higher is better)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Success Rate (%)")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    # Avg Steps Plot
    axes[0, 1].bar(controllers, avg_steps, color=colors, alpha=0.8, edgecolor="black")
    axes[0, 1].set_title("Avg Steps to Goal (lower is better)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=15)
    
    # Smoke Q3 Plot
    axes[1, 0].bar(controllers, smoke_q3, color=colors, alpha=0.8, edgecolor="black")
    axes[1, 0].set_title("Smoke Q3 (75th Percentile Exposure - lower is better)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Smoke Density")
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    # Min Separation Plot
    axes[1, 1].bar(controllers, min_seps, color=colors, alpha=0.8, edgecolor="black")
    axes[1, 1].set_title("Min Inter-Agent Separation (higher is better)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Separation Distance (meters)")
    axes[1, 1].axhline(
        y=0.8, color="r", linestyle="--", linewidth=1.5, label="Collision Limit (0.8m)"
    )
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, "benchmark_results_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved comparative study chart to '{plot_path}'.")


if __name__ == "__main__":
    main()
