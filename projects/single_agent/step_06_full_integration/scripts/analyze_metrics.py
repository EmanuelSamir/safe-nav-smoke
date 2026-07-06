# %% [markdown]
# # Analysis of Full Integration Experiments
# 
# This interactive script loads the raw transitions datasets from the experiments,
# computes safety and performance metrics, and plots comparisons.

# %%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from pathlib import Path
import yaml

# %%
EXPERIMENTS = {
    "No Risk": "no_risk",
    "HOCBF": "cbf",
    "Persistent MPPI": "persistent",
    "FNO MPPI": "fno"
}

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_dir = os.path.join(base_dir, "outputs/integration")
print(f"Base outputs directory: {outputs_dir}")

# %% [markdown]
# ## 1. Data Collection
# Iterating over the HuggingFace datasets to calculate episode metrics.

# %%
# We dynamically load grid resolution and dimensions from the underlying Playback dataset
# We can read any config file since they all use the same playback dataset
config_path = os.path.join(base_dir, "projects/single_agent/step_06_full_integration/config_no_risk.yaml")
with open(config_path, "r") as f:
    integration_cfg = yaml.safe_load(f)

playback_data_path = integration_cfg.get("simulator", {}).get("data_path", "data/structured_smoke_slow")
playback_ds_path = os.path.join(base_dir, playback_data_path) if not os.path.isabs(playback_data_path) else playback_data_path

if not os.path.exists(playback_ds_path):
    raise FileNotFoundError(f"Cannot find playback dataset at {playback_ds_path} to read grid properties.")

playback_ds = load_from_disk(playback_ds_path)
first_row = playback_ds[0]

if "resolution" not in first_row or "y_size" not in first_row or "x_size" not in first_row:
    raise ValueError(f"Dataset at {playback_ds_path} is missing required 'resolution', 'x_size' or 'y_size' properties.")

RESOLUTION = float(first_row["resolution"])
Y_SIZE = float(first_row["y_size"])
X_SIZE = float(first_row["x_size"])

H = int(Y_SIZE / RESOLUTION)
W = int(X_SIZE / RESOLUTION)
print(f"Grid setup: {H}x{W} at {RESOLUTION}m resolution")

all_data = []

for exp_name, exp_folder in EXPERIMENTS.items():
    exp_dir = os.path.join(outputs_dir, exp_folder)
    
    if not os.path.exists(exp_dir):
        print(f"Warning: Directory not found for '{exp_name}': {exp_dir}")
        continue
        
    try:
        ds = load_from_disk(exp_dir)
        ds = ds.with_format("numpy")
    except Exception as e:
        print(f"Failed to load dataset for {exp_name}: {e}")
        continue
        
    print(f"Processing {exp_name} with {len(ds)} transitions...")
    
    # We must group transitions into episodes based on terminations
    episodes = []
    current_ep = []
    
    for row in ds:
        current_ep.append(row)
        if row["terminations"] or row["truncations"]:
            episodes.append(current_ep)
            current_ep = []
    
    # If the last episode was not terminated properly but has data
    if len(current_ep) > 0:
        episodes.append(current_ep)
        
    # Compute metrics per episode
    for idx, ep in enumerate(episodes):
        smoke_on_robot_vals = []
        reached_goal = False
        time_taken = len(ep) * 0.1 # assuming dt = 0.1
        
        for transition in ep:
            if "smoke_in_robot" in transition:
                smoke_val = transition["smoke_in_robot"]
            else:
                loc = transition["obs_location"]
                readings = transition["obs_readings"]
                
                # Since sensor is global, readings is the full flattened grid
                # We map loc to grid coords to find the smoke at robot position
                y_coords = (loc[1] / RESOLUTION) - 0.5
                x_coords = (loc[0] / RESOLUTION) - 0.5
                
                y_idx = int(np.clip(round(y_coords), 0, H-1))
                x_idx = int(np.clip(round(x_coords), 0, W-1))
                
                try:
                    full_map = readings.reshape(H, W)
                    smoke_val = full_map[y_idx, x_idx]
                except Exception:
                    # If dimensions mismatch due to different resolution, we can fallback safely
                    smoke_val = 0.0
                
            smoke_on_robot_vals.append(smoke_val)
            
            if transition["reward"] > 0.0:
                reached_goal = True
                
        max_smoke = max(smoke_on_robot_vals) if smoke_on_robot_vals else np.nan
        mean_smoke = np.mean(smoke_on_robot_vals) if smoke_on_robot_vals else np.nan
        accumulated_smoke = sum(smoke_on_robot_vals) if smoke_on_robot_vals else np.nan
        
        all_data.append({
            'Experiment': exp_name,
            'Episode': idx,
            'Max Smoke': max_smoke,
            'Mean Smoke': mean_smoke,
            'Time to Goal': time_taken,
            'Reached Goal': reached_goal,
            'Accumulated Smoke': accumulated_smoke
        })

results_df = pd.DataFrame(all_data)
if results_df.empty:
    print("No data extracted!")
else:
    print(f"Successfully loaded {len(results_df)} total episodes.")

# %% [markdown]
# ## 2. Summary Statistics

# %%
print("-" * 50)
print(f"{'Experiment':<20} | {'Success Rate':<12} | {'Avg Max Smoke':<15} | {'Avg Time (s)'}")
print("-" * 50)

summary_stats = []
for exp_name in EXPERIMENTS.keys():
    exp_data = results_df[results_df['Experiment'] == exp_name]
    if exp_data.empty:
        continue
        
    success_rate = exp_data['Reached Goal'].mean() * 100
    avg_max_smoke = exp_data['Max Smoke'].mean()
    
    successful_episodes = exp_data[exp_data['Reached Goal']]
    avg_time = successful_episodes['Time to Goal'].mean() if not successful_episodes.empty else np.nan
    
    print(f"{exp_name:<20} | {success_rate:>6.2f}%      | {avg_max_smoke:>13.4f} | {avg_time:>10.2f}")

print("-" * 50)

# %% [markdown]
# ## 3. Visualizations

# %%
# Plot Formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
})

colors = sns.color_palette("colorblind", n_colors=len(EXPERIMENTS))
palette = {name: colors[i] for i, name in enumerate(EXPERIMENTS.keys())}

# %%
# Plot: Success Rate
plt.figure(figsize=(8, 6))
success_rates = results_df.groupby('Experiment')['Reached Goal'].mean() * 100
# Reorder according to EXPERIMENTS dict
success_rates = success_rates.reindex(list(EXPERIMENTS.keys())).dropna()

sns.barplot(x=success_rates.index, y=success_rates.values, hue=success_rates.index, palette=palette, legend=False)
plt.title("Navigation Success Rate")
plt.ylabel("Success Rate (%)")
plt.xlabel("Method")
plt.ylim(0, 110)
plt.tight_layout()
plt.show()

# %%
# Plot: Maximum Smoke Inhaled (Only for successful episodes)
success_df = results_df[results_df['Reached Goal'] == True]

if not success_df.empty:
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=success_df, 
        x="Experiment", 
        y="Max Smoke", 
        order=[k for k in EXPERIMENTS.keys() if k in success_df['Experiment'].unique()],
        hue="Experiment", 
        palette=palette, 
        legend=False,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
    )
    plt.title("Maximum Smoke Density Encountered (Successful Episodes)")
    plt.ylabel("Max Smoke Density")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.show()

# %%
# Plot: Time to Goal
if not success_df.empty:
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=success_df, 
        x="Experiment", 
        y="Time to Goal", 
        order=[k for k in EXPERIMENTS.keys() if k in success_df['Experiment'].unique()],
        hue="Experiment", 
        palette=palette, 
        legend=False,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
    )
    plt.title("Time to Goal (Successful Episodes)")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.show()
