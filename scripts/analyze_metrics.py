# %% [markdown]
# # Metrics Analysis and Comparison
# This is an interactive Python script. You can run individual cells in VS Code, PyCharm, or Jupyter.
# Make sure to update the `EXPERIMENTS` dictionary below with the names and paths of the experiments you want to compare.

# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_smoke_val(val_str):
    """
    Parses the smoke_on_robot which comes as a string representation of a list, e.g., '[0.]'
    """
    if pd.isna(val_str):
        return np.nan
    try:
        if isinstance(val_str, str):
            return float(val_str.strip('[] \n'))
        return float(val_str)
    except Exception:
        return np.nan

# %%
# Define your experiments here:
# Key: Name to display in plots (Title/Legend)
# Value: Folder name inside the `outputs/` directory 
# (e.g. if the path is outputs/cbf, value is "cbf")
EXPERIMENTS = {
    "CBF Controller": "cbf",
    "No Risk": "no_risk",
    "Static MPPI": "persistent",
    "Proposal": "behavior_prediction"
    # Add more experiments here to compare:
    # "Baseline Controller": "baseline",
    # "Other Method": "other_method"
}

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_dir = os.path.join(base_dir, "outputs")

print(f"Base outputs directory: {outputs_dir}")
print(f"Configured experiments to compare: {list(EXPERIMENTS.keys())}")

# %% [markdown]
# ## 1. Data Collection
# Iterating over all configured experiments to load episode metrics.

# %%
all_data = []

for exp_name, exp_folder in EXPERIMENTS.items():
    exp_dir = os.path.join(outputs_dir, exp_folder)
    
    if not os.path.exists(exp_dir):
        print(f"Warning: Directory not found for '{exp_name}': {exp_dir}")
        continue
        
    metrics_files = glob.glob(os.path.join(exp_dir, "ep_*", "metrics.csv"))
    
    if not metrics_files:
        print(f"Warning: No metrics.csv found for '{exp_name}' in {exp_dir}")
        continue
        
    print(f"Found {len(metrics_files)} episodes for '{exp_name}'.")
    
    for file in metrics_files:
        try:
            df = pd.read_csv(file)
            if df.empty:
                continue
                
            ep_folder = os.path.basename(os.path.dirname(file))
            
            # Smoke values
            if 'smoke_on_robot' in df.columns:
                smoke_vals = df['smoke_on_robot'].apply(parse_smoke_val)
                max_smoke = smoke_vals.max()
                mean_smoke = smoke_vals.mean()
                accumulated_smoke = smoke_vals.sum()
            else:
                max_smoke = np.nan
                mean_smoke = np.nan
                
            # Goal status
            if 'status' in df.columns:
                last_status = df['status'].iloc[-1]
                reached_goal = (last_status == 'reached_goal')
            else:
                reached_goal = False
                
            # Time to goal
            if 'time' in df.columns:
                time_taken = df['time'].iloc[-1]
            else:
                time_taken = np.nan
                
            all_data.append({
                'Experiment': exp_name,
                'Episode': ep_folder,
                'Max Smoke': max_smoke,
                'Mean Smoke': mean_smoke,
                'Time to Goal': time_taken,
                'Reached Goal': reached_goal,
                'Accumulated Smoke': accumulated_smoke
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Create centralized dataframe
results_df = pd.DataFrame(all_data)

if results_df.empty:
    raise ValueError("No valid data could be extracted from any experiment.")
else:
    print(f"\nSuccessfully loaded {len(results_df)} total episodes across all experiments.")

# %% [markdown]
# ## 2. Summary Statistics
# Let's print the general stats for each experiment.

# %%
print("-" * 50)
print(f"{'Experiment':<20} | {'Success Rate':<12} | {'Avg Max Smoke':<15} | {'Avg Time (Success)'}")
print("-" * 50)

summary_stats = []
for exp_name in EXPERIMENTS.keys():
    exp_data = results_df[results_df['Experiment'] == exp_name]
    if exp_data.empty:
        continue
        
    success_rate = exp_data['Reached Goal'].mean() * 100
    avg_max_smoke = exp_data['Max Smoke'].mean()
    avg_time = exp_data[exp_data['Reached Goal']]['Time to Goal'].mean()
    
    print(f"{exp_name:<20} | {success_rate:>6.2f}%      | {avg_max_smoke:>13.4f} | {avg_time:>10.2f} s")

print("-" * 50)

# %% [markdown]
# ## 3. Visualizations
# Creating boxplots for distributions and barplots for success rates.

# %%
sns.set_theme(style="whitegrid")

# 1. Violinplot: Max Smoke
plt.figure(figsize=(8, 6))
sns.violinplot(data=results_df, x='Experiment', y='Max Smoke', palette="Set2")
plt.title('Distribution of Max Smoke Value', fontsize=14)
plt.ylabel('Max Smoke')
plt.xlabel('')
plt.tight_layout()
plt.show()

# 2. Violinplot: Mean Smoke
plt.figure(figsize=(8, 6))
sns.violinplot(data=results_df, x='Experiment', y='Mean Smoke', palette="Set2")
plt.title('Distribution of Mean Smoke Value', fontsize=14)
plt.ylabel('Mean Smoke')
plt.xlabel('')
plt.tight_layout()
plt.show()

# 3. Violinplot: Time to Goal (Successful episodes only)
plt.figure(figsize=(8, 6))
success_df = results_df[results_df['Reached Goal'] == True]
if not success_df.empty:
    sns.violinplot(data=success_df, x='Experiment', y='Time to Goal', palette="Set2")
plt.title('Time to Goal (Successful episodes only)', fontsize=14)
plt.ylabel('Time (s)')
plt.xlabel('')
plt.tight_layout()
plt.show()

# 4. Barplot: Reached Goal comparison
plt.figure(figsize=(8, 6))
counts_df = results_df.groupby(['Experiment', 'Reached Goal']).size().reset_index(name='Count')
counts_df['Status'] = counts_df['Reached Goal'].map({True: 'Reached Goal', False: 'Failed'})

ax = sns.barplot(data=counts_df, x='Experiment', y='Count', hue='Status')
plt.title('Episodes Success Status', fontsize=14)
plt.ylabel('Number of Episodes')
plt.xlabel('')

# Annotate bars
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.show()


# %%
