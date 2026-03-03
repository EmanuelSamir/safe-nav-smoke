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
    "No Risk": "no_risk",
    "HOCBF": "cbf",
    "Static MPPI": "persistent",
    "PFNO-MPPI": "behavior_prediction"
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
# IEEE Paper Plot Formatting
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
})

# 1. Boxplots for distributions in one figure (Fig. Y)
# A typical IEEE 2-column figure width is ~7.16 inches
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))

sns.boxplot(data=results_df, x='Experiment', y='Max Smoke', hue='Experiment', legend=False, ax=axes[0], palette="Set2", fliersize=1)
axes[0].set_title('Max. Smoke Exposure')
axes[0].set_ylabel('Max Smoke')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=15)

sns.boxplot(data=results_df, x='Experiment', y='Mean Smoke', hue='Experiment', legend=False, ax=axes[1], palette="Set2")
axes[1].set_title('Mean Smoke Exposure')
axes[1].set_ylabel('Mean Smoke')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=15)

success_df = results_df[results_df['Reached Goal'] == True]
if not success_df.empty:
    sns.boxplot(data=success_df, x='Experiment', y='Time to Goal', hue='Experiment', legend=False, ax=axes[2], palette="Set2")
axes[2].set_title('Total Navigation Time')
axes[2].set_ylabel('Time (s)')
axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
boxplot_path_pdf = os.path.join(outputs_dir, "ieee_metrics_boxplot.pdf")
boxplot_path_png = os.path.join(outputs_dir, "ieee_metrics_boxplot.png")
plt.savefig(boxplot_path_pdf, bbox_inches='tight')
plt.savefig(boxplot_path_png, bbox_inches='tight')
plt.show()
print(f"Saved box plot to {boxplot_path_pdf} and {boxplot_path_png}")

# 2. Barplot: Reached Goal comparison
# A typical single column IEEE plot is ~3.5 inches width
plt.figure(figsize=(3.5, 3))
counts_df = results_df.groupby(['Experiment', 'Reached Goal']).size().reset_index(name='Count')
counts_df['Status'] = counts_df['Reached Goal'].map({True: 'Reached Goal', False: 'Failed'})

ax = sns.barplot(data=counts_df, x='Experiment', y='Count', hue='Status', palette="Set2")
plt.title('Episodes Success Status')
plt.ylabel('Number of Episodes')
plt.xlabel('')
plt.xticks(rotation=20, ha='right')
plt.legend(title='')

# Annotate bars
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 2), textcoords='offset points')

plt.tight_layout()
barplot_path_pdf = os.path.join(outputs_dir, "ieee_success_barplot.pdf")
barplot_path_png = os.path.join(outputs_dir, "ieee_success_barplot.png")
plt.savefig(barplot_path_pdf, bbox_inches='tight')
plt.savefig(barplot_path_png, bbox_inches='tight')
plt.show()
print(f"Saved bar plot to {barplot_path_pdf} and {barplot_path_png}")


# %%
