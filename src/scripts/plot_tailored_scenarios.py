import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Tailored blobs from the original script
    tailored_blobs = {
        "case_1": {
            "x_pos": [10.0, 10.0, 10.0, 20.0, 20.0],
            "y_pos": [3.0, 10.0, 17.0, 6.0, 14.0],
        },
        "case_2": {
            "x_pos": [20.0, 20.0, 20.0, 10.0, 10.0],
            "y_pos": [3.0, 10.0, 17.0, 6.0, 14.0],
        },
        "case_3": {
            "x_pos": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "y_pos": [3.0, 8.0, 13.0, 7.0, 12.0, 17.0],
        },
        "case_4": {
            "x_pos": [20.0, 20.0, 20.0, 10.0, 10.0, 10.0],
            "y_pos": [3.0, 8.0, 13.0, 7.0, 12.0, 17.0],
        },
        "case_5": {
            "x_pos": [8.0, 8.0, 15.0, 22.0, 22.0],
            "y_pos": [4.0, 16.0, 10.0, 4.0, 16.0],
        },
        "case_6": {
            "x_pos": [8.0, 15.0, 15.0, 15.0, 22.0],
            "y_pos": [10.0, 4.0, 10.0, 16.0, 10.0],
        },
    }

    # Robot start and end coordinates based on base.yaml
    start_pos = (3.0, 10.0)
    goal_pos = (28.0, 10.0)
    
    # Grid limits based on x_size=30, y_size=20
    x_lim = (0, 30)
    y_lim = (0, 20)

    # Styling for IEEE paper
    plt.rc('font', family='serif', size=10)
    plt.rc('axes', titlesize=10, labelsize=9)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)

    # 1x6 Grid
    fig, axes = plt.subplots(1, 6, figsize=(7.16*1.5, 1.5)) # IEEE full width is ~7.16 inches
    axes = axes.flatten()

    for i, (case_name, data) in enumerate(tailored_blobs.items()):
        ax = axes[i]
        
        # Plot smoke sources using patches for accurate radius representation
        # Minimum radius (1.0) and Maximum radius (3.0)
        for x_p, y_p in zip(data["x_pos"], data["y_pos"]):
            circle_max = plt.Circle((x_p, y_p), 3.0, facecolor='gray', alpha=0.3, edgecolor='none')
            circle_min = plt.Circle((x_p, y_p), 1.0, facecolor='gray', alpha=0.6, edgecolor='black')
            ax.add_patch(circle_max)
            ax.add_patch(circle_min)
            # Epicenter
            ax.scatter(x_p, y_p, color='black', s=10, marker='x', zorder=3)

        # Create dummy handles for legend
        ax.scatter([], [], facecolor='gray', alpha=0.3, label='Max Spread Radius (3.0m)', edgecolor='none')
        ax.scatter([], [], facecolor='gray', alpha=0.6, label='Min Spread Radius (1.0m)', edgecolor='black')
        ax.scatter([], [], color='black', s=10, marker='x', label='Smoke Center')

        # Plot Start and Goal positions
        ax.scatter(start_pos[0], start_pos[1], color='green', s=100, marker='^', label='Start', zorder=5)
        ax.scatter(goal_pos[0], goal_pos[1], color='dodgerblue', s=150, marker='*', label='Goal', zorder=5)

        # Connect Start and Goal with a dashed line loosely to represent trajectory intent
        ax.plot([start_pos[0], goal_pos[0]], [start_pos[1], goal_pos[1]], color='gray', linestyle='--', alpha=0.5, zorder=1)

        ax.set_title(f"Scenario {i+1}")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.set_xlabel("X (m)")
            
        if i == 0:
            ax.set_ylabel("Y (m)")
        else:
            ax.set_yticklabels([])

    # Add a single legend for the entire figure at the top or bottom
    handles, labels = axes[0].get_legend_handles_labels()
    # Filter out duplicate handles
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.3), frameon=False, prop={'size': 7})

    plt.tight_layout()
    plt.subplots_adjust(top=0.8) # Make space for the legend

    # Save to file
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "tailored_scenarios_plot.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {out_path}")
    
    # Also save as PNG for quick viewing
    png_path = out_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {png_path}")

    # Optionally show
    # plt.show()

if __name__ == "__main__":
    main()
