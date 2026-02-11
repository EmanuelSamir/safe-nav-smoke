
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging
from PIL import Image
import imageio
from scipy.interpolate import griddata

# Add src to path
sys.path.append(os.getcwd())

from src.models.model_free.rnp import RNP, RNPConfig
from src.models.shared.datasets import SequentialDataset, sequential_collate_fn
from src.models.shared.observations import Obs

log = logging.getLogger(__name__)

def create_heatmap(ax, data, title, vmin=0, vmax=1, cmap='viridis', extent=None):
    """
    Plot heatmap of dense data (H, W).
    """
    im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, interpolation='nearest')
    ax.set_title(title)
    return im

def sample_context(episode_data, t_end, num_points_ratio=0.2, device='cpu'):
    """
    Sample context points from episode_data up to t_end.
    episode_data: (T, H, W) numpy
    Returns: Obs (1, t_end, P, 1)
    """
    T, H, W = episode_data.shape
    # We need to create an Obs sequence from 0 to t_end-1
    # For each time step, sample points.
    
    xs_list = []
    ys_list = []
    vals_list = []
    
    # We assume simple square grid for indices logic, but coords need to scale
    # We don't have direct access to 'res' here unless passed, but we can assume [0, spatial_max]
    # Actually, let's pass the dataset or params.
    # For now, let's Assume normalized [0, 30] as per config default? 
    # Better: read from dataset object in main and pass here.
    pass

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint .pt file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .npz data file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--start_episode', type=int, default=0, help='Index of first episode to evaluate')
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # H folders
    h_steps = [1, 5, 10, 15]
    max_h = max(h_steps)
    
    for h in h_steps:
        (output_dir / f"H_{h}").mkdir(exist_ok=True)
        (output_dir / f"H_{h}" / "gifs").mkdir(exist_ok=True)

    # 1. Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['hyper_parameters']
    
    # Initialize Model
    model = RNP(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded.")

    # 2. Load Data
    print(f"Loading data from {args.data_path}")
    # We load efficiently manually to get full episodes
    raw_data = np.load(args.data_path)
    smoke_data = raw_data['smoke_data'] # (E, T, H, W)
    
    # Metadata
    x_size = float(raw_data.get('x_size', 30.0))
    y_size = float(raw_data.get('y_size', 30.0))
    res_val = float(raw_data.get('resolution', 1.0)) # grid resolution (e.g. 1.0 meter)
    
    # Model expects spatial_max. Config has it.
    spatial_max = config.spatial_max
    
    # Coordinate system extent for plots
    extent = [0, x_size, 0, y_size]
    
    num_episodes = min(args.num_episodes, len(smoke_data) - args.start_episode)
    print(f"Evaluating {num_episodes} episodes starting from {args.start_episode}")
    
    # Metrics
    # dict: H -> list of errors
    metrics_mae = {h: [] for h in h_steps}
    
    # Sampling helper
    def get_context_obs(ep_idx, t_current):
        """
        Construct context Obs for episode[ep_idx] from t=0 to t_current (inclusive? no, usually exclusive in python range)
        Let's say t_current is the index of the frame we want to predict NEXT? 
        No, t_current is the current time step. We have observed up to t_current.
        We forecast t_current+1, ...
        So sequence is 0..t_current.
        """
        # To match training distribution, we should sample sparse points.
        # However, for 'visualizing' what the model can do given history, 
        # usually we give it reasonable data. Let's use 20% points.
        
        ep = smoke_data[ep_idx] # (T_total, H, W)
        
        # We need to build Obs (1, T_seq, P, 1)
        # Slicing 0...t_current+1 (inclusive of t_current)
        frames = ep[:t_current+1]
        T_seq, H_grid, W_grid = frames.shape
        
        # We can implement a fast sampler or dense-to-sparse
        # For simplicity, let's randomly sample N points per frame.
        num_points = int(H_grid * W_grid * 0.15) 
        
        # Vectorized sampling
        # We need (1, T_seq, num_points, 1)
        
        # Grid coordinates
        # We need to map grid indices to continuous coords [0, spatial_max]
        # Assuming smoke_data is on grid [0, x_size]x[0, y_size]
        # And model expects [0, spatial_max] (usually 30).
        # We should assume x_size == spatial_max roughly?
        
        # Random indices
        y_idxs = np.random.randint(0, H_grid, size=(T_seq, num_points))
        x_idxs = np.random.randint(0, W_grid, size=(T_seq, num_points))
        
        # Values
        # (T_seq, P)
        # numpy advanced indexing
        t_range = np.arange(T_seq)[:, None]
        vals = frames[t_range, y_idxs, x_idxs] # (T_seq, P)
        
        # Coords
        # x_idxs * resolution
        # Assuming resolution is consistent with x_size. 
        # checking: x_size 30, res 1 -> 30 pixels? 
        # If smoke_data.shape is (T, 30, 30) -> then res=1 is correct.
        
        xs = (x_idxs * res_val).astype(np.float32)
        ys = (y_idxs * res_val).astype(np.float32)
        
        # To Tensor
        xs_t = torch.from_numpy(xs).float().unsqueeze(0).unsqueeze(-1).to(device) # (1, T, P, 1)
        ys_t = torch.from_numpy(ys).float().unsqueeze(0).unsqueeze(-1).to(device)
        vals_t = torch.from_numpy(vals).float().unsqueeze(0).unsqueeze(-1).to(device)
        
        return Obs(xs=xs_t, ys=ys_t, values=vals_t, mask=None, ts=None)

    for i in range(args.start_episode, args.start_episode + num_episodes):
        ep_data = smoke_data[i] # (T_total, H, W)
        T_total = ep_data.shape[0]
        
        print(f"Episode {i}: {T_total} frames. Running Rolling Forecast...")
        
        # Buffers for visuals: H -> list of (T_target, GT_frame, Pred_frame)
        visual_buffers = {h: [] for h in h_steps}
        
        # Start loop
        # We need enough context to start. Let's say 10 frames min.
        start_t = 10
        # We can go up to T_total - max_h
        end_t = T_total - max_h
        
        if end_t <= start_t:
            print(f"Skipping episode {i}: too short.")
            continue
            
        # Iterate
        for t in tqdm(range(start_t, end_t)):
            # 1. Prepare Context
            ctx_obs = get_context_obs(i, t)
            
            # 2. Forecast
            # Returns list of Obs (grids)
            # forecasts[k] corresponds to t + 1 + k
            forecasts = model.forecast(ctx_obs, horizon=max_h)
            
            # 3. Process Forecasts (Metrics & Visuals)
            # forecasts is list of 15 frames (H=1 to 15)
            for k, pred_obs in enumerate(forecasts):
                h_val = k + 1 # Horizon 1..15
                t_target = t + h_val
                
                # 1. Process Grid
                p_vals = pred_obs.values.squeeze().detach().cpu().numpy()
                grid_res = config.grid_res
                pred_grid = p_vals.reshape(grid_res, grid_res)
                
                # 2. GT
                if t_target >= T_total:
                    continue
                gt_frame = ep_data[t_target]
                H_gt, W_gt = gt_frame.shape

                # 3. Scaling Correction
                # Pred Grid covers [0, spatial_max] x [0, spatial_max]
                # GT covers [0, x_size] x [0, y_size]
                # We must crop the Pred Grid to the valid data region before resizing.
                
                spatial_max = config.spatial_max
                
                # Calculate portion of grid that corresponds to x_size / y_size
                # We clamp to 1.0 just in case x_size > spatial_max (unlikely)
                x_ratio = min(x_size / spatial_max, 1.0)
                y_ratio = min(y_size / spatial_max, 1.0)
                
                x_idx_limit = int(grid_res * x_ratio)
                y_idx_limit = int(grid_res * y_ratio)
                
                # Ensure at least 1 pixel
                x_idx_limit = max(x_idx_limit, 1)
                y_idx_limit = max(y_idx_limit, 1)
                
                # Crop (indexing='xy' means shape is (Y, X))
                pred_cropped = pred_grid[:y_idx_limit, :x_idx_limit]
                
                # Resize Cropped Pred to GT size
                pred_t = torch.from_numpy(pred_cropped).float().unsqueeze(0).unsqueeze(0)
                pred_resized = torch.nn.functional.interpolate(pred_t, size=(H_gt, W_gt), mode='bilinear', align_corners=True)
                pred_final = pred_resized.squeeze().numpy()
                
                # 4. Metric
                abs_err = np.abs(pred_final - gt_frame)
                mae = np.mean(abs_err)
                
                if h_val not in metrics_mae:
                    metrics_mae[h_val] = []
                metrics_mae[h_val].append(mae)
                
                # 5. Store Visuals (Only for specific H)
                if h_val in h_steps:
                    visual_buffers[h_val].append({
                        'time': t_target,
                        'gt': gt_frame,
                        'pred': pred_final,
                        'err': abs_err,
                        'mae': mae
                    })
        
        # Generate GIFs for this Episode
        print(f"Generating GIFs for Episode {i}...")
        for h in h_steps:
            frames = []
            buffer = visual_buffers[h]
            if not buffer: continue
            
            for item in buffer:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                t_lbl = item['time']
                
                # GT
                create_heatmap(axes[0], item['gt'], f"GT (t={t_lbl})", extent=extent)
                # Pred
                create_heatmap(axes[1], item['pred'], f"Pred H={h}", extent=extent)
                # Error
                im_err = create_heatmap(axes[2], item['err'], f"MAE: {item['mae']:.4f}", vmin=0, vmax=0.5, cmap='magma', extent=extent)
                plt.colorbar(im_err, ax=axes[2])
                
                plt.suptitle(f"Episode {i} | Horizon {h}")
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                frames.append(imageio.v2.imread(buf))
                plt.close(fig)
                buf.close()
                
            # Save GIF
            gif_path = output_dir / f"H_{h}" / "gifs" / f"ep_{i}_H_{h}.gif"
            imageio.mimsave(gif_path, frames, fps=5)

    # Final Metrics Plot
    print("Plotting Aggregate Metrics...")
    sorted_h = sorted(metrics_mae.keys())
    avg_maes = [np.mean(metrics_mae[h]) for h in sorted_h]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_h, avg_maes, marker='o', linestyle='-')
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Mean Absolute Error (Avg over episodes)")
    plt.title("Forecasting Performance vs Horizon")
    plt.xticks(sorted_h)
    plt.grid(True)
    plt.savefig(output_dir / "forecasting_performance.png")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame({'horizon': sorted_h, 'mae': avg_maes})
    df.to_csv(output_dir / "metrics.csv", index=False)
    
    print("Evaluation Complete.")
    
if __name__ == "__main__":
    import io
    main()
