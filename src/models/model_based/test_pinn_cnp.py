
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
import io

# Add src to path
sys.path.append(os.getcwd())

from src.models.model_based.pinn_cnp import PINN_CNP
from src.models.shared.observations import Obs

log = logging.getLogger(__name__)

def create_heatmap(ax, data, title, vmin=0, vmax=1, cmap='plasma', extent=None):
    """
    Plot heatmap of dense data (H, W).
    """
    im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, interpolation='nearest')
    ax.set_title(title)
    return im

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint .pt file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .npz data file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_pinn', help='Directory to save results')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--start_episode', type=int, default=0, help='Index of first episode to evaluate')
    parser.add_argument('--context_frames', type=int, default=10, help='Number of context frames (start of episode)')
    parser.add_argument('--target_frames', type=int, default=15, help='Number of future frames to predict')
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gifs").mkdir(exist_ok=True)

    # 1. Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hyper_params = ckpt['hyper_parameters']
    
    # Initialize Model
    model = PINN_CNP(**hyper_params).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print("Model loaded.")

    # 2. Load Data
    print(f"Loading data from {args.data_path}")
    raw_data = np.load(args.data_path)
    smoke_data = raw_data['smoke_data'] # (E, T, H, W)
    
    # Metadata
    dt = float(raw_data.get('dt', 0.1))
    x_size = float(raw_data.get('x_size', 50.0))
    y_size = float(raw_data.get('y_size', 50.0))
    res_val = float(raw_data.get('resolution', 1.0)) 
    
    # Assume grid size from data shape
    _, _, H, W = smoke_data.shape
    
    extent = [0, x_size, 0, y_size]
    
    num_episodes = min(args.num_episodes, len(smoke_data) - args.start_episode)
    print(f"Evaluating {num_episodes} episodes starting from {args.start_episode}")
    
    metrics_mae = []

    # Prepare Dense Query Grid coordinates (fixed for all steps)
    x_range = np.arange(W) * res_val
    y_range = np.arange(H) * res_val
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    flat_x = torch.from_numpy(grid_x.flatten()).float().to(device).unsqueeze(0).unsqueeze(-1) # (1, P, 1)
    flat_y = torch.from_numpy(grid_y.flatten()).float().to(device).unsqueeze(0).unsqueeze(-1)
    
    # Total sequence length needed
    total_seq_len = args.context_frames + args.target_frames
    
    for i in range(args.start_episode, args.start_episode + num_episodes):
        ep_data = smoke_data[i] # (T_total, H, W)
        
        if ep_data.shape[0] < total_seq_len:
            print(f"Skipping episode {i}: too short ({ep_data.shape[0]} < {total_seq_len})")
            continue
            
        print(f"Episode {i}...")
        
        # 1. Sample Context from first 'context_frames'
        # We sample sparse points from t=0 to t=context_frames-1
        ctx_t_indices = np.arange(args.context_frames)
        
        # Sparse sampling params
        ctx_ratio = 0.2
        num_ctx_pts = int(H * W * ctx_ratio)
        
        # Random sample
        t_idx_c = np.repeat(ctx_t_indices, num_ctx_pts) # This might be too many points if repeating per frame?
        # NonSequentialDataset samples N points TOTAL across time window, not N per frame.
        # Let's match typical training: ~15-25% of total volume? Or per frame?
        # NonSequentialDataset: ctx_min_pts = H*W*ratio. This is total points.
        # So we should sample N points distributed across the time window.
        
        # Sample N random (t, y, x) triplets
        N_ctx = num_ctx_pts
        t_idx_vars = np.random.choice(ctx_t_indices, N_ctx)
        y_idx_vars = np.random.randint(0, H, N_ctx)
        x_idx_vars = np.random.randint(0, W, N_ctx)
        
        # Values
        vals_c = ep_data[t_idx_vars, y_idx_vars, x_idx_vars]
        
        # Coords
        xs_c = (x_idx_vars * res_val).astype(np.float32)
        ys_c = (y_idx_vars * res_val).astype(np.float32)
        ts_c = (t_idx_vars * dt).astype(np.float32)
        
        # To Tensor
        # Batch size 1
        xs_c_t = torch.from_numpy(xs_c).unsqueeze(0).unsqueeze(-1).to(device)
        ys_c_t = torch.from_numpy(ys_c).unsqueeze(0).unsqueeze(-1).to(device)
        ts_c_t = torch.from_numpy(ts_c).unsqueeze(0).unsqueeze(-1).to(device)
        vals_c_t = torch.from_numpy(vals_c).float().unsqueeze(0).unsqueeze(-1).to(device)
        
        ctx_obs = Obs(xs=xs_c_t, ys=ys_c_t, ts=ts_c_t, values=vals_c_t, mask=None)
        
        # 2. Predict Future Frames
        frames = []
        episode_mae = []
        
        # Loop t from 0 to total_seq_len (Context + Target)
        # We visualize reconstruction of context AND prediction of future
        for t_idx in range(total_seq_len):
            t_val = t_idx * dt
            
            # Ground Truth
            gt_frame = ep_data[t_idx]
            
            # Query Model
            flat_t = torch.full_like(flat_x, t_val)
            query_obs = Obs(xs=flat_x, ys=flat_y, ts=flat_t, values=None)
            
            with torch.no_grad():
                output = model(ctx_obs, query_obs)
            
            pred_grid = output.smoke_dist.loc.reshape(H, W).cpu().numpy()
            
            # Error
            abs_err = np.abs(pred_grid - gt_frame)
            mae = np.mean(abs_err)
            
            if t_idx >= args.context_frames:
                episode_mae.append(mae)
            
            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            is_ctx = t_idx < args.context_frames
            stage = "Context" if is_ctx else "Forecast"
            
            create_heatmap(axes[0], gt_frame, f"GT (t={t_idx})", extent=extent)
            create_heatmap(axes[1], pred_grid, f"PINN {stage}", extent=extent)
            im_err = create_heatmap(axes[2], abs_err, f"MAE: {mae:.4f}", vmin=0, vmax=0.5, cmap='magma', extent=extent)
            plt.colorbar(im_err, ax=axes[2])
            
            plt.suptitle(f"Episode {i} | Frame {t_idx} ({stage})")
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.v2.imread(buf))
            plt.close(fig)
            buf.close()
            
        metrics_mae.append(np.mean(episode_mae))
        
        # Save GIF
        gif_path = output_dir / "gifs" / f"ep_{i}_pinn.gif"
        imageio.mimsave(gif_path, frames, fps=5)
        
    print(f"Average Forecasting MAE: {np.mean(metrics_mae):.4f}")
    
if __name__ == "__main__":
    main()
