# Comparison of RNP Models (Standard, Residual, Multistep)

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import logging
import imageio
import time
import argparse
import io
import pandas as pd

# Add src to path
sys.path.append(os.getcwd())

from src.models.model_free.rnp import RNP, RNPConfig
from src.models.model_free.rnp_residual import RNPResidual
from src.models.model_free.rnp_multistep import RNPMultistep
from src.models.shared.observations import Obs, slice_obs
from src.models.shared.datasets import SequentialDataset # Just for utils if needed

log = logging.getLogger(__name__)

def create_heatmap(ax, data, title, vmin=0, vmax=1, cmap='viridis', extent=None):
    """
    Plot heatmap of dense data (H, W).
    """
    im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, interpolation='nearest')
    ax.set_title(title, fontsize=8)
    return im

def get_dense_query_obs(B, device, x_size, y_size, res_val, H_grid, W_grid):
    """
    Create a dense grid of query points covering the entire image.
    Returns Obs (B, 1, P_grid, 0) - values are dummy
    """
    # Create grid coordinates
    
    # NOTE: x_size/y_size in load_data are physical sizes (e.g. 50.0).
    # res_val is resolution (e.g. 1.0 or 0.5?). 
    # H_grid = y_size / res_val ?
    # Actually we have H_grid, W_grid from the data shape.
    
    # We want coordinates corresponding to pixel centers? 
    # Or just grid indices * res.
    
    # x indices: 0..W-1
    # y indices: 0..H-1
    
    # We need to broadcast to (B, 1, ...)
    
    # Generate grid
    # indexing='xy' -> x varies first (W), then y (H). 
    # But usually imshow expects (H, W). 
    # Let's use 'xy' but reshape carefully.
    
    xs_idx, ys_idx = np.meshgrid(np.arange(W_grid), np.arange(H_grid)) # xs: (H, W), ys: (H, W) values
    
    # Flatten
    xs_flat = xs_idx.flatten() # (P,)
    ys_flat = ys_idx.flatten() # (P,)
    
    # Real World Coords -> Normalized Coords [0, 1]
    # Inferred that config expects [0, 1] (mapping to [-1, 1] internally) 
    # to avoid clipping negative values.
    
    xs_norm = (xs_flat * res_val) / x_size
    ys_norm = (ys_flat * res_val) / y_size
    
    # To Tensor
    xs_t = torch.from_numpy(xs_norm).float().to(device)
    ys_t = torch.from_numpy(ys_norm).float().to(device)
    
    # Expand dims for Obs: (B, 1, P, 1)
    xs_t = xs_t.view(1, 1, -1, 1).expand(B, -1, -1, -1)
    ys_t = ys_t.view(1, 1, -1, 1).expand(B, -1, -1, -1)
    
    P = xs_t.shape[2]
    
    return Obs(
        xs=xs_t,
        ys=ys_t,
        values=torch.zeros(B, 1, P, 1, device=device),
        mask=None,
        ts=None
    )

def autoregressive_rollout(model, context_obs, horizon, H_grid, W_grid, x_size, y_size, res_val, model_type='rnp', num_samples=1):
    """
    Perform autoregressive rollout with DENSE reconstruction.
    Supports Single (num_samples=1) and Ensemble (num_samples > 1).
    Returns: 
        - mean_preds: List of (H_grid, W_grid) tensors
        - std_preds: List of (H_grid, W_grid) tensors
        - abs_errs: List of (H_grid, W_grid) tensors (Mean vs GT is computed outside, here we just return Sampled stats)
           Wait, we return Mean and Std. Error is computed vs GT outside.
        - latency_ms: float
    """
    start_time = time.time()
    
    B = num_samples # Virtual Batch Size for Ensemble
    device = context_obs.xs.device
    
    # Expand Context for Ensemble
    # context_obs is (1, T, P, 1)
    # We want (B, T, P, 1)
    
    ctx = Obs(
        xs=context_obs.xs.expand(B, -1, -1, -1),
        ys=context_obs.ys.expand(B, -1, -1, -1),
        values=context_obs.values.expand(B, -1, -1, -1),
        mask=context_obs.mask, # mask is usually None or shared
        ts=context_obs.ts
    )
    
    # Initialize State
    state = model.init_state(batch_size=B, device=device)
    
    # Encode Context Autoregressively
    T_ctx = ctx.xs.shape[1]
    
    for t in range(T_ctx):
        r_step = model.encoder(slice_obs(ctx, t, t+1))
        r_step_sq = r_step.squeeze(1)
        _, state = model.forecaster(r_step_sq, state)
        
    # Prepare DENSE Query
    query_obs = get_dense_query_obs(B, device, x_size, y_size, res_val, H_grid, W_grid)
    
    current_input = slice_obs(ctx, T_ctx-1, T_ctx) 
    
    trajectories = [] # list of (B, 1, P, 1) tensors
    
    steps_generated = 0
    
    while steps_generated < horizon:
        
        # 1. Encode current input
        r_step = model.encoder(current_input)
        r_step_sq = r_step.squeeze(1)
        
        # 2. Forecaster Step
        r_next, state = model.forecaster(r_step_sq, state)
        r_next_expanded = r_next.unsqueeze(1)
        
        # 3. Decode
        if model_type == 'multistep':
            dists = model.decoder(r_next_expanded, query_obs)
            
            chunk_samples = []
            for dist in dists:
                if num_samples > 1:
                    s = dist.sample()
                else:
                    s = dist.mean
                chunk_samples.append(s)
                
            trajectories.extend(chunk_samples)
            steps_generated += len(chunk_samples)
            
            if steps_generated < horizon:
                # Feedback loop for Multistep (updates state)
                for pred_vals in chunk_samples:
                     obs_in = Obs(xs=query_obs.xs, ys=query_obs.ys, values=pred_vals)
                     r_s = model.encoder(obs_in)
                     r_s_sq = r_s.squeeze(1)
                     _, state = model.forecaster(r_s_sq, state)
                
                current_input = Obs(xs=query_obs.xs, ys=query_obs.ys, values=chunk_samples[-1])

        else:
            # Single Step Models
            if 'residual' in model_type:
                # Need r_step (encoding of current input) for decoder
                # Current Encoder r_step (Line 158) was computed on current_input
                # But we need r_step to match the context expected by decoder?
                # Decoder arg name is 'r_step'.
                # Yes, we pass the encoded current input.
                
                # Note: r_step from Line 158 is (B, 1, C, H, W).
                # Decoder expects (B, 1, C, H, W).
                
                dists = model.decoder(r_next_expanded, r_step, query_obs, current_input)
                dist_tp1 = dists[1]
                
                if model_type == 'residual_delta':
                    if steps_generated == 0:
                        # First step: Use full prediction
                        if num_samples > 1:
                            s = dist_tp1.sample()
                        else:
                            s = dist_tp1.mean
                    else:
                        # Subsequent steps: Use Delta logic
                        # y_{t+1} = y_t + delta
                        # y_t is current_input.values
                        
                        # Get Delta components
                        # We hacked RNPResidual to attach 'components'
                        comps = getattr(dist_tp1, 'components', None)
                        if comps is None:
                            raise RuntimeError("Residual model did not return components!")
                            
                        delta_mu = comps['delta_mu']
                        delta_sigma = comps['delta_sigma']
                        
                        dist_delta = torch.distributions.Normal(delta_mu, delta_sigma)
                        
                        if num_samples > 1:
                            delta = dist_delta.sample()
                        else:
                            delta = delta_mu # or dist_delta.mean
                            
                        s = current_input.values + delta
                else:
                    # Standard Residual
                    if num_samples > 1:
                        s = dist_tp1.sample()
                    else:
                        s = dist_tp1.mean
            else:
                # RNP
                dist = model.decoder(r_next_expanded, query_obs)
                if num_samples > 1:
                    s = dist.sample()
                else:
                    s = dist.mean
            
            trajectories.append(s)
            steps_generated += 1
            
            # Update input
            current_input = Obs(xs=query_obs.xs, ys=query_obs.ys, values=s)
            
    trajectories = trajectories[:horizon]
    
    # Stack: (B, H, P, 1)
    stack = torch.stack([t.squeeze(1) for t in trajectories], dim=1) 
    
    # Compute Stats
    mean_preds_t = stack.mean(dim=0) # (H, P, 1)
    std_preds_t = stack.std(dim=0)   # (H, P, 1)
    
    # Convert to Images
    mean_imgs = []
    std_imgs = []
    
    for t in range(horizon):
        # Mean
        v_mean = mean_preds_t[t].squeeze().detach().cpu().numpy()
        mean_imgs.append(v_mean.reshape(H_grid, W_grid))
        
        # Std
        v_std = std_preds_t[t].squeeze().detach().cpu().numpy()
        std_imgs.append(v_std.reshape(H_grid, W_grid))
        
    end_time = time.time()
    latency_ms = ((end_time - start_time) * 1000.0) / num_samples
    
    return mean_imgs, std_imgs, latency_ms

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    try:
        raw_data = np.load(data_path)
        # Check keys
        if 'smoke_data' not in raw_data:
             print(f"Keys found: {list(raw_data.keys())}")
             raise KeyError("smoke_data not found in npz")
        smoke_data = raw_data['smoke_data'] # (E, T, H, W)
        
        x_size = float(raw_data.get('x_size', 50.0))
        y_size = float(raw_data.get('y_size', 50.0))
        res_val = float(raw_data.get('resolution', 1.0))
        return smoke_data, x_size, y_size, res_val
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def get_context_obs(ep_data, t_end, device, x_size, res_val, y_size):
    """
    Get SPARSE context Obs up to t_end.
    To match training, use random points? Or use dense context?
    User said "use all xs and ys" which might imply dense context too.
    Let's use Dense Context to give best possible information?
    Or Sparse to match training?
    Let's go with 20% points as before to be consistent with training, BUT
    use the new dense decoding for output. 
    Actually, let's bump it to 50% for "better inference" if requested.
    """
    frames = ep_data[:t_end+1] # (T, H, W)
    T, H_grid, W_grid = frames.shape
    
    # Let's use 50% points for context
    num_points = int(H_grid * W_grid * 0.5)
    
    y_idxs = np.random.randint(0, H_grid, size=(T, num_points))
    x_idxs = np.random.randint(0, W_grid, size=(T, num_points))
    
    # Advanced indexing
    t_range = np.arange(T)[:, None]
    vals = frames[t_range, y_idxs, x_idxs] # (T, P)
    
    # Normalize Coords to [0, 1] for model input (which maps 0..1 to -1..1)
    xs_norm = (x_idxs * res_val) / x_size
    ys_norm = (y_idxs * res_val) / y_size
    
    # Convert to Tensor
    xs = torch.from_numpy(xs_norm).float().unsqueeze(0).unsqueeze(-1).to(device)
    ys = torch.from_numpy(ys_norm).float().unsqueeze(0).unsqueeze(-1).to(device)
    vals_t = torch.from_numpy(vals).float().unsqueeze(0).unsqueeze(-1).to(device)
    
    return Obs(xs=xs, ys=ys, values=vals_t)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_rnp', type=str, required=True)
    parser.add_argument('--ckpt_res', type=str, required=True)
    parser.add_argument('--ckpt_multi', type=str, required=True)
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='comparison_results')
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--stride', type=int, default=3, help='Stride for rolling forecast')
    parser.add_argument('--num_samples', type=int, default=10, help='Ignored (Single mode only)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Make subdirs for horizon GIFs
    horizons_to_vis = [h for h in [1, 5, 10, 15] if h <= args.horizon]
    for h in horizons_to_vis:
        (out_dir / f"H_{h}").mkdir(exist_ok=True)

    # 1. Load Models
    models = {}
    configs = {}
    
    def load_model(name, cls, path):
        print(f"Loading {name} from {path}...")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = ckpt['hyper_parameters']
        if isinstance(cfg_dict, dict):
            cfg = RNPConfig(**cfg_dict)
        else:
            cfg = cfg_dict
            
        if cls == RNPMultistep:
             model = cls(cfg, forecast_horizon=5)
        else:
             model = cls(cfg)
             
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        return model, cfg

    models['rnp'], configs['rnp'] = load_model('RNP', RNP, args.ckpt_rnp)
    models['residual'], configs['residual'] = load_model('Residual', RNPResidual, args.ckpt_res)
    models['multistep'], configs['multistep'] = load_model('Multistep', RNPMultistep, args.ckpt_multi)

    # 2. Data
    smoke_data, x_size, y_size, res_val = load_data(args.data_path)
    # Get shape from first episode
    _, H_grid, W_grid = smoke_data[0].shape
    
    # Modes to evaluate
    metric_keys = ['rnp', 'residual', 'residual_delta', 'multistep']
    metrics = {k: {h: [] for h in range(1, args.horizon+1)} for k in metric_keys}
    latencies = {k: [] for k in metric_keys}
    
    indices = np.linspace(0, len(smoke_data)-1, args.num_episodes, dtype=int)
    
    for ep_idx in tqdm(indices, desc="Episodes"):
        ep_data = smoke_data[ep_idx]
        T_total = ep_data.shape[0]
        
        t_start = 10
        t_end = T_total - args.horizon - 1
        
        if t_end <= t_start: 
            print(f"Skipping episode {ep_idx} (Too short)")
            continue
            
        # Buffers for this episode's GIFs
        episode_frames = {h: [] for h in horizons_to_vis}
        
        stride = args.stride
        time_steps = range(t_start, t_end, stride)
        
        for t_current in tqdm(time_steps, desc=f"Ep {ep_idx} (Stride {stride})", leave=False):
            
            # Common Context
            t0 = time.time()
            ctx_obs = get_context_obs(ep_data, t_current, device, x_size, res_val, y_size)
            print(f"  [Time] Context: {time.time()-t0:.3f}s")
            
            gt_frames_horizon = ep_data[t_current+1 : t_current+1+args.horizon]
            
            # Store results for visualization
            results_mean = {} # model -> list of images (horizon)
            results_std = {} 
            results_err = {}
            
            for m_name in metric_keys:
                # Resolve Model and Type
                if m_name == 'multistep':
                    model = models['multistep']
                    model_type = 'multistep'
                elif m_name == 'residual':
                    model = models['residual']
                    model_type = 'residual'
                elif m_name == 'residual_delta':
                    model = models['residual'] # Same model instance
                    model_type = 'residual_delta'
                else:
                    model = models['rnp']
                    model_type = 'rnp'
                
                # Rollout
                preds_mean, preds_std, lat = autoregressive_rollout(
                    model, ctx_obs, args.horizon, H_grid, W_grid, x_size, y_size, res_val, 
                    model_type=model_type, num_samples=args.num_samples
                )
                
                latencies[m_name].append(lat)
                results_mean[m_name] = preds_mean
                results_std[m_name] = preds_std
                
                # Compute Errors vs GT
                m_errs = []
                for h_idx, pred_img in enumerate(preds_mean):
                    gt = gt_frames_horizon[h_idx]
                    abs_diff = np.abs(pred_img - gt)
                    mae = np.mean(abs_diff)
                    metrics[m_name][h_idx+1].append(mae)
                    m_errs.append(abs_diff)
                    
                results_err[m_name] = m_errs

            # Visualization
            # Grid: Rows = Models, Cols = [Mean, Std, Error]
            # AND GT?
            # Let's put GT in a separate row at the top? OR as a column?
            # User wants: "sample mean y std. Y el error abs con el sampled."
            # And obviously GT to compare.
            
            # Layout: 
            # Cols: GT | Mean | Std | Error
            # Rows: RNP | Residual | ResDelta | Multistep
            
            # 4 models * 4 cols = 16 plots per horizon.
            
            for h in horizons_to_vis:
                h_idx = h - 1
                gt_frame = gt_frames_horizon[h_idx]
                
                n_models = len(metric_keys)
                fig, axes = plt.subplots(n_models, 4, figsize=(16, 3.5 * n_models))
                
                for r, m_name in enumerate(metric_keys):
                    # 1. GT
                    # We repeat GT for each row for easy comparison? Or just once?
                    # Easy comparison is better.
                    create_heatmap(axes[r, 0], gt_frame, "Ground Truth", vmin=0, vmax=1)
                    
                    # 2. Mean
                    pred_mean = results_mean[m_name][h_idx]
                    mae = np.mean(np.abs(pred_mean - gt_frame))
                    create_heatmap(axes[r, 1], pred_mean, f"{m_name.upper()} Mean\nMAE: {mae:.3f}", vmin=0, vmax=1)
                    
                    # 3. Std
                    pred_std = results_std[m_name][h_idx]
                    # Std range? usually small. Let's auto-scale or fix?
                    # Fix to 0.5?
                    create_heatmap(axes[r, 2], pred_std, f"Std Dev", vmin=0, vmax=0.2, cmap='magma')
                    
                    # 4. Abs Error
                    pred_err = results_err[m_name][h_idx]
                    create_heatmap(axes[r, 3], pred_err, f"Abs Error", vmin=0, vmax=0.5, cmap='hot')
                    
                    # Hide axes
                    for ax in axes[r]: ax.axis('off')
                
                plt.suptitle(f"Ep {ep_idx} | t={t_current} | Horizon +{h} | Samples={args.num_samples}", fontsize=16)
                plt.tight_layout()
                
                try:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    img = imageio.v2.imread(buf)
                    episode_frames[h].append(img)
                    buf.close()
                except Exception as e:
                     print(f"Plot Error: {e}")
                finally:
                     plt.close(fig)
                     
            # print(f"  [Time] Step Tot: {time.time()-t0:.3f}s") # Keeping noisy prints low?
        
        # Save GIFs
        for h in horizons_to_vis:
            if len(episode_frames[h]) > 0:
                gif_path = out_dir / f"H_{h}" / f"ep_{ep_idx}_dense.gif"
                imageio.mimsave(gif_path, episode_frames[h], fps=5)
    
    # 4. Summary Plots
    plt.figure(figsize=(10, 6))
    horizons = range(1, args.horizon+1)
    colors = {'rnp': 'blue', 'residual': 'green', 'multistep': 'red', 'residual_delta': 'purple'}
    
    for m_name in metric_keys:
        maes = [np.nanmean(metrics[m_name][h]) for h in horizons]
        plt.plot(horizons, maes, marker='o', label=m_name, color=colors[m_name])
        
    plt.xlabel("Horizon")
    plt.ylabel("MAE")
    plt.title("Error Growth (Dense Reconstruction)")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "error_growth.png")
    
    # Latency Bar
    plt.figure(figsize=(8, 6))
    means = [np.nanmean(latencies[k]) for k in metric_keys]
    stds = [np.nanstd(latencies[k]) for k in metric_keys]
    plt.bar(metric_keys, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency")
    plt.tight_layout()
    plt.savefig(out_dir / "latency.png")
    
    # CSV
    rows = []
    for h in horizons:
        row = {'horizon': h}
        for k in metric_keys:
            row[f'{k}_mae'] = np.nanmean(metrics[k][h])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    
    print("Comparison Complete.")

if __name__ == "__main__":
    main()
