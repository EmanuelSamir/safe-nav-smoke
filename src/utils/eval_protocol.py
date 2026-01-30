
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any, List

from src.models.model_free.utils import ObsSNP
from src.models.model_based.utils import ObsPINN

def evaluate_forecast_protocol(model, loader, device, model_type="model_free"):
    """
    Standardized benchmarking protocol:
    Evaluates the model's ability to forecast the next 15 frames given the first 10 frames.
    
    Args:
        model: The trained model.
        loader: DataLoader for the validation set. 
                - For model_free, expected sequence length >= 25.
                - For model_based, expected (context, target) split where context=10, target=15.
        device: Torch device.
        model_type: "model_free" or "model_based".
        
    Returns:
        float: Average MSE across all evaluation points.
    """
    model.eval()
    total_mse = 0.0
    total_points = 0
    
    with torch.no_grad():
        for batch in loader:
            # --- MODEL FREE (Sequential) ---
            if model_type == "model_free":
                # Batch is a list of T steps (dictionaries)
                # We require at least 25 steps (10 context + 15 target)
                if len(batch) < 25:
                    continue
                
                # Split
                context_steps = batch[:10]
                target_steps = batch[10:25]
                
                # Determine batch size from first step
                bs = batch[0]['action'].shape[0]
                state = model.init_state(bs, device)
                
                # 1. Context Phase (Teacher Forcing / Encoding)
                for step in context_steps:
                    obs = step['obs'].to(device)
                    action = step['action'].to(device)
                    done = step['done'].to(device)
                    
                    # Updates hidden state
                    out = model(state, action, done, obs=obs)
                    state = out.state
                
                # 2. Forecast Phase (Open Loop Prediction)
                for step in target_steps:
                    gt_obs = step['obs'].to(device)
                    action = step['action'].to(device)
                    done = step['done'].to(device)
                    
                    # Predict at gt_obs coordinates (Query)
                    # Pass obs=None to force prediction from state
                    out = model(state, action, done, obs=None, query=gt_obs)
                    state = out.state
                    
                    # Get prediction (mean)
                    pred = out.decoded if hasattr(out, 'decoded') else out.prediction
                    if hasattr(pred, 'loc'): pred = pred.loc
                    
                    # Compute MSE vs Ground Truth
                    # gt_obs.values: (B, N) or (B, N, 1)
                    gt_val = gt_obs.values
                    if gt_val.dim() == 2: gt_val = gt_val.unsqueeze(-1)
                    if pred.dim() == 2: pred = pred.unsqueeze(-1)
                    
                    sq_err = (pred - gt_val) ** 2
                    
                    if gt_obs.mask is not None:
                        # Mask invalid/padded points
                        mask = gt_obs.mask.float().unsqueeze(-1)
                        mse_sum = (sq_err * mask).sum()
                        pts_count = mask.sum()
                    else:
                        mse_sum = sq_err.sum()
                        pts_count = torch.tensor(sq_err.numel(), device=device)
                        
                    total_mse += mse_sum.item()
                    total_points += pts_count.item()

            # --- MODEL BASED (PINN / CNP) ---
            elif model_type == "model_based":
                # Batch is tuple from pinn_collate_fn
                # (context, target, total, *, *, *)
                # We expect context to be the 10 frames, target to be the 15 frames.
                # Validated by user config.
                
                if len(batch) >= 2:
                    context = batch[0].to(device)
                    target = batch[1].to(device)
                    # total = batch[2] # Not needed for pure prediction if model is CNP
                    
                    # LNP might need total for 'training' posterior, 
                    # but for validation/test we strictly use Prior (context-only).
                    # Models handles this: model(context, target) uses context to encode.
                    
                    output = model(context, target)
                    
                    # Prediction distribution at target locations
                    # output.smoke_dist.loc: (B, N_target, 1)
                    pred = output.smoke_dist.loc
                    
                    # Ground Truth
                    gt = target.values.unsqueeze(-1)
                    
                    # MSE
                    sq_err = (pred - gt) ** 2
                    total_mse += sq_err.sum().item()
                    total_points += sq_err.numel()

    if total_points > 0:
        return total_mse / total_points
    return 0.0


def evaluate_10_15_protocol(model, dataset, device, writer, epoch, model_type="model_free"):
    """
    Visualization-focused protocol for 10-context -> 15-forecast.
    Logs visuals for a single sequence to TensorBoard.
    """
    model.eval()
    
    try:
        if model_type == "model_free":
            # Retrieve one sequence from dataset
            # We assume dataset[0] is valid and long enough
            seq_data = dataset[0]
            if len(seq_data) < 25:
                return
                
            full_seq = seq_data[:25]
            context_seq = full_seq[:10]
            target_seq = full_seq[10:]
            
            # --- Inference for Viz ---
            bs = 1
            state = model.init_state(bs, device)
            
            # Context
            with torch.no_grad():
                for step in context_seq:
                    obs_raw = step['obs']
                    # Wrap single sample into batch of 1
                    obs = ObsSNP(
                        xs=obs_raw.xs.unsqueeze(0).to(device),
                        ys=obs_raw.ys.unsqueeze(0).to(device),
                        values=obs_raw.values.unsqueeze(0).to(device) if obs_raw.values is not None else None,
                        mask=obs_raw.mask.unsqueeze(0).to(device) if obs_raw.mask is not None else None
                    )
                    action = step['action'].unsqueeze(0).to(device)
                    done = step['done'].unsqueeze(0).to(device)
                    
                    out = model(state, action, done, obs=obs)
                    state = out.state
            
            # Forecast & Collect
            preds = []
            gts = []
            
            with torch.no_grad():
                for step in target_seq:
                    gt_raw = step['obs']
                    gt_obs = ObsSNP(
                        xs=gt_raw.xs.unsqueeze(0).to(device),
                        ys=gt_raw.ys.unsqueeze(0).to(device),
                        values=gt_raw.values.unsqueeze(0).to(device) if gt_raw.values is not None else None,
                        mask=gt_raw.mask.unsqueeze(0).to(device) if gt_raw.mask is not None else None
                    )
                    
                    action = step['action'].unsqueeze(0).to(device)
                    done = step['done'].unsqueeze(0).to(device)
                    
                    out = model(state, action, done, obs=None, query=gt_obs)
                    state = out.state
                    
                    pred = out.decoded if hasattr(out, 'decoded') else out.prediction
                    if hasattr(pred, 'loc'): pred = pred.loc
                    
                    preds.append(pred.cpu())
                    gts.append(gt_obs.values.cpu())

            # Visualize LAST frame (step 24)
            last_pred = preds[-1].squeeze() # (N,) or (N,1)
            last_gt = gts[-1].squeeze()
            last_xs = target_seq[-1]['obs'].xs.cpu()
            last_ys = target_seq[-1]['obs'].ys.cpu()
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # We assume points are scattered. If dense grid, scatter works too (albeit slow if huge).
            # If we knew resolution, we could imshow. For safety, scatter.
            sc1 = axes[0].scatter(last_xs, last_ys, c=last_gt, cmap='plasma', vmin=0, vmax=1, s=10)
            axes[0].set_title("GT (Frame 25)")
            plt.colorbar(sc1, ax=axes[0])
            
            sc2 = axes[1].scatter(last_xs, last_ys, c=last_pred, cmap='plasma', vmin=0, vmax=1, s=10)
            axes[1].set_title("Forecast (Frame 25)")
            plt.colorbar(sc2, ax=axes[1])
            
            log_figure_to_writer(fig, writer, "Protocol_10_15/Viz", epoch)

        elif model_type == "model_based":
            # Taking one random sample from validation is tricky if we don't have the loader here.
            # We'll try to use the dataset capability to sample a fresh window.
            if hasattr(dataset, 'smoke_data'):
                # Resample a window specifically for viz
                # We enforce index 0 to be stable across epochs if seed is set?
                # But PINN dataset is randomized in __getitem__.
                # Let's just call __getitem__(0) and hope for a valid window.
                # Actually, __getitem__ randomizes window.
                
                batch = dataset[0] # (ctx, trg, total, inflow, idx, t0)
                
                # Unwrap
                ctx_obs = batch[0].to(device)
                trg_obs = batch[1].to(device)
                t0_val = batch[5] if len(batch) > 5 else 0.0
                
                # Add Batch Dim
                ctx_batch = ObsPINN(
                     xs=ctx_obs.xs.unsqueeze(0),
                     ys=ctx_obs.ys.unsqueeze(0),
                     ts=ctx_obs.ts.unsqueeze(0),
                     values=ctx_obs.values.unsqueeze(0)
                )
                
                # For visualization, we want DENSE query at the last frame of target.
                # Target range is [10, 25). Last frame is 24.
                # Time of frame 24 = (24 * dt) - t0 (relative time).
                # t0 is the start time of the window (absolute).
                # Wait, dataset.__getitem__ returns relative times in .ts
                # ctx: 0..9, trg: 10..24
                # We need to construct a Dense Query at T=24 (relative).
                
                # Get dataset params
                H, W = dataset.H, dataset.W
                # We assume res=1 for simplicity or fetch from dataset
                res = getattr(dataset, 'res', 1.0)
                dt = getattr(dataset, 'dt', 0.1)
                
                # T=24 relative to window start
                # The window start was random, but 'ts' in observations are already relative to it.
                # ctx.ts goes from 0..9*dt
                # trg.ts goes from 10*dt..24*dt
                
                target_rel_t = (dataset.context_frames + dataset.target_frames - 1) * dt
                
                # Create grid
                ys, xs = torch.meshgrid(
                    torch.arange(H, device=device)*res, 
                    torch.arange(W, device=device)*res, 
                    indexing='ij'
                )
                ts = torch.full_like(xs, target_rel_t)
                
                query_viz = ObsPINN(
                    xs=xs.flatten().unsqueeze(0),
                    ys=ys.flatten().unsqueeze(0),
                    ts=ts.flatten().unsqueeze(0)
                )
                
                with torch.no_grad():
                    output = model(ctx_batch, query_viz)
                    
                pred_img = output.smoke_dist.loc.view(H, W).cpu().numpy()
                
                # We also need GT for layout comparison.
                # We can cheat and grab it from dataset.smoke_data if we knew the absolute T.
                # But dataset.__getitem__ returned a random window and only gave us sparse points.
                # We can't easily get the dense GT of that specific random window unless we knew the exact indices selected.
                # 't0' in batch is the offset.
                
                # We'll just plot Prediction for now, or Scatter GT if available.
                # Let's plot Scatter GT from trg_obs (sparse) vs Dense Prediction.
                
                # Filter trg_obs for the last frame only?
                # It's sparse in time. Might be empty for the last frame.
                
                fig, axes = plt.subplots(1, 1, figsize=(6, 5))
                im = axes.imshow(pred_img, origin='lower', cmap='plasma', vmin=0, vmax=1)
                axes.set_title("Forecast Frame 25 (Dense)")
                plt.colorbar(im, ax=axes)
                
                log_figure_to_writer(fig, writer, "Protocol_10_15/Viz", epoch)

    except Exception as e:
        print(f"Viz Error: {e}")
        plt.close('all')

def log_figure_to_writer(fig, writer, tag, epoch):
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    writer.add_image(tag, transforms.ToTensor()(img), epoch)
    plt.close(fig)
