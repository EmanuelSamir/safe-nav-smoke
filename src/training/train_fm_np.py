import sys
import os
sys.path.append(os.getcwd())
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import logging
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.models.model_based.fm_np import FlowNP
from src.models.shared.datasets import NonSequentialDataset, nonsequential_collate_fn
from src.models.shared.observations import Obs
from src.utils.visualize import log_physics_params # Might not be needed if no physics params, but kept for consistency

log = logging.getLogger(__name__)

def visualize_results(model, loader, device, writer, epoch, num_samples=4):
    """
    Visualize predictions vs ground truth via ODE Sampling.
    """
    model.eval()
    dataset = loader.dataset
    H, W = dataset.H, dataset.W
    res = dataset.res
    
    # Get a batch
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
        
    ctx, trg, t0, indices = batch
    
    # Process up to num_samples
    B = ctx.xs.shape[0]
    n_vis = min(B, num_samples)
    
    # Dense Query Grid
    # Use exact same coordinate definition as dataset to alignment Pred with GT
    x_idx = np.arange(W)
    y_idx = np.arange(H)
    x_range = 2.0 * (x_idx * res) / dataset.x_size - 1.0
    y_range = 2.0 * (y_idx * res) / dataset.y_size - 1.0
    
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    flat_x = torch.from_numpy(grid_x.flatten()).float().to(device).unsqueeze(0).unsqueeze(-1) # (1, P, 1)
    flat_y = torch.from_numpy(grid_y.flatten()).float().to(device).unsqueeze(0).unsqueeze(-1)
    
    # Visualization: 3 columns per time step, 2 time steps -> 6 columns
    fig, axes = plt.subplots(n_vis, 6, figsize=(30, 5 * n_vis))
    if n_vis == 1: axes = axes.reshape(1, -1)

    for ax in axes.flatten():
        ax.set_aspect(H/W)
    
    for i in range(n_vis):
        episode_idx = indices[i].item()
        # Full Dense Ground Truth for this episode
        # dataset.smoke_data[episode_idx] -> (T, H, W)
        
        unique_ts = torch.unique(trg.ts[i])
        if len(unique_ts) < 2: 
            times_to_vis = unique_ts
        else:
            # Pick 2 distinct times: e.g. midpoint and endpoint of the target sequence
            t_mid = unique_ts[len(unique_ts)//2]
            t_end = unique_ts[-1]
            times_to_vis = [t_mid, t_end]

        for j, t_val_tensor in enumerate(times_to_vis):
            col_offset = j * 3
            t_val = t_val_tensor.item()
            
            # --- 1. PREDICT DENSE ---
            flat_t = torch.full_like(flat_x, t_val)
            query_batch = Obs(xs=flat_x, ys=flat_y, ts=flat_t, values=None)
            
            ctx_sample = Obs(
                xs=ctx.xs[i:i+1].to(device),
                ys=ctx.ys[i:i+1].to(device),
                ts=ctx.ts[i:i+1].to(device),
                values=ctx.values[i:i+1].to(device),
                mask=ctx.mask[i:i+1].to(device) if ctx.mask is not None else None
            )
            
            with torch.no_grad():
                sampled_y = model.sample(ctx_sample, query_batch, steps=20)
            
            pred_grid = sampled_y[0].reshape(H, W).cpu().numpy()

            # --- 2. RETRIEVE DENSE GT ---
            # t_val is normalized time (-1 to 1) relative to t0
            # We need to map back to integer index in the original episode
            # ts = 2.0 * (t_real - t_offset) / max_diff_time - 1.0
            # => t_real - t_offset = (ts + 1) * max_diff_time / 2
            # => t_real = t_offset + ...
            # index = t_real / dt
            
            t_n = t_val
            max_diff = dataset.max_diff_time
            t_offset_val = t0[i].item()
            t_real = t_offset_val + (t_n + 1.0) * max_diff / 2.0
            t_idx = int(round(t_real / dataset.dt))
            
            # Clamp t_idx
            t_idx = min(max(t_idx, 0), dataset.n_steps - 1)
            
            gt_grid = dataset.smoke_data[episode_idx, t_idx]

            # --- PLOTTING ---
            
            # Plot 1: Context (Color=Time)
            ctx_xs = ctx.xs[i].cpu().numpy()
            ctx_ys = ctx.ys[i].cpu().numpy()
            ctx_ts = ctx.ts[i].cpu().numpy()
            
            sc0 = axes[i, col_offset + 0].scatter(ctx_xs, ctx_ys, c=ctx_ts, cmap='viridis', s=5)
            axes[i, col_offset + 0].set_title(f"Ctx (t_vis={t_val:.2f})")
            plt.colorbar(sc0, ax=axes[i, col_offset + 0])

            # Plot 2: Dense GT
            im1 = axes[i, col_offset + 1].imshow(gt_grid, origin='lower', extent=[-1, 1, -1, 1], cmap='viridis', vmin=0, vmax=1)
            axes[i, col_offset + 1].set_title(f"GT Dense (t={t_val:.2f})")
            plt.colorbar(im1, ax=axes[i, col_offset + 1])

            # Plot 3: Dense Pred
            im2 = axes[i, col_offset + 2].imshow(pred_grid, origin='lower', extent=[-1, 1, -1, 1], cmap='viridis', vmin=0, vmax=1)
            axes[i, col_offset + 2].set_title(f"Pred Sampled (t={t_val:.2f})")
            plt.colorbar(im2, ax=axes[i, col_offset + 2])
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    writer.add_image("Val/Visualization", transforms.ToTensor()(img), epoch)
    plt.close()


@hydra.main(version_base=None, config_path="../../config", config_name="training/fm_np_train")
def main(cfg: DictConfig):
    # 1. Setup
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Create subdirectories
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    
    # 2. Data
    data_path = cfg.training.data.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist")
        
    # Load data once to save memory
    log.info(f"Loading data from {data_path}...")
    loader = np.load(data_path)
    smoke_data = loader['smoke_data']
    
    # New Dataset Configuration
    train_ds = NonSequentialDataset(
        data_path=str(data_path),
        sequence_length=25,
        context_length=10, # First 10 frames for context
        mode='train',
        train_split=0.8,
        ctx_points_ratios=(0.2,0.3), #(0.005, 0.01),
        trg_points_ratio=0.5,
        max_episodes=cfg.training.data.get("max_samples", None),
        downsample_factor=cfg.training.data.get("downsample_factor", 5)
    )
    
    val_ds = NonSequentialDataset(
        data_path=str(data_path),
        sequence_length=25,
        context_length=10,
        mode='val',
        train_split=0.8,
        ctx_points_ratios=(0.2,0.3),
        trg_points_ratio=0.5,
        max_episodes=cfg.training.data.get("max_samples", None),
        downsample_factor=cfg.training.data.get("downsample_factor", 5)
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.data.batch_size, shuffle=True, collate_fn=nonsequential_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.data.batch_size, shuffle=False, collate_fn=nonsequential_collate_fn) 
    
    # 3. Model
    model_cfg = cfg.training.model
    # Adapt args:
    # dim_x=3 (x,y,t_phys) from config or hardcoded for smoke data
    dim_x = model_cfg.get("dim_x", 3)
    dim_y = model_cfg.get("dim_y", 1)
    
    model = FlowNP(
        dim_x=dim_x,
        dim_y=dim_y,
        fourier_mapping_size=model_cfg.get("fourier_mapping_size", 32),
        fourier_scale=model_cfg.get("fourier_scale", 10.0), # Added parameter
        d_model=model_cfg.d_model,
        emb_depth=model_cfg.emb_depth,
        dim_feedforward=model_cfg.dim_feedforward,
        nhead=model_cfg.nhead,
        dropout=model_cfg.dropout,
        num_layers=model_cfg.num_layers,
        timesteps=model_cfg.get("sampling_steps", 20)
    ).to(device)
    
    log.info(f"Model: FlowNP with d_model={model_cfg.d_model}, layers={model_cfg.num_layers}")
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=40)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(cfg.training.optimizer.max_epochs):
        model.train()
        train_loss_sum = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            context, target, t0, indices = batch
            context = context.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # --- Flow Matching Training Step ---
            # Forward pass: Returns Flow Matching Loss directly (MSE on velocity)
            loss = model(context, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.optimizer.grad_clip)
            optimizer.step()
            
            train_loss_sum += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
        # Logging
        avg_train_loss = train_loss_sum / len(train_loader)
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        
        # --- VAL ---
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                context, target, t0, indices = batch
                context = context.to(device)
                target = target.to(device)
                
                # Flow Matching Loss on Validation
                loss = model(context, target)
                
                val_loss_sum += loss.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        log.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_name = f"best_fm_np.pt"
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, save_name))
            
        if epoch % 1 == 0:
            visualize_results(model, val_loader, device, writer, epoch)

if __name__ == "__main__":
    main()
