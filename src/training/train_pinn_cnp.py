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

from src.models.model_based.pinn_cnp import PINN_CNP
from src.models.shared.datasets import NonSequentialDataset, nonsequential_collate_fn
from src.models.shared.observations import Obs
from src.models.model_based.losses import BlindDiscoveryLoss
from src.utils.visualize import log_physics_params

log = logging.getLogger(__name__)

def visualize_results(model, loader, device, writer, epoch, num_samples=4):
    """
    Visualize predictions vs ground truth.
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
        
    ctx, trg, t0, _ = batch 
    
    # Process up to num_samples
    B = ctx.xs.shape[0]
    n_vis = min(B, num_samples)
    print("n_vis: ", n_vis)
    
    # Dense Query Grid
    x_range = np.linspace(-1, 1, W) # Normalized version
    y_range = np.linspace(-1, 1, H)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    flat_x = torch.from_numpy(grid_x.flatten()).float().to(device).unsqueeze(0).unsqueeze(-1) # (1, P, 1)
    flat_y = torch.from_numpy(grid_y.flatten()).float().to(device).unsqueeze(0).unsqueeze(-1)
    
    fig, axes = plt.subplots(n_vis, 4, figsize=(20, 5 * n_vis))
    if n_vis == 1: axes = axes.reshape(1, -1)

    for ax in axes.flatten():
        ax.set_aspect(H/W)
    
    for i in range(n_vis):
        
        unique_ts = torch.unique(trg.ts[i])
        if len(unique_ts) == 0: continue
        
        # Pick last time
        t_val = unique_ts[-1].item()
        
        # Query Model at t_val (Dense)
        flat_t = torch.full_like(flat_x, t_val)
        query = Obs(xs=flat_x, ys=flat_y, ts=flat_t, values=None) # (1, P) but we need (B, P) if batching, or just 1
        
        # Context for this sample
        # ctx is (B, N_ctx)
        ctx_sample = Obs(
            xs=ctx.xs[i:i+1].to(device),
            ys=ctx.ys[i:i+1].to(device),
            ts=ctx.ts[i:i+1].to(device),
            values=ctx.values[i:i+1].to(device),
            mask=ctx.mask[i:i+1].to(device) if ctx.mask is not None else None
        )
        
        with torch.no_grad():
            output = model(ctx_sample, query)
            
        pred_grid = output.smoke_dist.loc.reshape(H, W).cpu().numpy()
        
        # Ground Truth at this time
        # We need the full grid from the dataset really, or reconstruction from trg points
        # But trg points might be sparse.
        # Let's use the sparse points we have in trg matching t_val
        
        gt_xs = trg.xs[i].cpu().numpy()
        gt_ys = trg.ys[i].cpu().numpy()
        gt_vals = trg.values[i].cpu().numpy()
        
        # Plot Context (All times)
        ctx_xs = ctx.xs[i].cpu().numpy()
        ctx_ys = ctx.ys[i].cpu().numpy()
        ctx_ts = ctx.ts[i].cpu().numpy()
        
        sc0 = axes[i, 0].scatter(ctx_xs, ctx_ys, c=ctx_ts, cmap='viridis', s=5)
        axes[i, 0].set_title("Context (Color=Time)")
        plt.colorbar(sc0, ax=axes[i, 0])

        # Plot GT Sparse
        sc1 = axes[i, 1].scatter(gt_xs, gt_ys, c=gt_vals, cmap='viridis', vmin=0, vmax=1, s=10)
        axes[i, 1].set_title(f"GT Sparse (t={t_val:.2f})")
        plt.colorbar(sc1, ax=axes[i, 1])

        # Plot Pred Dense
        im2 = axes[i, 2].imshow(pred_grid, origin='lower', extent=[0, W*res, 0, H*res], cmap='viridis', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Pred Dense (t={t_val:.2f})")
        plt.colorbar(im2, ax=axes[i, 2])
        
        # Plot Error (if we have enough GT coverage to approximate, otherwise skip or do point error)
        # Let's map GT sparse to grid for error? Hard if sparse.
        # Just show Pred again or Velocity
        if output.u is not None:
            u_grid = output.u.reshape(H, W).cpu().numpy()
            v_grid = output.v.reshape(H, W).cpu().numpy()
            # Subsample for quiver
            skip = 4
            X, Y = np.meshgrid(x_range, y_range)
            axes[i, 3].quiver(X[::skip, ::skip], Y[::skip, ::skip], u_grid[::skip, ::skip], v_grid[::skip, ::skip])
            axes[i, 3].set_title("Velocity Field")
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    writer.add_image("Val/Visualization", transforms.ToTensor()(img), epoch)
    plt.close()


@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_cnp_train")
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
    
    # New Dataset Configuration
    train_ds = NonSequentialDataset(
        data_path=str(data_path),
        sequence_length=25,
        context_length=10, # First 10 frames for context
        mode='train',
        train_split=0.8,
        ctx_points_ratios=(0.3,0.5), #(0.005, 0.01),
        trg_points_ratio=1.0,
        max_episodes=cfg.training.data.get("max_samples", None),
        downsample_factor=cfg.training.data.get("downsample_factor", 5)
    )
    
    val_ds = NonSequentialDataset(
        data_path=str(data_path),
        sequence_length=25,
        context_length=10,
        mode='val',
        train_split=0.8,
        ctx_points_ratios=(0.3,0.5),
        trg_points_ratio=1.0,
        max_episodes=cfg.training.data.get("max_samples", None),
        downsample_factor=cfg.training.data.get("downsample_factor", 5)
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.data.batch_size, shuffle=True, collate_fn=nonsequential_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.data.batch_size, shuffle=False, collate_fn=nonsequential_collate_fn) # Batch 1 for val usually safer for visualization
    
    # 3. Model & Losses
    hyper_params = {
        "latent_dim": cfg.training.model.latent_dim,
        "hidden_dim": cfg.training.model.hidden_dim,
        "fourier_mapping_size": cfg.training.model.fourier_mapping_size,
        "aggregator_type": cfg.training.model.aggregator
    }
    model = PINN_CNP(**hyper_params).to(device)
    
    # Configurable Loss Type
    physics_loss_fn = BlindDiscoveryLoss().to(device)
    log.info("Using BlindDiscoveryLoss (Navier-Stokes)")
    
    # Optimizer includes physics parameters (D, tau, nu)
    optimizer = optim.Adam(
        list(model.parameters()) + list(physics_loss_fn.parameters()), 
        lr=cfg.training.optimizer.lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # 4. Training Loop
    best_val = float('inf')
    
    for epoch in range(cfg.training.optimizer.max_epochs):
        model.train()
        train_mse, train_phys = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Unpack NonSequential Collate
            # batch is (ctx, trg, t0) where ctx and trg are Obs
            context, target, t0, _ = batch
            
            context = context.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Query points are the target points
            # We want to predict values at target locations
            output = model(context, target)
            
            # 1. Data Loss (MSE on smoke)
            loss_mse = -output.smoke_dist.log_prob(target.values).mean()

            # 2. Physics Loss
            # Flatten batch for PDE calculation
            # output.u etc are (B, N, 1) -> (B*N, 1)
            pred_tensor = torch.cat([
                output.u.view(-1, 1), 
                output.v.view(-1, 1), 
                output.smoke_dist.loc.view(-1, 1),
                output.fu.view(-1, 1), 
                output.fv.view(-1, 1)
            ], dim=-1)
            
            # Use the exact coords tensor that has requires_grad=True
            coords_tensor = output.coords.view(-1, 3)
            
            # loss_pde, pde_stats = physics_loss_fn(pred_tensor, coords_tensor)
            
            mse_w = cfg.training.loss.mse_weight
            # pde_w = cfg.training.loss.pde_weight
            total_loss = mse_w * loss_mse 
            
            total_loss.backward()
            optimizer.step()
            
            train_mse += loss_mse.item()
            # train_phys += loss_pde.item()
            pbar.set_postfix({'MSE': loss_mse.item()})
            
        # Logging
        avg_mse = train_mse / len(train_loader)
        # avg_phys = train_phys / len(train_loader)
        writer.add_scalar("Train/MSE", avg_mse, epoch)
        # writer.add_scalar("Train/PDE", avg_phys, epoch)
        
        # for stat_name, stat_val in pde_stats.items():
        #     writer.add_scalar(f"PDE_Stats/{stat_name}", stat_val, epoch)
        log_physics_params(physics_loss_fn, writer, epoch)
        
        # --- VAL ---
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                context, target, t0, _ = batch
                context = context.to(device)
                target = target.to(device)
                
                output = model(context, target)
                mse = -output.smoke_dist.log_prob(target.values).mean()
                val_mse += mse.item()
        
        avg_val_mse = val_mse / len(val_loader)
        writer.add_scalar("Val/MSE", avg_val_mse, epoch)
        log.info(f"Epoch {epoch}: Train MSE={avg_mse:.6f}, Val MSE={avg_val_mse:.6f}")
        
        scheduler.step(avg_val_mse)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)
        
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            save_name = f"best_pinn_m{int(mse_w)}.pt"#_p{int(pde_w)}.pt"
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'hyper_parameters': hyper_params,
                'best_val_mse': best_val,
                'optimizer_state_dict': optimizer.state_dict(),
                # 'physics_params': physics_loss_fn.state_dict()
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, save_name))
            
        if epoch % 1 == 0:
            visualize_results(model, val_loader, device, writer, epoch)

if __name__ == "__main__":
    main()
