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

from src.models.model_based.pinn_cnp import PINN_CNP
from src.models.model_based.utils import ObsPINN
from src.models.model_based.dataset import GlobalSmokeDataset, pinn_collate_fn
from src.models.model_based.losses import BlindDiscoveryLoss
from src.utils.eval_protocol import evaluate_10_15_protocol, evaluate_forecast_protocol
from src.utils.visualize import log_pinn_fields, log_physics_params

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_cnp_train")
def main(cfg: DictConfig):
    # 1. Setup
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    output_dir = cfg.hydra.runtime.output_dir
    
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
    
    train_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        info_ratio_per_frame=0.2,
        mode='train',
        max_samples=cfg.training.data.get("max_samples", None)
    )
    
    val_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        info_ratio_per_frame=0.2,
        mode='val',
        max_samples=cfg.training.data.get("max_samples", None)
    )
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=pinn_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=pinn_collate_fn)
    
    # 3. Model & Losses
    # Use "full" mode for BlindDiscoveryLoss (Navier-Stokes)
    hyper_params = {
        "latent_dim": 128, "hidden_dim": 128, "out_mode": "lite",
        "aggregator_type": cfg.training.model.get("aggregator", "attention")
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
    # Scheduler para bajar el LR cuando el error se estanca
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # 4. Training Loop
    best_val = float('inf')
    
    for epoch in range(cfg.training.optimizer.max_epochs):
        model.train()
        train_mse, train_phys = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for context, target, total, inflow_map, ep_idx, t0 in pbar:
            context = context.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(context, target)
            
            # 1. Data Loss (MSE on smoke)
            # ObsPINN target.values are the real smoke values
            pred_s = output.smoke_dist.loc
            loss_mse = torch.mean((pred_s.squeeze(-1) - target.values)**2)
            
            # 2. Physics Loss
            # We need to reshape for the loss function which expects (TotalPoints, Dim)
            # and coordinated for derivatives
            B, N, _ = output.u.shape
            
            # Flatten batch for PDE calculation
            pred_tensor = torch.cat([
                output.u.view(-1, 1), 
                output.v.view(-1, 1), 
                output.smoke_dist.loc.view(-1, 1),
                output.fu.view(-1, 1), 
                output.fv.view(-1, 1)
            ], dim=-1)
            
            # Use the exact coords tensor that has requires_grad=True
            coords_tensor = output.coords.view(-1, 3)
            
            loss_pde, pde_stats = physics_loss_fn(pred_tensor, coords_tensor)
            
            mse_w = cfg.training.loss.mse_weight
            pde_w = cfg.training.loss.pde_weight
            total_loss = mse_w * loss_mse + pde_w * loss_pde # Scaling
            
            total_loss.backward()
            optimizer.step()
            
            train_mse += loss_mse.item()
            train_phys += loss_pde.item()
            pbar.set_postfix({'MSE': loss_mse.item(), 'PDE': loss_pde.item()})
            
        # Logging
        avg_mse = train_mse / len(train_loader)
        avg_phys = train_phys / len(train_loader)
        writer.add_scalar("Train/MSE", avg_mse, epoch)
        writer.add_scalar("Train/PDE", avg_phys, epoch)
        # Log PDE Internals
        for stat_name, stat_val in pde_stats.items():
            writer.add_scalar(f"PDE_Stats/{stat_name}", stat_val, epoch)
        log_physics_params(physics_loss_fn, writer, epoch)
        
        # --- VAL ---
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for context, target, total, inflow_map, ep_idx, t0 in val_loader:
                output = model(context.to(device), target.to(device))
                mse = torch.mean((output.smoke_dist.loc.squeeze(-1) - target.values.to(device))**2)
                val_mse += mse.item()
        
        avg_val_mse = val_mse / len(val_loader)
        writer.add_scalar("Val/MSE", avg_val_mse, epoch)
        log.info(f"Epoch {epoch}: Train MSE={avg_mse:.6f}, Val MSE={avg_val_mse:.6f}")
        
        # Step del scheduler
        scheduler.step(avg_val_mse)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)
        
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            save_name = f"best_pinn_m{int(mse_w)}_p{int(pde_w)}.pt"
            # Updated Save Logic
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'hyper_parameters': hyper_params,
                'best_val_mse': best_val,
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, save_name))
            
        if epoch % 1 == 0:
            log_pinn_fields(model, val_loader, device, writer, epoch, name="Val")
            evaluate_10_15_protocol(model, val_ds, device, writer, epoch, model_type="model_based")

        # Full Dataset Benchmark 10->15
        proto_mse = evaluate_forecast_protocol(model, val_loader, device, model_type="model_based")
        writer.add_scalar("Protocol_10_15/Dataset_MSE", proto_mse, epoch)
        log.info(f"Protocol 10-15 MSE: {proto_mse:.4f}")

def visualize_pinn(model, loader, device, writer, epoch):
    
    model.eval()
    
    # Select random sample from entire dataset
    dataset = loader.dataset
    idx = np.random.randint(0, len(dataset))
    sample = dataset[idx]
    
    # Collate (expects list of samples)
    if 'pinn_collate_fn' in globals():
        collate_fn = pinn_collate_fn
    else:
        from src.models.model_based.dataset import pinn_collate_fn as collate_fn
        
    batch = collate_fn([sample])
    context, target, total = batch[:3] # Ignore extra returns if any
    
    # Get metadata for dense query
    H, W = dataset.H, dataset.W
    res = dataset.res
    
    batch_size = context.xs.shape[0] # Should be 1
    sample_idx = 0
    
    # Pick a random timestamp from the target sequence of this sample
    # target.ts is (B, N)
    unique_ts = torch.unique(target.ts[sample_idx])
    # Filter out padding or invalid if any? Usually just valid times.
    t_val = unique_ts[np.random.randint(0, len(unique_ts))].item()
    
    # GT Cloud at this specific time
    # We find points that are close to t_val (float comparison tolerance)
    time_mask = torch.abs(target.ts[sample_idx] - t_val) < 1e-4
    gt_xs = target.xs[sample_idx][time_mask].cpu().numpy()
    gt_ys = target.ys[sample_idx][time_mask].cpu().numpy()
    gt_vals = target.values[sample_idx][time_mask].cpu().numpy()
    
    # Dense Query at t_val
    x_range = np.arange(W) * res
    y_range = np.arange(H) * res
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    flat_x = torch.from_numpy(grid_x.flatten()).float().to(device).unsqueeze(0)
    flat_y = torch.from_numpy(grid_y.flatten()).float().to(device).unsqueeze(0)
    flat_t = torch.full_like(flat_x, t_val)
    
    query = ObsPINN(xs=flat_x, ys=flat_y, ts=flat_t)
    
    # Context for this sample
    ctx_sample = ObsPINN(
        xs=context.xs[sample_idx:sample_idx+1].to(device),
        ys=context.ys[sample_idx:sample_idx+1].to(device),
        ts=context.ts[sample_idx:sample_idx+1].to(device),
        values=context.values[sample_idx:sample_idx+1].to(device)
    )
    
    with torch.no_grad():
        output = model(ctx_sample, query)
        
    pred_grid = output.smoke_dist.loc.reshape(H, W).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot GT as scatter 
    # Use bounds [0, W*res]
    sc1 = axes[0].scatter(gt_xs, gt_ys, c=gt_vals, cmap='plasma', vmin=0, vmax=1)
    axes[0].set_xlim(0, W*res)
    axes[0].set_ylim(0, H*res)
    axes[0].set_title(f"GT Sparse (t={t_val:.2f})")
    plt.colorbar(sc1, ax=axes[0])
    
    # Plot Dense Pred as Image
    im2 = axes[1].imshow(pred_grid, origin='lower', extent=[0, W*res, 0, H*res], cmap='plasma', vmin=0, vmax=1)
    axes[1].set_title(f"PINN Dense Pred (t={t_val:.2f})")
    plt.colorbar(im2, ax=axes[1])
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    writer.add_image("Val/GlobalForecasting", transforms.ToTensor()(Image.open(buf)), epoch)
    plt.close()

if __name__ == "__main__":
    main()
