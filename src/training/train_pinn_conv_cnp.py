import sys
import os
import warnings

sys.path.append(os.getcwd())
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import logging

from src.models.model_based.pinn_conv_cnp import PINN_Conv_CNP
from src.models.model_based.utils import ObsPINN
from src.models.shared.datasets import GlobalSmokeDataset, pinn_collate_fn
# from src.models.model_based.losses import BlindDiscoveryLoss
from src.utils.eval_protocol import evaluate_10_15_protocol, evaluate_forecast_protocol
from src.utils.visualize import log_pinn_fields, log_physics_params

log = logging.getLogger(__name__)

def split_context_target(total: ObsPINN, min_ctx=10, max_ctx=None) -> Tuple[ObsPINN, ObsPINN]:
    """
    Splits a set of observations into context and target subsets randomly.
    For ConvCNP training, we usually want Context \subset Target.
    Here we take random subset as Context, and Full set as Target (reconstruction).
    """
    B, N = total.xs.shape
    device = total.xs.device
    
    if max_ctx is None: max_ctx = N // 2
    
    # We create a mask or index list. 
    # Handling batch logic: we can just pick a fixed number of context points for simplicity 
    # or randomized per batch. Let's do random number of points, same indices for batch (simplified).
    
    num_ctx = np.random.randint(min_ctx, max_ctx)
    indices = torch.randperm(N, device=device)
    ctx_idx = indices[:num_ctx]
    
    # Context Subset
    ctx = ObsPINN(
        xs=total.xs[:, ctx_idx],
        ys=total.ys[:, ctx_idx],
        ts=total.ts[:, ctx_idx],
        values=total.values[:, ctx_idx]
    )
    
    # Target is everything (reconstruction loss on all available points)
    return ctx, total

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_conv_cnp_train")
def main(cfg: DictConfig):
    # 1. Setup
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    
    # 2. Data
    data_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.data_path
    
    # Dataset loads 2.5s windows. 
    # For Training: we will ignore the internal 10/15 split and use the 'total' output.
    train_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        min_points_ratio=0.05,
        max_points_ratio=0.25,
        mode='train',
        max_samples=cfg.training.data.get("max_samples", None)
    )
    
    val_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        min_points_ratio=0.05,
        max_points_ratio=0.25,
        mode='val',
        max_samples=cfg.training.data.get("max_samples", None)
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.data.batch_size, shuffle=True, collate_fn=pinn_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.data.batch_size, shuffle=False, collate_fn=pinn_collate_fn)
    
    # 3. Model
    # Determine bounds from data if possible, or config. 
    # Config usually 50x50m.
    # ConvCNP needs to know grid range.
    spatial_max = cfg.training.model.spatial_max
    
    model = PINN_Conv_CNP(
        grid_res=cfg.training.model.grid_res,
        hidden_dim=cfg.training.model.hidden_dim,
        out_mode=cfg.training.model.out_mode,
        spatial_min=0.0,
        spatial_max=spatial_max,
        temporal_max=cfg.training.model.temporal_max
    ).to(device)
    
    # physics_loss_fn = BlindDiscoveryLoss().to(device)
    log.info("Using pure ConvCNP (No Physics Loss)")
    
    optimizer = optim.Adam(
        list(model.parameters()), 
        lr=cfg.training.optimizer.lr
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 4. Training Loop
    best_val = float('inf')
    
    for epoch in range(cfg.training.optimizer.max_epochs):
        model.train()
        train_mse, train_phys = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for _, _, total, _, _, _ in pbar: # Ignore split context/target, use total
            total = total.to(device)
            
            # Sub-sample context from total to effectively train on full time range
            context, query = split_context_target(total, min_ctx=10, max_ctx=total.xs.shape[1]//2)
            
            optimizer.zero_grad()
            
            output = model(context, query)
            
            # Metric Losses
            pred_s = output.smoke_dist.loc
            
            if query.mask is not None:
                valid_mask = (~query.mask).float().unsqueeze(-1)
                loss_mse = ((pred_s.squeeze(-1) - query.values)**2 * valid_mask.squeeze(-1)).sum() / (valid_mask.sum() + 1e-5)
            else:
                loss_mse = torch.mean((pred_s.squeeze(-1) - query.values)**2)
            
            # Physics Loss Removed
            
            mse_w = cfg.training.loss.mse_weight
            total_loss = mse_w * loss_mse
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_mse += loss_mse.item()
            # train_phys += loss_pde.item()
            pbar.set_postfix({'MSE': loss_mse.item()})
            
        # Logging
        avg_mse = train_mse / len(train_loader)
        writer.add_scalar("Train/MSE", avg_mse, epoch)
        writer.add_scalar("Train/PDE", train_phys / len(train_loader), epoch)
        
        # --- VAL (Forecasting Split) ---
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for context, target, total, _, _, _ in val_loader:
                # Use standard split: Context (past) -> Target (future)
                output = model(context.to(device), target.to(device))
                mse = torch.mean((output.smoke_dist.loc.squeeze(-1) - target.values.to(device))**2)
                val_mse += mse.item()
        
        avg_val_mse = val_mse / len(val_loader)
        writer.add_scalar("Val/MSE", avg_val_mse, epoch)
        log.info(f"Epoch {epoch}: Train MSE={avg_mse:.6f}, Val MSE={avg_val_mse:.6f}")
        
        scheduler.step(avg_val_mse)
        
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            save_name = f"best_convpinn.pt"
            torch.save(model.state_dict(), os.path.join(ckpt_dir, save_name))
            
        if epoch % 5 == 0:
            log_pinn_fields(model, val_loader, device, writer, epoch, name="Val")
            evaluate_10_15_protocol(model, val_ds, device, writer, epoch, model_type="model_based")

if __name__ == "__main__":
    main()
