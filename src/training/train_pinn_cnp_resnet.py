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
from tqdm import tqdm
from pathlib import Path
import logging

from src.models.model_based.pinn_cnp_resnet import PINN_CNP_ResNet
from src.models.model_based.dataset import GlobalSmokeDataset, pinn_collate_fn
from src.models.model_based.losses import BlindDiscoveryLoss
from src.utils.eval_protocol import evaluate_10_15_protocol, evaluate_forecast_protocol
from src.utils.visualize import log_pinn_fields, log_physics_params

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_cnp_resnet_train")
def main(cfg: DictConfig):
    # 1. Setup
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Create subdirectories
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    
    # 2. Data
    data_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.data_path
    
    train_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        info_ratio_per_frame=0.1,
        mode='train',
        max_samples=cfg.training.data.get("max_samples", None)
    )
    
    val_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        info_ratio_per_frame=0.1,
        mode='val',
        max_samples=cfg.training.data.get("max_samples", None)
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.data.batch_size, shuffle=True, collate_fn=pinn_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.data.batch_size, shuffle=False, collate_fn=pinn_collate_fn)
    
    # 3. Model & Losses
    # Using ResNet version with 3 blocks and hidden_dim 256
    hyper_params = {
        "latent_dim": 128, 
        "hidden_dim": 128, 
        "num_blocks": 4, 
        "out_mode": "lite"
    }
    model = PINN_CNP_ResNet(**hyper_params).to(device)
    
    # Configurable Loss Type
    physics_loss_fn = BlindDiscoveryLoss().to(device)
    log.info("Using BlindDiscoveryLoss (Navier-Stokes)")
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
        for context, target, total, inflow_map, ep_idx, t0 in pbar:
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(context, target)
            
            # 1. Data Loss
            pred_s = output.smoke_dist.loc
            loss_mse = torch.mean((pred_s.squeeze(-1) - target.values)**2)
            
            # 2. Physics Loss
            pred_tensor = torch.cat([
                output.u.view(-1, 1), 
                output.v.view(-1, 1), 
                output.smoke_dist.loc.view(-1, 1), 
                output.fu.view(-1, 1), 
                output.fv.view(-1, 1)
            ], dim=-1)
            
            coords_tensor = output.coords.view(-1, 3)
            loss_pde, pde_stats = physics_loss_fn(pred_tensor, coords_tensor)
            
            # Loss Total
            mse_w = cfg.training.loss.mse_weight
            pde_w = cfg.training.loss.pde_weight
            total_loss = mse_w * loss_mse + pde_w * loss_pde
            
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
        
        scheduler.step(avg_val_mse)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)
        
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            save_name = f"best_pinn_resnet_m{int(mse_w)}_p{int(pde_w)}.pt"
            # Updated Save Logic
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'hyper_parameters': hyper_params,
                'best_val_mse': best_val,
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, save_name))
            
        # Visualization
        if epoch % 5 == 0:
            log_pinn_fields(model, val_loader, device, writer, epoch, name="Val")
            evaluate_10_15_protocol(model, val_ds, device, writer, epoch, model_type="model_based")

            # Full Dataset Benchmark 10->15
            proto_mse = evaluate_forecast_protocol(model, val_loader, device, model_type="model_based")
            writer.add_scalar("Protocol_10_15/Dataset_MSE", proto_mse, epoch)
            log.info(f"Protocol 10-15 MSE: {proto_mse:.4f}")

if __name__ == "__main__":
    main()
