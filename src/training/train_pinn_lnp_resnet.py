import sys
import os
sys.path.append(os.getcwd())

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, kl_divergence
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging

from src.models.model_based.pinn_lnp_resnet import PINN_LNP_ResNet
from src.models.model_based.dataset import GlobalSmokeDataset, pinn_collate_fn
from src.models.model_based.losses import BlindDiscoveryLoss
from src.utils.eval_protocol import evaluate_10_15_protocol, evaluate_forecast_protocol
from src.utils.visualize import log_pinn_fields, log_physics_params

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_lnp_resnet_train")
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
        mode='train'
    )
    
    val_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        mode='val'
    )
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=pinn_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=pinn_collate_fn)
    
    # 3. Model & Losses
    model = PINN_LNP_ResNet(
        latent_dim=128, 
        hidden_dim=256, 
        num_blocks=4, # Estabilidad ResNet
        out_mode="full"
    ).to(device)
    
    physics_loss_fn = BlindDiscoveryLoss().to(device)
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(physics_loss_fn.parameters()), 
        lr=cfg.training.optimizer.lr
    )
    
    # 4. Training Loop
    best_val = float('inf')
    
    for epoch in range(cfg.training.optimizer.max_epochs):
        model.train()
        train_stats = {"total": 0, "mse": 0, "kl": 0, "pde": 0}
        
        # Sigmoid scheduling centered at 100
        beta = 1.0 / (1.0 + np.exp(-0.1 * (epoch - 100)))
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for context, target, total, _, _, _ in pbar:
            context, target, total = context.to(device), target.to(device), total.to(device)
            optimizer.zero_grad()
            
            # ELBO: Prior vs Posterior
            mu_post, sigma_post = model.encode(total)
            z_sampled = model.reparameterize(mu_post, sigma_post)
            
            mu_prior, sigma_prior = model.encode(context)
            
            # Predict and compute physics losses using the sampled z
            output = model(context, target, z=z_sampled)
            
            # Loss parts
            pred_s = output.smoke_dist.loc
            loss_mse = torch.mean((pred_s.squeeze(-1) - target.values)**2)
            
            pred_tensor = torch.cat([
                output.u.view(-1, 1), 
                output.v.view(-1, 1), 
                output.p.view(-1, 1), 
                output.smoke_dist.loc.view(-1, 1), 
                output.q.view(-1, 1)
            ], dim=-1)
            loss_pde, pde_stats = physics_loss_fn(pred_tensor, output.coords.view(-1, 3))
            
            q_dist = Normal(mu_post, sigma_post)
            p_dist = Normal(mu_prior, sigma_prior)
            loss_kl = kl_divergence(q_dist, p_dist).sum(dim=-1).mean()
            
            total_loss = cfg.training.loss.mse_weight * loss_mse + cfg.training.loss.pde_weight * loss_pde + beta * loss_kl
            
            total_loss.backward()
            optimizer.step()
            
            train_stats["total"] += total_loss.item()
            train_stats["mse"] += loss_mse.item()
            train_stats["kl"] += loss_kl.item()
            train_stats["pde"] += loss_pde.item()
            pbar.set_postfix({'MSE': loss_mse.item(), 'KL': loss_kl.item(), 'B': beta})
            
        # Logging
        for k in train_stats: train_stats[k] /= len(train_loader)
        writer.add_scalar("Train/MSE", train_stats["mse"], epoch)
        writer.add_scalar("Train/KL", train_stats["kl"], epoch)
        writer.add_scalar("Train/PDE", train_stats["pde"], epoch)
        writer.add_scalar("Train/Beta", beta, epoch)
        
        # --- VAL ---
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for context, target, total, _, _, _ in val_loader:
                output = model(context.to(device), target.to(device))
                mse = torch.mean((output.smoke_dist.loc.squeeze(-1) - target.values.to(device))**2)
                val_mse += mse.item()
        
        avg_val_mse = val_mse / len(val_loader)
        writer.add_scalar("Val/MSE", avg_val_mse, epoch)
        log.info(f"Epoch {epoch}: T-MSE={train_stats['mse']:.6f}, V-MSE={avg_val_mse:.6f}, Beta={beta:.3f}")
        
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_pinn_lnp_resnet.pt"))
            
        if epoch % 5 == 0:
            log_pinn_fields(model, val_loader, device, writer, epoch, name="Val")
            evaluate_10_15_protocol(model, val_ds, device, writer, epoch, model_type="model_based")

            # Full Dataset Benchmark 10->15
            proto_mse = evaluate_forecast_protocol(model, val_loader, device, model_type="model_based")
            writer.add_scalar("Protocol_10_15/Dataset_MSE", proto_mse, epoch)
            log.info(f"Protocol 10-15 MSE: {proto_mse:.4f}")

if __name__ == "__main__":
    main()
