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
from src.utils.visualize import log_pinn_fields, log_physics_params

from src.utils.eval_protocol import evaluate_10_15_protocol, evaluate_forecast_protocol

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_lnp_resnet_multisample_train")
def main(cfg: DictConfig):
    # 1. Setup
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=output_dir)
    
    # Parámetros de Batching Latente
    K_SAMPLES = 4 # Número de muestras de z por cada actualización (Batch Latente)
    
    # 2. Data
    data_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.data_path
    train_ds = GlobalSmokeDataset(data_path=str(data_path), context_frames=10, target_frames=15, mode='train', max_samples=cfg.training.data.get("max_samples", None))
    val_ds = GlobalSmokeDataset(data_path=str(data_path), context_frames=10, target_frames=15, mode='val', max_samples=cfg.training.data.get("max_samples", None))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.data.batch_size, shuffle=True, collate_fn=pinn_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.data.batch_size, shuffle=False, collate_fn=pinn_collate_fn)
    
    # 3. Model & Losses
    model = PINN_LNP_ResNet(latent_dim=128, hidden_dim=256, num_blocks=4, out_mode="full").to(device)
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
        train_stats = {"total": 0, "mse": 0, "kl": 0, "pde": 0}
        beta = 1.0 / (1.0 + np.exp(-0.1 * (epoch - 100))) # Sigmoid Beta
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for context, target, total, inflow_map, ep_idx, t0 in pbar:
            context, target, total = context.to(device), target.to(device), total.to(device)
            optimizer.zero_grad()
            
            # --- STEP 1: Obtener distribuciones (Prior y Posterior) ---
            mu_post, sigma_post = model.encode(total)
            mu_prior, sigma_prior = model.encode(context)
            
            # --- STEP 2: MULTI-SAMPLE BATCHING ---
            batch_loss_mse = 0
            batch_loss_pde = 0
            
            # Evaluamos K suposiciones de z simultáneamente
            # Evaluamos K suposiciones de z simultáneamente
            for i in range(K_SAMPLES):
                z_sampled = model.reparameterize(mu_post, sigma_post)
                output = model(context, target, z=z_sampled)
                
                # 1. Reconstruction Loss (Likelihood)
                pred_s = output.smoke_dist.loc
                batch_loss_mse += torch.mean((pred_s.squeeze(-1) - target.values)**2)
                
                # 2. Physics Loss (PDE)
                pred_tensor = torch.cat([
                    output.u.view(-1, 1), 
                    output.v.view(-1, 1), 
                    output.smoke_dist.loc.view(-1, 1), 
                    output.fu.view(-1, 1), 
                    output.fv.view(-1, 1)
                ], dim=-1)
                l_pde, p_stats = physics_loss_fn(pred_tensor, output.coords.view(-1, 3))
                batch_loss_pde += l_pde
                
                # Acumular stats
                if i == 0:
                    accum_stats = {k: v for k, v in p_stats.items()}
                else:
                    for k, v in p_stats.items(): accum_stats[k] += v
            
            # Promediamos los resultados de las K muestras
            avg_loss_mse = batch_loss_mse / K_SAMPLES
            avg_loss_pde = batch_loss_pde / K_SAMPLES
            for k in accum_stats: accum_stats[k] /= K_SAMPLES
            pde_stats = accum_stats # Para el logging posterior
            
            # --- STEP 3: KL Divergence (Solo una vez, depende de los parámetros mu, sigma) ---
            q_dist = Normal(mu_post, sigma_post)
            p_dist = Normal(mu_prior, sigma_prior)
            loss_kl = kl_divergence(q_dist, p_dist).sum(dim=-1).mean()
            
            # Loss Total
            mse_w = cfg.training.loss.mse_weight
            pde_w = cfg.training.loss.pde_weight
            total_loss = mse_w * avg_loss_mse + pde_w * avg_loss_pde + beta * loss_kl
            
            total_loss.backward()
            optimizer.step()
            
            train_stats["total"] += total_loss.item()
            train_stats["mse"] += avg_loss_mse.item()
            train_stats["kl"] += loss_kl.item()
            train_stats["pde"] += avg_loss_pde.item()
            pbar.set_postfix({'MSE': avg_loss_mse.item(), 'KL': loss_kl.item(), 'K': K_SAMPLES})
            
        # Logging & Val (Similar al anterior)
        for k in train_stats: train_stats[k] /= len(train_loader)
        writer.add_scalar("Train/MSE", train_stats["mse"], epoch)
        writer.add_scalar("Train/KL", train_stats["kl"], epoch)
        writer.add_scalar("Train/Beta", beta, epoch)
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
        log.info(f"Epoch {epoch}: T-MSE={train_stats['mse']:.6f}, V-MSE={avg_val_mse:.6f}, KL={train_stats['kl']:.4f}")
        
        scheduler.step(avg_val_mse)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)
        
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            save_name = f"best_pinn_lnp_resnet_multi_m{int(mse_w)}_p{int(pde_w)}.pt"
            torch.save(model.state_dict(), os.path.join(output_dir, save_name))
            
        # Visualization
        if epoch % 5 == 0:
            log_pinn_fields(model, val_loader, device, writer, epoch, name="Val")
            evaluate_10_15_protocol(model, val_ds, device, writer, epoch, model_type="model_based")
            
        proto_mse = evaluate_forecast_protocol(model, val_loader, device, model_type="model_based")
        writer.add_scalar("Protocol_10_15/Dataset_MSE", proto_mse, epoch)
        log.info(f"Protocol 10-15 MSE: {proto_mse:.4f}")

if __name__ == "__main__":
    main()
