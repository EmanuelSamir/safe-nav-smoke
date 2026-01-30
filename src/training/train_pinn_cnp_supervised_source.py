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
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import logging

from src.models.model_based.pinn_cnp import PINN_CNP, ObsPINN
from src.models.model_based.dataset import GlobalSmokeDataset, pinn_collate_fn
from src.models.model_based.losses import BlindDiscoveryLoss
from src.utils.visualize import log_pinn_fields, log_physics_params
from src.utils.eval_protocol import evaluate_forecast_protocol

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_cnp_supervised_train")
def main(cfg: DictConfig):
    # 1. Setup
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=output_dir)
    
    # 2. Data
    data_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.data_path
    
    train_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        info_ratio_per_frame=0.4,
        mode='train'
    )
    
    val_ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=10,
        target_frames=15,
        info_ratio_per_frame=0.4,
        mode='val'
    )
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=pinn_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=pinn_collate_fn)
    
    # 3. Model & Losses
    # Use "full" mode for BlindDiscoveryLoss (Navier-Stokes)
    model = PINN_CNP(latent_dim=128, hidden_dim=256, out_mode="full").to(device)
    
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
        for context, target, total, inflow_map, _, _ in pbar:
            context = context.to(device)
            target = target.to(device)
            inflow_map = inflow_map.to(device)
            
            optimizer.zero_grad()
            
            # 1. Prediction (Forecast)
            output = model(context, target)
            
            # 2. MSE Loss (Reconstruction)
            pred_s = output.smoke_dist.loc.squeeze(-1)
            loss_mse = torch.mean((pred_s - target.values)**2)
            
            # 3. Physics Loss (Blind Discovery with Supervised Source)
            # Layout: [u, v, p, s, q]
            pred_tensor = torch.cat([
                output.u.view(-1, 1), 
                output.v.view(-1, 1), 
                output.p.view(-1, 1), 
                output.smoke_dist.loc.view(-1, 1), 
                output.q.view(-1, 1)
            ], dim=-1)
            
            coords_tensor = output.coords.view(-1, 3) # [x, y, t]
            
            # --- SUPERVISIÓN DE FUENTE (q) ---
            # Muestreamos el inflow_map (B, H, W) en las coordenadas coords_tensor (B*N, 3)
            # inflow_map es Frame 0 del episodio.
            B, H, W = inflow_map.shape
            N = target.xs.shape[1]
            
            # Grid sample espera (B, 1, N, 2) y coordenadas en [-1, 1]
            # Coords: [x, y, t]. Usamos x, y. 
            qy = (output.coords[..., 1:2] / train_ds.y_size) * 2 - 1 # Normalizado a [-1, 1]
            qx = (output.coords[..., 0:1] / train_ds.x_size) * 2 - 1
            grid_coords = torch.cat([qx, qy], dim=-1).unsqueeze(1) # (B, 1, N, 2)
            
            # Samplear el mapa de fuentes (B, 1, H, W)
            q_real = F.grid_sample(inflow_map.unsqueeze(1), grid_coords, align_corners=True) # (B, 1, 1, N)
            q_real = q_real.view(-1, 1) # (B*N, 1)
            
            loss_pde, pde_stats = physics_loss_fn(pred_tensor, coords_tensor, q_real=q_real)
            
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
        # Log all PDE internals
        for stat_name, stat_val in pde_stats.items():
            writer.add_scalar(f"PDE_Stats/{stat_name}", stat_val, epoch)
        log_physics_params(physics_loss_fn, writer, epoch)
        
        # --- VAL ---
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for context, target, total, inflow_map in val_loader:
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
            save_name = f"best_pinn_supervised_m{int(mse_w)}_p{int(pde_w)}.pt"
            torch.save(model.state_dict(), os.path.join(output_dir, save_name))
            
        if epoch % 5 == 0:
            log_pinn_fields(model, val_loader, device, writer, epoch, name="Val")
        
        # Protocol evaluation
        proto_mse = evaluate_forecast_protocol(model, val_loader, device, model_type="model_based")
        writer.add_scalar("Protocol_10_15/Dataset_MSE", proto_mse, epoch)
        log.info(f"Protocol 10-15 MSE: {proto_mse:.4f}")

def visualize_pinn(model, loader, device, writer, epoch):
    
    model.eval()
    context, target, total = next(iter(loader))
    
    with torch.no_grad():
        output = model(context.to(device), target.to(device))
        
    # Visualize first sample of batch
    # We'll plot the target points (which are scattered in space and time)
    # To make it readable, we just plot them as a cloud
    gt = target.values[0].cpu().numpy()
    pr = output.smoke_dist.loc[0].squeeze().cpu().numpy()
    xs = target.xs[0].cpu().numpy()
    ys = target.ys[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = axes[0].scatter(xs, ys, c=gt, cmap='plasma', vmin=0, vmax=1)
    axes[0].set_title("GT (Global Target Samples)")
    plt.colorbar(sc1, ax=axes[0])
    
    sc2 = axes[1].scatter(xs, ys, c=pr, cmap='plasma', vmin=0, vmax=1)
    axes[1].set_title("PINN Prediction")
    plt.colorbar(sc2, ax=axes[1])
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    writer.add_image("Val/GlobalForecasting", transforms.ToTensor()(Image.open(buf)), epoch)
    plt.close()

if __name__ == "__main__":
    main()
