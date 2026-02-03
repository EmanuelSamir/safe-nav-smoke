import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm
import numpy as np

# Imports
from src.models.model_based.flow_matching_np import FlowNP
from src.models.shared.datasets import GlobalSmokeDataset, pinn_collate_fn
from src.models.model_based.utils import ObsPINN

log = logging.getLogger(__name__)

def visualize_static_reconstruction(model, context_obs, gt_grid, device, save_path, title_suffix="", x_size=50, y_size=50):
    """
    Visualizes reconstruction of a static frame on a dense grid.
    """
    model.eval()
    with torch.no_grad():
        # Create Dense Grid Query
        H, W = gt_grid.shape
        y_coords = torch.linspace(0, y_size, H)
        x_coords = torch.linspace(0, x_size, W)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        flat_x = grid_x.reshape(-1)
        flat_y = grid_y.reshape(-1)
        
        query_xs = flat_x.unsqueeze(0).to(device) # (1, H*W)
        query_ys = flat_y.unsqueeze(0).to(device)
        
        query_mask = torch.zeros_like(query_xs, dtype=torch.bool)
        
        query = ObsPINN(xs=query_xs, ys=query_ys, values=torch.zeros_like(query_xs), mask=query_mask)
        
        # Sample from model
        # steps=20 ensures good integration
        pred_vals = model.sample(context_obs, query, steps=20) # (1, N_q, 1)
        pred_grid = pred_vals[0].reshape(H, W).cpu().numpy()
        
        # Context Points
        cx = context_obs.xs[0].cpu().numpy()
        cy = context_obs.ys[0].cpu().numpy()
        cv = context_obs.values[0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Ground Truth
        im1 = axes[0].imshow(gt_grid, origin='lower', extent=[0, x_size, 0, y_size], cmap='plasma', vmin=0, vmax=1)
        axes[0].set_title(f"Ground Truth {title_suffix}")
        plt.colorbar(im1, ax=axes[0])
        
        # 2. Context Points
        axes[1].scatter(cx, cy, c=cv, cmap='plasma', vmin=0, vmax=1, s=10)
        axes[1].set_title(f"Context Points (N={len(cx)})")
        axes[1].set_xlim(0, x_size)
        axes[1].set_ylim(0, y_size)
        axes[1].set_aspect('equal')
        
        # 3. Model Reconstruction
        im3 = axes[2].imshow(pred_grid, origin='lower', extent=[0, x_size, 0, y_size], cmap='plasma', vmin=0, vmax=1)
        axes[2].set_title(f"Reconstruction")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_conv_cnp_train")
def main(cfg: DictConfig):
    try:
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        log.info(f"Using device: {device}")
        
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        log.info("Debugging FlowNP: Training on Batch of Static Frames")
        
        data_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.data_path
        
        # Load Data
        ds = GlobalSmokeDataset(
            data_path=str(data_path),
            context_frames=1,   # Single frame per sample
            target_frames=0, 
            min_points_ratio=0.05, 
            max_points_ratio=0.25, # Sparse enough to be interesting
            mode='train', 
            max_samples=32 
        )
        
        x_size = getattr(ds, 'x_size', 50.0)
        y_size = getattr(ds, 'y_size', 50.0)
        
        # Loader
        loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=pinn_collate_fn, drop_last=True)
        
        # Init Model (dim_x=2 for Static)
        model = FlowNP(
            dim_x=2,
            dim_y=1,
            d_model=128,  
            num_layers=4, 
            nhead=4,      
            dim_posenc=16,
            timesteps=20,
            bounds=[x_size, y_size]
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=5e-4) # Slightly higher LR for batch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        
        epochs = 200
        pbar = tqdm(range(epochs), desc="Training")
        
        for epoch in pbar:
            model.train()
            batch_losses = []
            
            for batch in loader:
                # Unpack Batch
                # batch: ctx, trg, total, inflow, idx, t0
                full_obs = batch[2] # Total points available
                
                full_obs = full_obs.to(device)
                
                # Create Dynamic Context (Inpainting Task)
                # Take first 20% of points as context
                B, N_max = full_obs.xs.shape
                num_ctx = int(N_max * 0.5)
                if num_ctx < 5: num_ctx = 5 # Safety
                
                # Context
                c_xs = full_obs.xs[:, :num_ctx]
                c_ys = full_obs.ys[:, :num_ctx]
                c_vs = full_obs.values[:, :num_ctx]
                c_mask = full_obs.mask[:, :num_ctx] # Preserves padding mask if present
                
                ctx = ObsPINN(xs=c_xs, ys=c_ys, values=c_vs, mask=c_mask)
                
                # Target: All points (Reconstruction)
                trg = full_obs
                
                optimizer.zero_grad()
                loss = model(ctx, trg)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())

            scheduler.step()
            avg_loss = np.mean(batch_losses)
            pbar.set_postfix({'Loss': f"{avg_loss:.6f}"})
        
        log.info("Training Complete. Generating Visualizations...")
        
        # Visualize 5 Random Cases
        for i in range(5):
            # Sample random item from dataset directly (getting fresh points)
            rand_idx = np.random.randint(0, len(ds))
            raw_item = ds[rand_idx]
            
            # Unpack (ObsPINN, ObsPINN, ObsPINN, Inflow, Idx, t_offset)
            _, _, total_obs, _, ep_idx, t_offset = raw_item
            
            # Prepare GT Grid for this specific sample time
            # t_offset is the exact time of the frame sampled
            # We need to find the frame index in the raw data
            # dt = ds.dt
            frame_idx = int(round(t_offset / ds.dt))
            gt_grid = ds.smoke_data[ep_idx, frame_idx]
            
            # Prepare Context (First 20%)
            # Need to put into batch format (1, N)
            N_pts = len(total_obs.xs)
            num_ctx = int(N_pts * 0.5)
            
            # Slicing 1D tensors
            c_xs = total_obs.xs[:num_ctx].unsqueeze(0).to(device)
            c_ys = total_obs.ys[:num_ctx].unsqueeze(0).to(device)
            c_vs = total_obs.values[:num_ctx].unsqueeze(0).to(device)
            c_mask = torch.zeros((1, num_ctx), dtype=torch.bool).to(device)
            
            ctx = ObsPINN(xs=c_xs, ys=c_ys, values=c_vs, mask=c_mask)
            
            save_path = os.path.join(output_dir, f"viz_sample_{i}_ep{ep_idx}_t{t_offset:.1f}.png")
            visualize_static_reconstruction(
                model, ctx, gt_grid, device, save_path, 
                title_suffix=f"(Ep {ep_idx} t={t_offset:.1f})",
                x_size=x_size, y_size=y_size
            )
            
        log.info(f"Done. Check {output_dir}")

    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
