
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

def visualize_reconstruction(context, target, pred_samples, b_idx, save_dir, epoch, case_id, x_lim=50, y_lim=50):
    """
    Visualizes Flow Matching Reconstruction.
    """
    if context.mask is not None:
        c_mask = ~context.mask[b_idx]
    else:
        c_mask = slice(None)
        
    cx = context.xs[b_idx][c_mask].detach().cpu().numpy()
    cy = context.ys[b_idx][c_mask].detach().cpu().numpy()
    cv = context.values[b_idx][c_mask].detach().cpu().numpy()
    
    tx = target.xs[b_idx].detach().cpu().numpy()
    ty = target.ys[b_idx].detach().cpu().numpy()
    tv = target.values[b_idx].detach().cpu().numpy()
    pv = pred_samples[b_idx].detach().cpu().numpy().flatten()
    
    x_max = max(tx.max(), cx.max(), x_lim)
    y_max = max(ty.max(), cy.max(), y_lim)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    sc1 = axes[0].scatter(cx, cy, c=cv, cmap='plasma', vmin=0, vmax=1)
    axes[0].set_title(f"Context N={len(cx)}")
    axes[0].set_xlim(0, x_max)
    axes[0].set_ylim(0, y_max)
    plt.colorbar(sc1, ax=axes[0])
    
    sc2 = axes[1].scatter(tx, ty, c=tv, cmap='plasma', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].set_xlim(0, x_max)
    axes[1].set_ylim(0, y_max)
    plt.colorbar(sc2, ax=axes[1])
    
    sc3 = axes[2].scatter(tx, ty, c=pv, cmap='plasma', vmin=0, vmax=1)
    axes[2].set_title(f"Flow Sample (Ep {epoch})")
    axes[2].set_xlim(0, x_max)
    axes[2].set_ylim(0, y_max)
    plt.colorbar(sc3, ax=axes[2])
    
    # Error Map
    err = np.abs(tv - pv)
    sc4 = axes[3].scatter(tx, ty, c=err, cmap='viridis', vmin=0, vmax=0.5)
    axes[3].set_title("Abs Error")
    axes[3].set_xlim(0, x_max)
    axes[3].set_ylim(0, y_max)
    plt.colorbar(sc4, ax=axes[3])
    
    save_path = os.path.join(save_dir, f"flownp_case_{case_id}_ep{epoch}.png")
    plt.savefig(save_path)
    plt.close()

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_conv_cnp_train")
def main(cfg: DictConfig):
    try:
        torch.manual_seed(cfg.training.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # device = torch.device("cpu") # Force CPU to debug "Salio error"
        log.info(f"Using device: {device}")
        
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        log.info("Debugging FlowNP (Transformer) on Static Data")
        
        data_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.data_path
        
        ds = GlobalSmokeDataset(
            data_path=str(data_path),
            context_frames=1,
            target_frames=0, 
            min_points_ratio=0.05,
            max_points_ratio=0.5,
            mode='val', 
            max_samples=32
        )
        
        x_size = getattr(ds, 'x_size', 50.0)
        y_size = getattr(ds, 'y_size', 50.0)
        
        batch_size = 1 # Reduced from 4
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=pinn_collate_fn)
        
        # Init FlowNP
        # dim_x=3 (x,y,t)
        # Reduced size for speed/memory on Mac
        model = FlowNP(
            dim_x=2,
            dim_y=1,
            d_model=64,   # Reduced from 128
            num_layers=2, # Reduced from 4
            nhead=2,      # Reduced from 4
            dim_posenc=10,
            timesteps=20
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for Transformer sometimes better
        
        epochs = 200 # More epochs for Transformer convergence
        epoch_pbar = tqdm(range(epochs), desc="Training")
        
        history_loss = []

        for epoch in epoch_pbar:
            model.train()
            batch_losses = []
            
            for i, batch in enumerate(loader):
                full_obs = batch[0]
                B, N = full_obs.xs.shape
                
                # Sorted Time Split (Forecasting)
                # Since data is sequential frames, splitting by index splits by time implicitly.
                # Ratio 0.2 ~ 1 frame. 0.8 ~ 4 frames.
                ratio = np.random.uniform(0.2, 0.4) 
                num_ctx = max(10, int(N * ratio))
                
                c_xs = full_obs.xs[:, :num_ctx].to(device)
                c_ys = full_obs.ys[:, :num_ctx].to(device)
                c_vs = full_obs.values[:, :num_ctx].to(device)
                c_mask = full_obs.mask[:, :num_ctx].to(device)
                
                ctx = ObsPINN(xs=c_xs, ys=c_ys, values=c_vs, mask=c_mask)
                
                trg = full_obs # Reconstruct full
                trg.xs = trg.xs.to(device)
                trg.ys = trg.ys.to(device)
                trg.values = trg.values.to(device)
                trg.mask = trg.mask.to(device)
                
                optimizer.zero_grad()
                loss = model(ctx, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            avg_loss = np.mean(batch_losses)
            history_loss.append(avg_loss)
            
            if epoch % 10 == 0:
                 tqdm.write(f"Ep {epoch}: Loss {avg_loss:.5f}")
                 
            epoch_pbar.set_postfix({'Loss': f"{avg_loss:.4f}"})

        # --- Forecasting Visualization ---
        log.info("Generating Forecasting Visualizations (IC -> Future Sequence)...")
        model.eval()
        cases_plotted = 0
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if cases_plotted >= 3: break
                
                full_obs = batch[0]
                B, N = full_obs.xs.shape
                
                # For each element in batch
                for b_idx in range(B):
                    if cases_plotted >= 3: break
                    
                    # Extract valid data for this sample
                    if full_obs.mask is not None:
                         valid_mask = ~full_obs.mask[b_idx]
                    else:
                         valid_mask = slice(None)
                         
                    b_xs = full_obs.xs[b_idx][valid_mask]
                    b_ys = full_obs.ys[b_idx][valid_mask]
                    b_ts = full_obs.ts[b_idx][valid_mask]
                    b_vs = full_obs.values[b_idx][valid_mask]
                    
                    # Identify Unique Times
                    unique_times = torch.unique(b_ts).sort()[0]
                    if len(unique_times) < 2:
                        log.warning(f"Skipping case {cases_plotted}: Not enough time steps ({len(unique_times)})")
                        continue
                    
                    # Define Context (IC): First Time Step
                    t0 = unique_times[0]
                    is_ctx = (b_ts == t0)
                    
                    # Context Data
                    ctx_xs = b_xs[is_ctx].unsqueeze(0).to(device) # (1, N_c)
                    ctx_ys = b_ys[is_ctx].unsqueeze(0).to(device)
                    ctx_ts = b_ts[is_ctx].unsqueeze(0).to(device)
                    ctx_vs = b_vs[is_ctx].unsqueeze(0).to(device)
                    ctx_mask = torch.zeros((1, ctx_xs.shape[1]), dtype=torch.bool).to(device)
                    
                    ctx = ObsPINN(xs=ctx_xs, ys=ctx_ys, ts=ctx_ts, values=ctx_vs, mask=ctx_mask)
                    
                    # Plot Setup
                    num_frames = min(len(unique_times), 5) # Max 5 frames
                    fig, axes = plt.subplots(2, num_frames, figsize=(4*num_frames, 8))
                    
                    # If only 1 frame column, wrap in list
                    if num_frames == 1: axes = np.array([axes]).T
                    
                    for t_idx in range(num_frames):
                        t_val = unique_times[t_idx]
                        is_frame = (b_ts == t_val)
                        
                        # Target Points for this frame
                        trg_xs = b_xs[is_frame].unsqueeze(0).to(device)
                        trg_ys = b_ys[is_frame].unsqueeze(0).to(device)
                        trg_ts = b_ts[is_frame].unsqueeze(0).to(device)
                        trg_vs = b_vs[is_frame].unsqueeze(0).to(device)
                        trg_mask = torch.zeros((1, trg_xs.shape[1]), dtype=torch.bool).to(device)
                        
                        trg = ObsPINN(xs=trg_xs, ys=trg_ys, ts=trg_ts, values=trg_vs, mask=trg_mask)
                        
                        # Predict
                        # We predict ONE frame at a time query
                        pred_vals = model.sample(ctx, trg, steps=20) # (1, N_t, 1)
                        pred_flat = pred_vals[0].cpu().numpy().flatten()
                        
                        # Ground Truth
                        gt_flat = trg_vs[0].cpu().numpy()
                        frame_x = trg_xs[0].cpu().numpy()
                        frame_y = trg_ys[0].cpu().numpy()
                        
                        # Plot GT (Top Row)
                        sc_gt = axes[0, t_idx].scatter(frame_x, frame_y, c=gt_flat, cmap='plasma', vmin=0, vmax=1)
                        axes[0, t_idx].set_title(f"GT t={t_val:.2f}")
                        axes[0, t_idx].set_xlim(0, x_size)
                        axes[0, t_idx].set_ylim(0, y_size)
                        
                        # Plot Pred (Bottom Row)
                        sc_pred = axes[1, t_idx].scatter(frame_x, frame_y, c=pred_flat, cmap='plasma', vmin=0, vmax=1)
                        title = "IC (Context)" if t_idx == 0 else f"Pred t={t_val:.2f}"
                        axes[1, t_idx].set_title(title)
                        axes[1, t_idx].set_xlim(0, x_size)
                        axes[1, t_idx].set_ylim(0, y_size)
                    
                    plt.tight_layout()
                    save_path = os.path.join(output_dir, f"forecast_case_{cases_plotted}_ep{epochs}.png")
                    plt.savefig(save_path)
                    plt.close()
                    log.info(f"Saved forecasting sequence to {save_path}")
                    cases_plotted += 1

        log.info(f"Completed. Output: {output_dir}")
    
    except Exception as e:
        log.error(f"Error during execution: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
