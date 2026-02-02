
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

# Imports from our codebase
from src.models.model_based.pinn_conv_cnp import PINN_Conv_CNP
from src.models.shared.datasets import GlobalSmokeDataset, pinn_collate_fn
from src.models.model_based.utils import ObsPINN

log = logging.getLogger(__name__)

def visualize_reconstruction(context, target, pred_s, b_idx, save_dir, epoch, case_id, x_lim=50, y_lim=50):
    """
    Visualizes Context Points, Ground Truth Scalar Field, and Predicted Scalar Field.
    """
    # Valid context points
    if context.mask is not None:
        c_mask = ~context.mask[b_idx] # True = Valid
    else:
        c_mask = slice(None)
        
    cx = context.xs[b_idx][c_mask].detach().cpu().numpy()
    cy = context.ys[b_idx][c_mask].detach().cpu().numpy()
    cv = context.values[b_idx][c_mask].detach().cpu().numpy()
    
    tx = target.xs[b_idx].detach().cpu().numpy()
    ty = target.ys[b_idx].detach().cpu().numpy()
    tv = target.values[b_idx].detach().cpu().numpy()
    pv = pred_s[b_idx].detach().cpu().numpy().flatten()
    
    # Check limits from data if needed
    x_max = max(tx.max(), cx.max(), x_lim)
    y_max = max(ty.max(), cy.max(), y_lim)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    axes[2].set_title(f"Pred (Ep {epoch})")
    axes[2].set_xlim(0, x_max)
    axes[2].set_ylim(0, y_max)
    plt.colorbar(sc3, ax=axes[2])
    
    save_path = os.path.join(save_dir, f"static_case_{case_id}_ep{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved visualization to {save_path}")

@hydra.main(version_base=None, config_path="../../config", config_name="training/pinn_conv_cnp_train")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    log.info("Debugging ConvCNP on Static Data (NLL Loss)")
    
    data_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.data_path
    
    ds = GlobalSmokeDataset(
        data_path=str(data_path),
        context_frames=1, 
        target_frames=0, 
        min_points_ratio=0.05,
        max_points_ratio=0.5,
        mode='val', 
        max_samples=1
    )
    
    x_size = getattr(ds, 'x_size', 50.0)
    y_size = getattr(ds, 'y_size', 50.0)
    
    batch_size = cfg.training.data.batch_size if cfg.training.data.batch_size else 4
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=pinn_collate_fn)
    print("Datasize:", ds.__len__())
    
    spatial_max = cfg.training.model.spatial_max
    model = PINN_Conv_CNP(
        grid_res=cfg.training.model.grid_res,
        hidden_dim=cfg.training.model.hidden_dim,
        out_mode=cfg.training.model.out_mode,
        spatial_min=0.0,
        spatial_max=spatial_max,
        temporal_max=cfg.training.model.temporal_max,
        use_fourier_features=False
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    epochs = 600
    epoch_pbar = tqdm(range(epochs), desc="Training Epochs")
    
    history = []
    
    for epoch in epoch_pbar:
        model.train()
        batch_losses = []
        
        batch_pbar = tqdm(loader, desc=f"Ep {epoch}", leave=False)
        
        for i, batch in enumerate(batch_pbar):
            full_obs = batch[0]
            B, N = full_obs.xs.shape
            
            ratio = np.random.uniform(0.05, 0.25)
            num_ctx = max(10, int(N * ratio))
            
            c_xs = full_obs.xs[:, :num_ctx].to(device)
            c_ys = full_obs.ys[:, :num_ctx].to(device)
            c_ts = full_obs.ts[:, :num_ctx].to(device)
            c_vs = full_obs.values[:, :num_ctx].to(device)
            c_mask = full_obs.mask[:, :num_ctx].to(device)
            
            ctx = ObsPINN(xs=c_xs, ys=c_ys, ts=c_ts, values=c_vs, mask=c_mask)
            
            trg = full_obs
            trg.xs = trg.xs.to(device)
            trg.ys = trg.ys.to(device)
            trg.ts = trg.ts.to(device)
            trg.values = trg.values.to(device)
            trg.mask = trg.mask.to(device)
            
            optimizer.zero_grad()
            output = model(ctx, trg) 
            
            dist = output.smoke_dist
            target_vals = trg.values.unsqueeze(-1)

            print("Mean:", dist.loc.mean())
            print("Max:", dist.loc.max())
            print("Min:", dist.loc.min())
            
            log_p = dist.log_prob(target_vals)
            
            if trg.mask is not None:
                valid = (~trg.mask).float().unsqueeze(-1)
                loss_nll = - (log_p * valid).sum() / (valid.sum() + 1e-5)
            else:
                loss_nll = - log_p.mean()
                
            loss_nll.backward()
            optimizer.step()
            batch_losses.append(loss_nll.item())
            
            batch_pbar.set_postfix({'b_loss': f"{loss_nll.item():.4f}"})
        
        avg_loss = np.mean(batch_losses)
        history.append(avg_loss)
        
        # Log to create history in terminal
        # Using tqdm.write avoids breaking the progress bar layout
        tqdm.write(f"Epoch {epoch:02d}: Avg NLL {avg_loss:.5f}")
        
        epoch_pbar.set_postfix({'Avg NLL': f"{avg_loss:.4f}"})

    # --- Final Visualization (5 Cases) ---
    log.info("Generating Final Visualizations (5 cases)...")
    model.eval()
    cases_plotted = 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if cases_plotted >= 5: break
            
            full_obs = batch[0]
            B, N = full_obs.xs.shape
            num_ctx = max(10, int(N * 0.1))
            
            c_xs = full_obs.xs[:, :num_ctx].to(device)
            c_ys = full_obs.ys[:, :num_ctx].to(device)
            c_ts = full_obs.ts[:, :num_ctx].to(device)
            c_vs = full_obs.values[:, :num_ctx].to(device)
            c_mask = full_obs.mask[:, :num_ctx].to(device)
            
            ctx = ObsPINN(xs=c_xs, ys=c_ys, ts=c_ts, values=c_vs, mask=c_mask)
            
            trg = full_obs
            trg.xs = trg.xs.to(device)
            trg.ys = trg.ys.to(device)
            trg.ts = trg.ts.to(device)
            trg.values = trg.values.to(device)
            trg.mask = trg.mask.to(device)
            
            output = model(ctx, trg)
            pred = output.smoke_dist.loc
            
            for b_idx in range(B):
                if cases_plotted >= 5: break
                visualize_reconstruction(ctx, trg, pred, b_idx, output_dir, epochs, cases_plotted, x_lim=x_size, y_lim=y_size)
                cases_plotted += 1

    log.info(f"Debugging Run Complete. Visualized {cases_plotted} cases. Output: {output_dir}")

if __name__ == "__main__":
    main()
