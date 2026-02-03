import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Imports
from src.models.model_free.rnp import RNP, RNPConfig
from src.models.shared.datasets import SequentialDataset, sequential_collate_fn
from pathlib import Path# Fixed import assumption

log = logging.getLogger(__name__)

from src.models.shared.observations import Obs

def slice_obs(obs, start=0, end=None):
    """Helper to slice Obs tensors in time dimension (dim 1)."""
    sl = slice(start, end)
    return Obs(
        xs=obs.xs[:, sl],
        ys=obs.ys[:, sl],
        ts=obs.ts[:, sl] if obs.ts is not None else None,
        values=obs.values[:, sl] if obs.values is not None else None,
        mask=obs.mask[:, sl] if obs.mask is not None else None
    )

def save_checkpoint(model, optimizer, epoch, loss, params, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'hyper_parameters': params,
    }, path)

def log_visualization_to_tensorboard(model, loader, device, writer, epoch):
    model.eval()
    
    # Get a batch
    ctx, trg, _ = next(iter(loader))
    
    # Move to device
    ctx = ctx.to(device)
    trg = trg.to(device)

    ctx = slice_obs(ctx, 0, -1)
    trg = slice_obs(trg, 1, None)
    
    # Pick first in batch
    b_idx = 0
    
    # RNP State
    state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
    
    with torch.no_grad():
        output = model(state, context_obs=ctx, target_obs=trg)
        pred_dist = output.prediction
        
        # Visualize a random time step
        T = ctx.xs.shape[1]
        t = np.random.randint(0, T)
        
        # Ground Truth
        # (B, T, P, 1) -> (P,)
        if trg.mask is not None:
             mask_t = trg.mask[b_idx, t].squeeze().cpu().numpy()
        else:
             mask_t = np.ones_like(trg.xs[b_idx, t].squeeze().cpu().numpy(), dtype=bool)

        xs = trg.xs[b_idx, t].squeeze().cpu().numpy()[mask_t]
        ys = trg.ys[b_idx, t].squeeze().cpu().numpy()[mask_t]
        gt = trg.values[b_idx, t].squeeze().cpu().numpy()[mask_t]
        
        # Prediction
        pr = pred_dist.mean[b_idx, t].squeeze().cpu().numpy()[mask_t]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sc = axes[0].scatter(xs, ys, c=gt, s=20, vmin=0, vmax=1, cmap='viridis')
        axes[0].set_title(f"GT Frame: {t}")
        plt.colorbar(sc, ax=axes[0])
        
        sc2 = axes[1].scatter(xs, ys, c=pr, s=20, vmin=0, vmax=1, cmap='viridis')
        axes[1].set_title(f"Pred Frame: {t} | Ctx Len: {ctx.xs.shape}")
        plt.colorbar(sc2, ax=axes[1])
        
        plt.tight_layout()
        
        # To Tensorboard
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        writer.add_image("Val/Prediction", image_tensor, epoch)
        plt.close()

@hydra.main(version_base=None, config_path="../../config/training", config_name="rnp_train")
def train(cfg: DictConfig):
    # 1. Setup
    print(f"Training RNP with config: {cfg.training.experiment_name}")
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")
    
    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Create subdirectories
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # 2. Data Loading
    try:
        root_dir = Path(hydra.utils.get_original_cwd())
    except:
        root_dir = Path(os.getcwd())
        
    # Assuming config has data_path or buffer_path pointing to .npz
    data_path_str = cfg.training.data.get("data_path")
    data_path = root_dir / data_path_str
    
    if not data_path.exists():
        log.error(f"Data not found at: {data_path}")
        return
        
    seq_len = cfg.training.data.sequence_length
    max_episodes = cfg.training.data.get("max_samples", None)
    
    train_dataset = SequentialDataset(
        data_path=str(data_path),
        sequence_length=seq_len,
        mode='train',
        train_split=cfg.training.data.train_split,
        max_episodes=max_episodes
    )
    
    val_dataset = SequentialDataset(
        data_path=str(data_path),
        sequence_length=seq_len,
        mode='val',
        train_split=cfg.training.data.train_split,
        max_episodes=max_episodes
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.data.batch_size, 
        shuffle=True, # SequentialDataset samples random windows, so shuffling is good
        collate_fn=sequential_collate_fn,
        num_workers=cfg.training.data.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.data.batch_size, 
        shuffle=False, 
        collate_fn=sequential_collate_fn,
        num_workers=cfg.training.data.num_workers
    )

    # 3. Model
    model_cfg = cfg.training.model
    params = RNPConfig(
        r_dim=model_cfg.embed_dim,
        h_dim=model_cfg.hidden_dim,
        # Config might have different keys, mapping best effort
        encoder_num_layers=model_cfg.get('num_layers', 3),
        decoder_num_layers=model_cfg.get('num_layers', 3), 
        lstm_num_layers=model_cfg.lstm_layers,
        use_fourier_encoder=model_cfg.get('use_fourier_encoder', False),
        use_fourier_decoder=model_cfg.get('use_fourier_decoder', False),
        fourier_frequencies=model_cfg.get('fourier_frequencies', 128),
        fourier_scale=model_cfg.get('fourier_scale', 20.0),
        spatial_max=model_cfg.get('spatial_max', 30.0)
    )
    model = RNP(params).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # 4. Optimizer
    opt_cfg = cfg.training.optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_cfg.max_epochs, eta_min=opt_cfg.min_lr)

    # 5. Training Loop
    best_val_ll = -float('inf')
    
    for epoch in range(opt_cfg.max_epochs):
        model.train()
        train_loss = 0
        train_ll = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # batch is (ctx_obs, trg_obs, idx)
            ctx, trg, _ = batch
            ctx = ctx.to(device)
            trg = trg.to(device)

            # Shift trg by 1 to predict next state
            trg = slice_obs(trg, 1, None)
            ctx = slice_obs(ctx, 0, -1)
            
            optimizer.zero_grad()
            
            # Init state
            state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
            
            # Forward pass (sequence)
            output = model(state, context_obs=ctx, target_obs=trg)
            pred_dist = output.prediction
            
            # Loss Calculation (Batched over Time and Space)
            gt_val = trg.values # (B, T, P, 1)
            
            # Log Prob: (B, T, P, 1)
            ll_per_point = pred_dist.log_prob(gt_val)
            
            if trg.mask is not None:
                 mask = trg.mask.float() # (B, T, P, 1)
                 # Sum log probs over valid points, divide by count
                 # Prevent div by zero
                 count = mask.sum(dim=-2).clamp(min=1.0) # Sum over points P -> (B, T, 1)
                 ll_step = (ll_per_point * mask).sum(dim=-2) / count
            else:
                 ll_step = ll_per_point.mean(dim=-2)
            
            # ll_step is (B, T, 1)
            # Average over Time and Batch
            ll = ll_step.mean()
            
            loss = -ll
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            train_ll += ll.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ll = train_ll / len(train_loader)
        scheduler.step()
        
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        
        # Validation
        model.eval()
        val_ll = 0
        with torch.no_grad():
            for batch in val_loader:
                ctx, trg, _ = batch
                ctx = ctx.to(device)
                trg = trg.to(device)
                
                # Forecasting Validation
                trg = slice_obs(trg, 1, None)
                ctx = slice_obs(ctx, 0, -1)

                state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
                
                output = model(state, context_obs=ctx, target_obs=trg)
                pred_dist = output.prediction
                
                gt_val = trg.values
                ll_per_point = pred_dist.log_prob(gt_val)
                
                if trg.mask is not None:
                     mask = trg.mask.float()
                     count = mask.sum(dim=-2).clamp(min=1.0)
                     ll_step = (ll_per_point * mask).sum(dim=-2) / count
                else:
                     ll_step = ll_per_point.mean(dim=-2)
                     
                val_ll += ll_step.mean().item()

        avg_val_ll = val_ll / len(val_loader)
        writer.add_scalar("Val/LL", avg_val_ll, epoch)
        log.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val LL={avg_val_ll:.4f}")
        
        # Visualization
        if epoch % cfg.training.visualizer.visualize_every == 0: # Visualize every 5 epochs
             log_visualization_to_tensorboard(model, val_loader, device, writer, epoch)
             
        # Checkpoint
        save_checkpoint(model, optimizer, epoch, -avg_val_ll, params, os.path.join(ckpt_dir, "last_model.pt"))
        if avg_val_ll > best_val_ll:
            best_val_ll = avg_val_ll
            save_checkpoint(model, optimizer, epoch, -avg_val_ll, params, os.path.join(ckpt_dir, "best_model.pt"))

    writer.close()

if __name__ == "__main__":
    train()
