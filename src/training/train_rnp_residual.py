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
from src.models.model_free.rnp_residual import RNPResidual
from src.models.model_free.rnp import RNPConfig
from src.models.shared.datasets import SequentialDataset, sequential_collate_fn
from pathlib import Path# Fixed import assumption
from src.models.shared.optimizer import LinearBetaScheduler

log = logging.getLogger(__name__)

from src.models.shared.observations import Obs, slice_obs

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

    # Context: 0..T-1. Target: 1..T
    ctx_seq = slice_obs(ctx, 0, -1)
    trg_seq = slice_obs(trg, 1, None)
    
    # RNP State
    state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
    
    T = ctx_seq.xs.shape[1]
    
    predictions = []
    
    with torch.no_grad():
        for t in range(T):
            # Slice single step
            ctx_t = slice_obs(ctx_seq, t, t+1)
            trg_t = slice_obs(trg_seq, t, t+1)
            
            output = model(state, context_obs=ctx_t, target_obs=trg_t)
            state = output.state

            predictions.append(output.prediction[1])
        
        cases = 4
        fig, axes = plt.subplots(cases, 3, figsize=(15, 3*cases))

        for i in range(cases):
            # Visualize a random time step
            t_vis = np.random.randint(0, T)
            
            # Use different batch index if possible, else wrap around
            curr_b_idx = i % ctx.xs.shape[0]

            # Ground Truth from trg_seq (shifted)
            if trg_seq.mask is not None:
                mask_t = trg_seq.mask[curr_b_idx, t_vis].squeeze().cpu().numpy()
            else:
                mask_t = np.ones_like(trg_seq.xs[curr_b_idx, t_vis].squeeze().cpu().numpy(), dtype=bool)

            xs = trg_seq.xs[curr_b_idx, t_vis].squeeze().cpu().numpy()[mask_t]
            ys = trg_seq.ys[curr_b_idx, t_vis].squeeze().cpu().numpy()[mask_t]
            gt = trg_seq.values[curr_b_idx, t_vis].squeeze().cpu().numpy()[mask_t]
            
            # Prediction
            # predictions is list of normals. 
            pred_dist_t = predictions[t_vis]
            pr = pred_dist_t.mean[curr_b_idx, 0].squeeze().cpu().numpy()[mask_t]
            pr_std = pred_dist_t.stddev[curr_b_idx, 0].squeeze().cpu().numpy()[mask_t]
            
            sc = axes[i, 0].scatter(xs, ys, c=gt, s=20, vmin=0, vmax=1, cmap='rainbow')
            axes[i, 0].set_title(f"GT Frame: {t_vis} | Batch: {curr_b_idx}")
            plt.colorbar(sc, ax=axes[i, 0])
            
            sc2 = axes[i, 1].scatter(xs, ys, c=pr, s=20, vmin=0, vmax=1, cmap='rainbow')
            axes[i, 1].set_title(f"Pred Frame: {t_vis} | Ctx Len: {ctx.xs.shape}")
            plt.colorbar(sc2, ax=axes[i, 1])

            sc3 = axes[i, 2].scatter(xs, ys, c=pr_std, s=20, vmin=0, vmax=1, cmap='rainbow')
            axes[i, 2].set_title(f"Pred Std Frame: {t_vis} | Ctx Len: {ctx.xs.shape}")
            plt.colorbar(sc3, ax=axes[i, 2])
            
        plt.tight_layout()
        
        # To Tensorboard
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        writer.add_image("Val/Prediction", image_tensor, epoch)
        plt.close()

@hydra.main(version_base=None, config_path="../../config/training", config_name="rnp_residual_train")
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
        shuffle=True, 
        collate_fn=sequential_collate_fn,
        num_workers=cfg.training.data.num_workers
    )

    # 3. Model
    model_cfg = cfg.training.model
    params = RNPConfig(
        h_dim=model_cfg.hidden_dim,
        decoder_num_layers=model_cfg.get('num_layers', 3), 
        lstm_num_layers=model_cfg.lstm_layers,
        decoder_fourier_size=model_cfg.get('decoder_fourier_size', 128),
        decoder_fourier_scale=model_cfg.get('decoder_fourier_scale', 20.0),
    )

    model = RNPResidual(params).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # 4. Optimizer
    opt_cfg = cfg.training.optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_cfg.max_epochs, eta_min=opt_cfg.min_lr)

    sampling_scheduler = LinearBetaScheduler(
        beta_start=opt_cfg.sampling.beta_start,
        beta_end=opt_cfg.sampling.beta_end,
        num_steps=opt_cfg.sampling.warmup_epochs
    )

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
            trg_seq = slice_obs(trg, 1, None)
            ctx_seq = slice_obs(ctx, 0, -1)
            
            optimizer.zero_grad()
            
            # Init state
            state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
            
            T = ctx_seq.xs.shape[1]
            loss_over_time = 0
            prev_pred_obs = None
            ratio_force = 0.0 
            
            for t in range(T):
                # Slice single step
                ctx_t = slice_obs(ctx_seq, t, t+1)
                trg_t = slice_obs(trg_seq, t, t+1)

                if prev_pred_obs is not None:
                    rand_number = random.random()
                    if rand_number < sampling_scheduler.update(epoch):
                        # Forces to learn longer sequences
                        ctx_t = prev_pred_obs
                        ratio_force += 1
                
                # Forward pass (single step)
                output = model(state, context_obs=ctx_t, target_obs=trg_t)
                state = output.state # Update state for next step
                
                pred_dist_t = output.prediction[0]
                pred_dist_tp1 = output.prediction[1]
                
                # Loss Calculation (One Step)
                gt_val = ctx_t.values # (B, 1, P, 1)
                
                # Log Prob: (B, 1, P, 1)
                ll_per_point_t = pred_dist_t.log_prob(gt_val)
                    
                if ctx_t.mask is not None:
                     mask = ctx_t.mask.float() # (B, 1, P, 1)
                     count = mask.sum(dim=-2).clamp(min=1.0) # (B, 1, 1)
                     ll_step_t = (ll_per_point_t * mask).sum(dim=-2) / count
                else:
                     ll_step_t = ll_per_point_t.mean(dim=-2)

                gt_val = trg_t.values # (B, 1, P, 1)
                ll_per_point_tp1 = pred_dist_tp1.log_prob(gt_val)

                if trg_t.mask is not None:
                     mask = trg_t.mask.float() # (B, 1, P, 1)
                     count = mask.sum(dim=-2).clamp(min=1.0) # (B, 1, 1)
                     ll_step_tp1 = (ll_per_point_tp1 * mask).sum(dim=-2) / count
                else:
                     ll_step_tp1 = ll_per_point_tp1.mean(dim=-2)

                prev_pred_obs = trg_t
                prev_pred_obs.values = pred_dist_tp1.sample()
                
                # ll_step is (B, 1, 1). Mean over batch.
                loss_over_time += -ll_step_t.mean() - ll_step_tp1.mean()

            ratio_force /= T
            print(f"Ratio force: {ratio_force}")

            # Average loss over time steps (or sum? Backprop through time)
            # Standard is mean over time
            loss = loss_over_time / T
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            train_ll += -loss.item()

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
                trg_seq = slice_obs(trg, 1, None)
                ctx_seq = slice_obs(ctx, 0, -1)

                state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
                
                T = ctx_seq.xs.shape[1]
                val_ll_batch = 0
                
                # Validation Loop
                for t in range(T):
                    ctx_t = slice_obs(ctx_seq, t, t+1)
                    trg_t = slice_obs(trg_seq, t, t+1)

                    output = model(state, context_obs=ctx_t, target_obs=trg_t)
                    state = output.state
                    pred_dist = output.prediction[1]
                    
                    gt_val = trg_t.values
                    ll_per_point = pred_dist.log_prob(gt_val)
                    
                    if trg_t.mask is not None:
                         mask = trg_t.mask.float()
                         count = mask.sum(dim=-2).clamp(min=1.0)
                         ll_step = (ll_per_point * mask).sum(dim=-2) / count
                    else:
                         ll_step = ll_per_point.mean(dim=-2)
                         
                    val_ll_batch += ll_step.mean().item()
                
                val_ll += val_ll_batch / T

        avg_val_ll = val_ll / len(val_loader)
        writer.add_scalar("Val/LL", avg_val_ll, epoch)
        log.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val LL={avg_val_ll:.4f}")
        
        # Visualization
        if epoch % cfg.training.visualizer.visualize_every == 0:
             log_visualization_to_tensorboard(model, val_loader, device, writer, epoch)
             
        # Checkpoint
        save_checkpoint(model, optimizer, epoch, -avg_val_ll, params, os.path.join(ckpt_dir, "last_model.pt"))
        if avg_val_ll > best_val_ll:
            best_val_ll = avg_val_ll
            save_checkpoint(model, optimizer, epoch, -avg_val_ll, params, os.path.join(ckpt_dir, "best_model.pt"))

    writer.close()

if __name__ == "__main__":
    train()
