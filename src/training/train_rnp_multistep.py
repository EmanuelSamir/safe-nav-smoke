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
from src.models.model_free.rnp_multistep import RNPMultistep, RNPConfig
from src.models.shared.datasets import SequentialDataset, sequential_collate_fn
from pathlib import Path
from src.models.shared.schedulers import LinearBetaScheduler

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

def log_visualization_to_tensorboard(model, loader, device, writer, epoch, forecast_horizon):
    model.eval()
    
    # Get a batch
    ctx, trg, _ = next(iter(loader))
    
    # Move to device
    ctx = ctx.to(device)
    trg = trg.to(device)

    # Context: 0..T-1. Target: 1..T
    ctx_seq = slice_obs(ctx, 0, -1) 
    
    # RNP State
    state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
    
    T = ctx_seq.xs.shape[1]
    
    predictions = [] # List of List[Normal]
    
    with torch.no_grad():
        for t in range(T):
            # Slice single step
            ctx_t = slice_obs(ctx_seq, t, t+1) # Context at t
            
            # Target at t (contains future values for t+1...t+H)
            trg_t = slice_obs(trg, t, t+1) # Dimensions (B, 1, P, H)
            
            output = model(state, context_obs=ctx_t, target_obs=trg_t)
            state = output.state
            
            # output.prediction is List[Normal] of length H
            predictions.append(output.prediction)
        
        # Visualize
        cases = 1 # Just 4 cases for visualization
        # We want to show multiple steps for a single case.
        # Format: Rows = Examples. Cols = Steps (GT vs Pred).
        
        # Let's show 2 examples.
        # For each example, show 3 time steps (e.g., t=10, 15, 20).
        # For each time step, show forecast horizon steps (e.g., h=1, 3, 5).
        
        num_examples = 2
        visualize_t_indices = [T//2] # Middle of sequence
        horizon_indices = [0, forecast_horizon//2, forecast_horizon-1] # Start, Mid, End
        
        cols = len(horizon_indices) * 3 # GT, Pred Mean, Pred Std
        rows = num_examples * len(visualize_t_indices)
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1: axes = axes[None, :]
        if cols == 1: axes = axes[:, None]
        
        for i in range(num_examples):
            # Batch index
            b_idx = i % ctx.xs.shape[0]
            
            for j, t_vis in enumerate(visualize_t_indices):
                row_idx = i * len(visualize_t_indices) + j
                
                # Get GT info for step t_vis
                # trg at t_vis contains values for t_vis+1 ... t_vis+H
                trg_obs = slice_obs(trg, t_vis, t_vis+1) # (B, 1, P, H)
                mask = trg_obs.mask[b_idx, 0].squeeze() # (P) - Fix Squeeze
                xs = trg_obs.xs[b_idx, 0].squeeze()[mask].cpu().numpy()
                ys = trg_obs.ys[b_idx, 0].squeeze()[mask].cpu().numpy()
                
                # Get Predictions
                # predictions[t_vis] is List[Normal] of length H
                preds_at_t = predictions[t_vis]
                
                for k, h_idx in enumerate(horizon_indices):
                    if h_idx >= len(preds_at_t): continue
                    
                    col_offset = k * 3
                    
                    # GT
                    # trg_obs.values: (B, 1, P, H)
                    gt_val = trg_obs.values[b_idx, 0, :, h_idx][mask].cpu().numpy()
                    
                    # Pred
                    pred_dist = preds_at_t[h_idx]
                    pred_mean = pred_dist.mean[b_idx, 0, :, 0][mask].cpu().numpy() # (B, 1, P, 1)
                    pred_std = pred_dist.stddev[b_idx, 0, :, 0][mask].cpu().numpy()
                    
                    # Plot GT
                    ax = axes[row_idx, col_offset]
                    sc = ax.scatter(xs, ys, c=gt_val, s=10, vmin=0, vmax=1, cmap='rainbow')
                    ax.set_title(f"GT (t={t_vis}, h={h_idx+1})")
                    plt.colorbar(sc, ax=ax)
                    
                    # Plot Pred
                    ax = axes[row_idx, col_offset+1]
                    sc = ax.scatter(xs, ys, c=pred_mean, s=10, vmin=0, vmax=1, cmap='rainbow')
                    ax.set_title(f"Pred Mean (t={t_vis}, h={h_idx+1})")
                    plt.colorbar(sc, ax=ax)
                    
                    # Plot Std
                    ax = axes[row_idx, col_offset+2]
                    sc = ax.scatter(xs, ys, c=pred_std, s=10, vmin=0, vmax=1, cmap='rainbow')
                    ax.set_title(f"Pred Std (t={t_vis}, h={h_idx+1})")
                    plt.colorbar(sc, ax=ax)

        plt.tight_layout()
        
        # To Tensorboard
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        writer.add_image(f"Val/Forecast_Horizon_Vis", image_tensor, epoch)
        plt.close()

@hydra.main(version_base=None, config_path="../../config/training", config_name="rnp_train")
def train(cfg: DictConfig):
    
    # 1. Setup
    print(f"Training RNP Multistep with config: {cfg.training.experiment_name}")
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
        
    data_path_str = cfg.training.data.get("data_path")
    data_path = root_dir / data_path_str
    
    if not data_path.exists():
        log.error(f"Data not found at: {data_path}")
        return
        
    seq_len = cfg.training.data.sequence_length
    max_episodes = cfg.training.data.get("max_samples", None)
    
    forecast_horizon = cfg.training.model.get("forecast_horizon", 5)
    
    train_dataset = SequentialDataset(
        data_path=str(data_path),
        sequence_length=seq_len,
        forecast_horizon=forecast_horizon,
        mode='train',
        train_split=cfg.training.data.train_split,
        max_episodes=max_episodes
    )
    
    val_dataset = SequentialDataset(
        data_path=str(data_path),
        sequence_length=seq_len,
        forecast_horizon=forecast_horizon,
        mode='val',
        train_split=cfg.training.data.train_split,
        max_episodes=max_episodes
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.data.batch_size, 
        shuffle=True, 
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

    model = RNPMultistep(params, forecast_horizon=forecast_horizon).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # 4. Optimizer
    opt_cfg = cfg.training.optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_cfg.max_epochs, eta_min=opt_cfg.min_lr)

    # 5. Training Loop
    best_val_ll = -float('inf')
    sampling_scheduler = LinearBetaScheduler(
        beta_start=cfg.training.sampling.beta_start,
        beta_end=cfg.training.sampling.beta_end,
        num_steps=cfg.training.sampling.warmup_epochs
    )

    
    for epoch in range(opt_cfg.max_epochs):
        model.train()
        train_loss = 0
        train_ll = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # batch is (ctx_seq, trg_seq, idx)
            ctx, trg, _ = batch
            ctx = ctx.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            
            # Init state
            state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
            
            T = ctx.xs.shape[1]
            loss_over_time = 0
            
            # Iterate time steps
            predictions_buffer = {} # maps t -> list of prediction Obs or values
            ratio_force = 0
            
            for t in range(T):
                # Slice single step
                ctx_t = slice_obs(ctx, t, t+1)
                
                # Target contains H steps of values. 
                # Shape: (B, 1, P, H) if properly collated.
                trg_t = slice_obs(trg, t, t+1) 
                
                # Check Scheduler for Scheduled Sampling
                # Available predictions for step t
                if t in predictions_buffer and len(predictions_buffer[t]) > 0:
                    rand_number = random.random()
                    if rand_number < sampling_scheduler.update(epoch):
                        # Randomly pick from previous predictions for time t
                        sampled_idx = random.randint(0, len(predictions_buffer[t]) - 1)
                        
                        # Use predicted Obs as context for current step
                        ctx_t = predictions_buffer[t][sampled_idx]
                        ratio_force += 1
                
                # Forward pass (single step input, multi-step output)
                output = model(state, context_obs=ctx_t, target_obs=trg_t)
                state = output.state 
                
                # output.prediction is List[Normal] of length H
                predictions = output.prediction # List[Normal] len H
                
                # Calculate loss over H steps
                # sum NLL for each step
                step_loss = 0
                
                if trg_t.mask is not None:
                     mask = trg_t.mask.float() # (B, 1, P, 1) or (B, 1, P, H)
                     count = mask.sum(dim=-2).clamp(min=1.0) # (B, 1, 1, 1)
                
                for h_idx in range(forecast_horizon):
                    # Ground Truth for step h
                    # trg_t.values is (B, 1, P, H)
                    gt_h = trg_t.values[..., h_idx].unsqueeze(-1) # (B, 1, P, 1)
                    
                    pred_dist = predictions[h_idx] # Normal(mu, sigma) (B, 1, P, 1)
                    
                    ll_per_point = pred_dist.log_prob(gt_h) # (B, 1, P, 1)
                    
                    if trg_t.mask is not None:
                         # Mask same for all H
                         ll_step_h = (ll_per_point * mask).sum(dim=-2) / count
                         # ll_step_h is (B, 1, 1)
                    else:
                         ll_step_h = ll_per_point.mean(dim=-2)
                         
                    step_loss += -ll_step_h.mean()
                
                step_loss = step_loss / forecast_horizon
                
                loss_over_time += step_loss
                
                # Store predictions for future steps in the buffer (detached samples)
                with torch.no_grad():
                    for h_idx in range(forecast_horizon):
                        pred_t = t + 1 + h_idx
                        if pred_t not in predictions_buffer:
                            predictions_buffer[pred_t] = []
                        
                        sampled_vals = predictions[h_idx].sample() # usually (B, 1, P, 1) or (B, P, 1)
                        
                        # We must assure shape matches context expectations (B, 1, P, 1)
                        if sampled_vals.dim() == 3: # (B, P, 1) -> (B, 1, P, 1)
                             sampled_vals = sampled_vals.unsqueeze(1)
                        elif sampled_vals.dim() == 2: # (B, P) -> (B, 1, P, 1)
                             sampled_vals = sampled_vals.unsqueeze(1).unsqueeze(-1)
                        elif sampled_vals.shape[1] > 1: # if for some reason it's (B, T, P, 1)
                             sampled_vals = sampled_vals[:, 0:1, :, :]
                             
                        pred_obs = Obs(
                            xs=trg_t.xs,
                            ys=trg_t.ys,
                            values=sampled_vals,
                            mask=trg_t.mask,
                            ts=trg_t.ts
                        )
                        predictions_buffer[pred_t].append(pred_obs)
                
                # Clear memory for current step t
                if t in predictions_buffer:
                    del predictions_buffer[t]

            # Average loss over sequence time steps
            loss = loss_over_time / T
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            train_ll += -loss.item() # This is approximate

            pbar.set_postfix({
                'loss': loss.item(),
                'force_ratio': f"{ratio_force}/{T}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
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
                
                state = model.init_state(batch_size=ctx.xs.shape[0], device=device)
                
                T = ctx.xs.shape[1]
                val_ll_batch = 0
                
                for t in range(T):
                    ctx_t = slice_obs(ctx, t, t+1)
                    trg_t = slice_obs(trg, t, t+1)

                    output = model(state, context_obs=ctx_t, target_obs=trg_t)
                    state = output.state
                    
                    # Validation only on next step (h=0) per user request
                    # "El validation step de este nuevo modelo debe ser solo con el siguiente prediction"
                    pred_dist = output.prediction[0]
                    
                    gt_val = trg_t.values[..., 0].unsqueeze(-1) # h=0
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
        writer.add_scalar("Val/LL_NextStep", avg_val_ll, epoch)
        log.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val LL (Next Step)={avg_val_ll:.4f}")
        
        # Visualization
        if epoch % cfg.training.visualizer.visualize_every == 0:
             log_visualization_to_tensorboard(model, val_loader, device, writer, epoch, forecast_horizon)
             
        # Checkpoint
        save_checkpoint(model, optimizer, epoch, -avg_val_ll, params, os.path.join(ckpt_dir, "last_model.pt"))
        if avg_val_ll > best_val_ll:
            best_val_ll = avg_val_ll
            save_checkpoint(model, optimizer, epoch, -avg_val_ll, params, os.path.join(ckpt_dir, "best_model.pt"))

    writer.close()

if __name__ == "__main__":
    train()
