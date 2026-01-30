"""
Clean training script for SNP v1 model.
Standardized structure, no backward compatibility.
"""
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
from pathlib import Path
from tqdm import tqdm
import logging

from src.models.model_free.snp_v1 import SNP_v1 as SNP, SNPConfig
from src.models.shared.datasets import SequentialDataset, collate_sequences
from envs.replay_buffer import GenericReplayBuffer

import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms

log = logging.getLogger(__name__)

def log_visualization_to_tensorboard(model, loader, device, writer, epoch):
    model.eval()
    model.eval()
    
    # Select random sample from entire dataset
    dataset = loader.dataset
    idx = np.random.randint(0, len(dataset))
    sample = dataset[idx]
    
    batch = collate_sequences([sample])
    inputs = batch[:-1]
    targets = batch[1:]
    
    batch_size = inputs[0]['action'].shape[0]
    
    # Select random sample and time step
    sample_idx = 0
    time_idx = np.random.randint(0, len(inputs))
    
    state = model.init_state(batch_size, device)
    
    with torch.no_grad():
        for t, (step_in, step_target) in enumerate(zip(inputs, targets)):
            out = model(
                state=state,
                action=step_in['action'].to(device) if model.config.use_actions else None,
                obs=step_in['obs'].to(device),
                query=step_target['obs'].to(device)
            )
            state = out.state
            
            if t == time_idx:
               gt = step_target['obs'].values[sample_idx].cpu().numpy()
               pr = out.prediction.loc[sample_idx].squeeze().cpu().numpy()
               xs = step_target['obs'].xs[sample_idx].cpu().numpy()
               ys = step_target['obs'].ys[sample_idx].cpu().numpy()
               
               fig, axes = plt.subplots(1, 2, figsize=(10, 5))
               sc = axes[0].scatter(xs, ys, c=gt, s=20, vmin=0, vmax=1, cmap='gray')
               axes[0].set_title(f"GT Frame: {time_idx+1} (dt=+1)")
               plt.colorbar(sc, ax=axes[0])
               
               sc2 = axes[1].scatter(xs, ys, c=pr, s=20, vmin=0, vmax=1, cmap='gray')
               axes[1].set_title(f"Pred Frame: {time_idx+1} | Context: {time_idx}")
               plt.colorbar(sc2, ax=axes[1])
               
               plt.tight_layout()
               
               # To Tensorboard
               buf = BytesIO()
               plt.savefig(buf, format='png')
               buf.seek(0)
               image = Image.open(buf)
               image_tensor = transforms.ToTensor()(image)
               writer.add_image("Val/Prediction", image_tensor, epoch)
               plt.close()


@hydra.main(version_base=None, config_path="../../config", config_name="training/snp_v1_train")
def main(cfg: DictConfig):
    # Setup
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Device: {device}")
    log.info(f"Output: {output_dir}")
    
    # Save config
    with open(Path(output_dir) / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # Create subdirectories
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    
    # Data
    buffer_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.buffer_path
    buffer = GenericReplayBuffer(buffer_size=1, data_keys=[])
    buffer.load_from_file(str(buffer_path))
    
    datasize = min(buffer.current_size, cfg.training.data.max_samples) if cfg.training.data.max_samples else buffer.current_size
    seq_len = cfg.training.data.sequence_length
    
    valid_indices = np.arange(datasize - seq_len - 1)
    split_idx = int(len(valid_indices) * cfg.training.data.train_split)
    
    train_ds = SequentialDataset(
        buffer, 
        seq_len, 
        valid_indices[:split_idx],
        use_actions=cfg.training.model.get('use_actions', True),
        action_dim=cfg.training.model.get('action_dim', 2),
        use_robot_state=cfg.training.model.get('use_robot_state', False),
        robot_state_dim=cfg.training.model.get('robot_state_dim', 0),
        info_ratio_per_frame=cfg.training.data.get("info_ratio_per_frame", 0.2)
    )
    val_ds = SequentialDataset(
        buffer, 
        seq_len, 
        valid_indices[split_idx:],
        use_actions=cfg.training.model.get('use_actions', True),
        action_dim=cfg.training.model.get('action_dim', 2),
        use_robot_state=cfg.training.model.get('use_robot_state', False),
        robot_state_dim=cfg.training.model.get('robot_state_dim', 0),
        info_ratio_per_frame=cfg.training.data.get("info_ratio_per_frame", 0.2)
    )
    
    train_sampler = None
    shuffle_train = True
    epoch_sample_ratio = cfg.training.data.get("epoch_sample_ratio", None)
    
    if epoch_sample_ratio is None:
         # Backward compatibility
         samples_per_epoch = cfg.training.data.get("samples_per_epoch", None)
         if samples_per_epoch is not None:
              num_samples = min(samples_per_epoch, len(train_ds))
              train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=False, num_samples=num_samples)
              shuffle_train = False
    elif epoch_sample_ratio < 1.0:
         num_samples = int(len(train_ds) * epoch_sample_ratio)
         num_samples = max(1, num_samples)
         train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=False, num_samples=num_samples)
         shuffle_train = False
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.data.batch_size, 
                              shuffle=shuffle_train, sampler=train_sampler, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.data.batch_size,
                           shuffle=False, collate_fn=collate_sequences)
    
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    model_config = SNPConfig(
        z_dim=cfg.training.model.z_dim,
        h_dim=cfg.training.model.h_dim,
        r_dim=cfg.training.model.r_dim,
        encoder_hidden=cfg.training.model.encoder_hidden,
        decoder_hidden=cfg.training.model.decoder_hidden,
        prior_hidden=cfg.training.model.prior_hidden,
        posterior_hidden=cfg.training.model.posterior_hidden,
        use_actions=cfg.training.model.use_actions,
        action_dim=cfg.training.model.action_dim,
        aggregator=cfg.training.model.get('aggregator', 'mean'),
        use_fourier_encoder=cfg.training.model.get('use_fourier_encoder', False),
        use_fourier_decoder=cfg.training.model.get('use_fourier_decoder', False),
        fourier_frequencies=cfg.training.model.get('fourier_frequencies', 128),
        fourier_scale=cfg.training.model.get('fourier_scale', 20.0),
        spatial_max=cfg.training.model.get('spatial_max', 100.0)
    )
    model = SNP(model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(cfg.training.optimizer.max_epochs):
        # Train
        model.train()
        train_loss, train_kl, train_mse = 0, 0, 0
        beta = min(cfg.training.optimizer.beta_max, epoch / 10.0)
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, targets = batch[:-1], batch[1:]
            B = inputs[0]['action'].shape[0]
            state = model.init_state(B, device)
            
            optimizer.zero_grad()
            seq_loss, seq_kl, seq_mse = 0, 0, 0
            
            for step_in, step_tgt in zip(inputs, targets):
                out = model(
                    state=state,
                    action=step_in['action'].to(device) if cfg.training.model.use_actions else None,
                    obs=step_in['obs'].to(device),
                    query=step_tgt['obs'].to(device)
                )
                
                # KL divergence
                if out.post_mu is not None:
                    dist_post = torch.distributions.Normal(out.post_mu, out.post_sigma)
                    dist_prior = torch.distributions.Normal(out.prior_mu, out.prior_sigma)
                    kl = torch.distributions.kl_divergence(dist_post, dist_prior).sum(-1)
                else:
                    kl = torch.zeros(B, device=device)
                
                # MSE
                tgt_obs = step_tgt['obs'].to(device)
                tgt_vals = tgt_obs.values
                if tgt_vals.dim() == 2:
                    tgt_vals = tgt_vals.unsqueeze(-1)  # (B, N) -> (B, N, 1)
                
                pred_vals = out.prediction.loc
                mse_per_point = (pred_vals - tgt_vals) ** 2
                
                mask = step_tgt['obs'].to(device).mask
                if mask is not None:
                    valid = mask.float().unsqueeze(-1)
                    mse = (mse_per_point * valid).sum(1) / valid.sum(1).clamp(min=1.0)
                else:
                    mse = mse_per_point.mean(1)
                
                # Loss
                valid_step = (1.0 - step_in['done'].to(device)).squeeze()
                step_loss = (beta * kl + mse.squeeze(-1)) * valid_step
                
                seq_loss += step_loss.mean()
                seq_kl += (kl * valid_step).mean()
                seq_mse += (mse.squeeze(-1) * valid_step).mean()
                
                state = out.state
            
            seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.optimizer.grad_clip)
            optimizer.step()
            
            train_loss += seq_loss.item()
            train_kl += seq_kl.item()
            train_mse += seq_mse.item()
        
        train_loss /= len(train_loader)
        train_kl /= len(train_loader)
        train_mse /= len(train_loader)
        
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/KL", train_kl, epoch)
        writer.add_scalar("Train/MSE", train_mse, epoch)
        
        # Validation
        model.eval()
        val_loss, val_kl, val_mse = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[:-1], batch[1:]
                state = model.init_state(inputs[0]['action'].shape[0], device)
                
                for step_in, step_tgt in zip(inputs, targets):
                    out = model(
                        state=state,
                        action=step_in['action'].to(device) if cfg.training.model.use_actions else None,
                        obs=step_in['obs'].to(device),
                        query=step_tgt['obs'].to(device)
                    )
                    
                    if out.post_mu is not None:
                        dist_post = torch.distributions.Normal(out.post_mu, out.post_sigma)
                        dist_prior = torch.distributions.Normal(out.prior_mu, out.prior_sigma)
                        kl = torch.distributions.kl_divergence(dist_post, dist_prior).sum(-1)
                    else:
                        kl = torch.zeros(state[0].shape[0], device=device)
                    
                    tgt_obs = step_tgt['obs'].to(device)
                    tgt_vals = tgt_obs.values
                    if tgt_vals.dim() == 2:
                        tgt_vals = tgt_vals.unsqueeze(-1)
                    
                    mse_per_point = (out.prediction.loc - tgt_vals) ** 2
                    mask = tgt_obs.mask
                    
                    if mask is not None:
                        valid = mask.float().unsqueeze(-1)
                        mse = (mse_per_point * valid).sum(1) / valid.sum(1).clamp(min=1.0)
                    else:
                        mse = mse_per_point.mean(1)
                    
                    valid_step = (1.0 - step_in['done'].to(device)).squeeze()
                    val_loss += ((beta * kl + mse.squeeze(-1)) * valid_step).mean().item()
                    val_kl += (kl * valid_step).mean().item()
                    val_mse += (mse.squeeze(-1) * valid_step).mean().item()
                    
                    state = out.state
        
        val_loss /= len(val_loader)
        val_kl /= len(val_loader)
        val_mse /= len(val_loader)
        
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/KL", val_kl, epoch)
        writer.add_scalar("Val/MSE", val_mse, epoch)
        
        
        log.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Protocol Evaluation (10 context -> 15 forecast)
        # Create a loader for 25-step sequences
        valid_indices_25 = val_loader.dataset.indices
        valid_indices_25 = valid_indices_25[valid_indices_25 < (datasize - 26)]
        
        if len(valid_indices_25) > 0:
            val_ds_proto = SequentialDataset(
                buffer, 
                25, 
                valid_indices_25,
                use_actions=cfg.training.model.get('use_actions', True),
                action_dim=cfg.training.model.get('action_dim', 2),
                use_robot_state=cfg.training.model.get('use_robot_state', False),
                robot_state_dim=cfg.training.model.get('robot_state_dim', 0)
            )
            val_loader_proto = DataLoader(val_ds_proto, batch_size=cfg.training.data.batch_size, shuffle=False, collate_fn=collate_sequences)
            
            from src.utils.eval_protocol import evaluate_forecast_protocol
            proto_mse = evaluate_forecast_protocol(model, val_loader_proto, device, model_type="model_free")
            writer.add_scalar("Protocol_10_15/Dataset_MSE", proto_mse, epoch)
            log.info(f"Protocol 10-15 MSE: {proto_mse:.4f}")

        # Visualization
        if epoch % 1 == 0:
             log_visualization_to_tensorboard(model, val_loader, device, writer, epoch)
             from src.utils.eval_protocol import evaluate_10_15_protocol
             evaluate_10_15_protocol(model, val_ds, device, writer, epoch, model_type="model_free")

        # Save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'loss': best_val_loss
            }, Path(ckpt_dir) / "best_model.pt")
    
    writer.close()


if __name__ == "__main__":
    main()
