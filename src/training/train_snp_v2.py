import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import logging
import datetime

from src.models.model_free.snp_v2 import SNP_v2 as SnpV2, SNP_v2_Config as SnpV2Params
from src.models.shared.observations import Obs
from src.models.shared.datasets import SequentialDataset, collate_sequences as sequential_collate_fn


from src.utils.eval_protocol import evaluate_10_15_protocol
from envs.replay_buffer import GenericReplayBuffer

log = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, loss, params, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'hyper_parameters': params,
    }, path)

@hydra.main(version_base=None, config_path="../../config", config_name="training/snp_v2_train")
def main(cfg: DictConfig):
    # 1. Setup
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Using device: {device}")
    log.info(f"Output directory: {output_dir}")
    
    # Save a visible copy of the used config
    config_path = os.path.join(output_dir, "config_used.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # Tensorboard
    # Create subdirectories
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    
    # 2. Data
    buffer_path = Path(hydra.utils.get_original_cwd()) / cfg.training.data.buffer_path
    if not buffer_path.exists():
        log.error(f"Buffer not found at: {buffer_path}")
        return

    replay_buffer = GenericReplayBuffer(buffer_size=1, data_keys=[]) 
    replay_buffer.load_from_file(str(buffer_path))
    
    datasize = replay_buffer.current_size
    if cfg.training.data.get("max_samples") is not None:
        datasize = min(datasize, cfg.training.data.max_samples)
        
    seq_len = cfg.training.data.sequence_length
    all_indices = np.arange(datasize)
    valid_indices = all_indices[all_indices < (datasize - (seq_len + 1))]
    
    n_valid = len(valid_indices)
    train_end = int(n_valid * cfg.training.data.train_split)
    train_idx = valid_indices[:train_end]
    val_idx = valid_indices[train_end:]
    
    train_ds = SequentialDataset(
        replay_buffer, 
        seq_len, 
        train_idx,
        use_actions=cfg.training.model.get('use_actions', True),
        action_dim=cfg.training.model.get('action_dim', 2),
        use_robot_state=cfg.training.model.get('use_robot_state', False),
        robot_state_dim=cfg.training.model.get('robot_state_dim', 0),
        info_ratio_per_frame=cfg.training.data.get("info_ratio_per_frame", 0.2)
    )
    val_ds = SequentialDataset(
        replay_buffer, 
        seq_len, 
        val_idx,
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
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.training.data.batch_size, 
        shuffle=shuffle_train, 
        sampler=train_sampler,
        collate_fn=sequential_collate_fn
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.data.batch_size, shuffle=False, collate_fn=sequential_collate_fn)
    
    log.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 3. Model
    model_params = SnpV2Params(
        r_dim=cfg.training.model.embed_dim,
        embed_dim=cfg.training.model.embed_dim,
        h_dim=cfg.training.model.hidden_dim,
        z_dim=cfg.training.model.latent_dim,
        encoder_hidden_dim=cfg.training.model.encoder_hidden_dim,
        prior_hidden_dim=cfg.training.model.prior_hidden_dim,
        posterior_hidden_dim=cfg.training.model.posterior_hidden_dim,
        decoder_hidden_dim=cfg.training.model.decoder_hidden_dim,
        use_actions=cfg.training.model.get('use_actions', True),
        action_dim=cfg.training.model.get('action_dim', 2),
        use_fourier_encoder=cfg.training.model.get('use_fourier_encoder', False),
        use_fourier_decoder=cfg.training.model.get('use_fourier_decoder', False),
        fourier_frequencies=cfg.training.model.get('fourier_frequencies', 128),
        fourier_scale=cfg.training.model.get('fourier_scale', 20.0),
        spatial_max=cfg.training.model.get('spatial_max', 100.0)
    )
    model = SnpV2(model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(cfg.training.optimizer.max_epochs):
        # --- TRAIN ---
        model.train()
        train_stats = {"loss": 0, "kl": 0, "mse": 0}
        beta = min(cfg.training.optimizer.beta_max, epoch / 10.0) 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # batch is a list of T steps, each being a dict with 'obs', 'action', 'done'
            inputs = batch[:-1]
            targets = batch[1:]
            batch_size = inputs[0]['action'].shape[0]
            
            state = model.init_state(batch_size, device)
            
            optimizer.zero_grad()
            seq_loss, seq_kl, seq_mse = 0, 0, 0
            
            for step_in, step_target in zip(inputs, targets):
                obs_in = step_in['obs'].to(device)
                obs_target = step_target['obs'].to(device)
                action = step_in['action'].to(device)
                done = step_in['done'].to(device)
                
                if cfg.training.model.get('use_actions', True):
                    action_in = action
                else:
                    action_in = torch.zeros_like(action)

                output = model(state, action_in, done, obs=obs_in, query=obs_target)
                
                # 1. KL Divergence
                if output.posterior is not None:
                    kl = torch.distributions.kl_divergence(output.posterior, output.prior).sum(dim=-1)
                else:
                    kl = torch.zeros(batch_size, device=device)
                
                # 2. MSE Reconstruction (since user changed decoder to return tensor)
                target_vals = obs_target.values
                if target_vals.dim() == 2:
                    target_vals = target_vals.unsqueeze(-1)
                pred_vals = output.decoded
                
                mse_per_point = (pred_vals - target_vals)**2
                
                if obs_target.mask is not None:
                    # mask is True for PADDING
                    valid_mask = obs_target.mask.float().unsqueeze(-1)
                    mse = (mse_per_point * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
                else:
                    mse = mse_per_point.mean(dim=1)
                
                # mse is (B, 1)
                
                # Masking invalid transitions
                valid_step = (1.0 - done).squeeze()
                step_loss = (beta * kl + mse.squeeze(-1)) * valid_step
                
                seq_loss += step_loss.mean()
                seq_kl += (kl * valid_step).mean()
                seq_mse += (mse.squeeze(-1) * valid_step).mean()
                
                state = output.state
                
            seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.optimizer.grad_clip)
            optimizer.step()
            
            train_stats["loss"] += seq_loss.item()
            train_stats["kl"] += seq_kl.item()
            train_stats["mse"] += seq_mse.item()
            pbar.set_postfix({'L': seq_loss.item()})
            
        # Log Train
        for k in train_stats: train_stats[k] /= len(train_loader)
        writer.add_scalar("Train/Loss", train_stats["loss"], epoch)
        writer.add_scalar("Train/KL", train_stats["kl"], epoch)
        writer.add_scalar("Train/MSE", train_stats["mse"], epoch)
        
        # --- VAL ---
        model.eval()
        val_stats = {"loss": 0, "kl": 0, "mse": 0}
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[:-1], batch[1:]
                state = model.init_state(inputs[0]['action'].shape[0], device)
                
                for step_in, step_target in zip(inputs, targets):
                    if cfg.training.model.get('use_actions', True):
                        action_in = step_in['action'].to(device)
                    else:
                        action_in = torch.zeros_like(step_in['action'].to(device))
                        
                    output = model(state, action_in, step_in['done'].to(device), 
                                   obs=step_in['obs'].to(device), query=step_target['obs'].to(device))
                    
                    kl = torch.distributions.kl_divergence(output.posterior, output.prior).sum(dim=-1) if output.posterior else torch.zeros(state.z.shape[0], device=device)
                    target_vals = step_target['obs'].to(device).values.unsqueeze(-1)
                    mse_per_point = (output.decoded - target_vals)**2
                    valid_mask = step_target['obs'].mask.float().unsqueeze(-1).to(device) if step_target['obs'].mask is not None else torch.ones_like(mse_per_point)
                    mse = (mse_per_point * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
                    
                    valid_step = (1.0 - step_in['done']).squeeze().to(device)
                    val_stats["loss"] += ((beta * kl + mse.squeeze(-1)) * valid_step).mean().item()
                    val_stats["kl"] += (kl * valid_step).mean().item()
                    val_stats["mse"] += (mse.squeeze(-1) * valid_step).mean().item()
                    state = output.state

        for k in val_stats: val_stats[k] /= len(val_loader)
        writer.add_scalar("Val/Loss", val_stats["loss"], epoch)
        writer.add_scalar("Val/KL", val_stats["kl"], epoch)
        writer.add_scalar("Val/MSE", val_stats["mse"], epoch)
        
        log.info(f"Epoch {epoch}: T-Loss={train_stats['loss']:.4f}, V-Loss={val_stats['loss']:.4f}")
        
        # Protocol Evaluation
        valid_indices_25 = val_loader.dataset.indices
        valid_indices_25 = valid_indices_25[valid_indices_25 < (datasize - 26)]
        
        if len(valid_indices_25) > 0:
            val_ds_proto = SequentialDataset(
                replay_buffer, 
                25, 
                valid_indices_25,
                use_actions=cfg.training.model.get('use_actions', True),
                action_dim=cfg.training.model.get('action_dim', 2),
                use_robot_state=cfg.training.model.get('use_robot_state', False),
                robot_state_dim=cfg.training.model.get('robot_state_dim', 0)
            )
            val_loader_proto = DataLoader(val_ds_proto, batch_size=cfg.training.data.batch_size, shuffle=False, collate_fn=sequential_collate_fn, num_workers=0)
            
            from src.utils.eval_protocol import evaluate_forecast_protocol
            proto_mse = evaluate_forecast_protocol(model, val_loader_proto, device, model_type="model_free")
            writer.add_scalar("Protocol_10_15/Dataset_MSE", proto_mse, epoch)
            log.info(f"Protocol 10-15 MSE: {proto_mse:.4f}")
        
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(model, optimizer, epoch, best_val_loss, model_params, os.path.join(ckpt_dir, "best_model.pt"))
        save_checkpoint(model, optimizer, epoch, val_stats["loss"], model_params, os.path.join(ckpt_dir, "last_model.pt"))

        if epoch % 5 == 0:
            log_visualization(model, val_loader, device, writer, epoch)
            evaluate_10_15_protocol(model, val_ds, device, writer, epoch, model_type="model_free")

def log_visualization(model, loader, device, writer, epoch):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import torchvision.transforms as transforms
    from PIL import Image
    
    model.eval()
    
    # Select random sample from entire dataset
    dataset = loader.dataset
    idx = np.random.randint(0, len(dataset))
    sample = dataset[idx]
    
    batch = collate_sequences([sample])
    inputs, targets = batch[:-1], batch[1:]
    
    batch_size = inputs[0]['action'].shape[0]
    sample_idx = 0
    time_idx = np.random.randint(0, len(inputs))
    
    state = model.init_state(batch_size, device)
    
    with torch.no_grad():
        for t, (step_in, step_target) in enumerate(zip(inputs, targets)):
            output = model(state, step_in['action'].to(device), step_in['done'].to(device), 
                           obs=step_in['obs'].to(device), query=step_target['obs'].to(device))
            state = output.state
            
            if t == time_idx:
                obs_target = step_target['obs']
                gt = obs_target.values[sample_idx].cpu().numpy()
                pr = output.decoded[sample_idx].squeeze().cpu().numpy()
                xs, ys = obs_target.xs[sample_idx].cpu().numpy(), obs_target.ys[sample_idx].cpu().numpy()
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].scatter(xs, ys, c=gt, cmap='plasma', vmin=0, vmax=1)
                axes[0].set_title(f"GT Frame: {time_idx+1} (dt=+1)")
                axes[1].scatter(xs, ys, c=pr, cmap='plasma', vmin=0, vmax=1)
                axes[1].set_title(f"Pred Frame: {time_idx+1} | Context: {time_idx}")
                
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                writer.add_image("Val/Visualization", transforms.ToTensor()(Image.open(buf)), epoch)
                plt.close()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    writer.add_image("Val/Visualization", transforms.ToTensor()(Image.open(buf)), epoch)
    plt.close()

if __name__ == "__main__":
    main()
