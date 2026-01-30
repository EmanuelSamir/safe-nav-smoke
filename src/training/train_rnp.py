import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
from src.models.shared.datasets import SequentialDataset, collate_sequences as sequential_collate_fn
from envs.replay_buffer import GenericReplayBuffer
from pathlib import Path
from src.utils.eval_protocol import evaluate_10_15_protocol

log = logging.getLogger(__name__)

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
    
    # Select random sample from entire dataset
    dataset = loader.dataset
    idx = np.random.randint(0, len(dataset))
    sample = dataset[idx] # Returns a sequence list
    
    # Collate into batch of 1
    batch = sequential_collate_fn([sample])
    
    inputs = batch[:-1]
    targets = batch[1:]
    
    batch_size = inputs[0]['action'].shape[0] # Should be 1
    
    # Select random sample (always 0 here) and time step
    sample_idx = 0
    time_idx = np.random.randint(0, len(inputs))
    
    # RNP State: (h, c)
    # RNP State: (h, c)
    # RNPConfig uses h_dim, not hidden_dim
    h = torch.zeros(model.config.lstm_layers, batch_size, model.config.h_dim).to(device)
    c = torch.zeros(model.config.lstm_layers, batch_size, model.config.h_dim).to(device)
    state = (h, c)
    
    with torch.no_grad():
        for t, (step_in, step_target) in enumerate(zip(inputs, targets)):
            obs_in = step_in['obs'].to(device)
            obs_target = step_target['obs'].to(device)
            act_in = step_in['action'].to(device)
            
            output = model(state, act_in, obs=obs_in, query=obs_target) # RNP forward uses 'query' not 'query_obs'
            next_state = output.state
            pred_dist = output.prediction # RNPOutput has 'prediction'
            state = next_state
            
            if t == time_idx: # Random step
               gt = obs_target.values[sample_idx].cpu().numpy()
               pr = pred_dist.mean[sample_idx].squeeze().cpu().numpy()
               xs = obs_target.xs[sample_idx].cpu().numpy()
               ys = obs_target.ys[sample_idx].cpu().numpy()
               
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
        
    buffer_path = root_dir / cfg.training.data.buffer_path
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
    
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    
    train_dataset = SequentialDataset(
        replay_buffer, 
        seq_len, 
        train_idx,
        use_actions=cfg.training.model.get('use_actions', True),
        action_dim=cfg.training.model.get('action_dim', 2),
        use_robot_state=cfg.training.model.get('use_robot_state', False),
        robot_state_dim=cfg.training.model.get('robot_state_dim', 0),
        info_ratio_per_frame=cfg.training.data.get("info_ratio_per_frame", 0.2)
    )
    
    val_dataset = SequentialDataset(
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
    train_sampler = None
    shuffle_train = True
    epoch_sample_ratio = cfg.training.data.get("epoch_sample_ratio", None)
    
    # Backward compatibility or direct check
    if epoch_sample_ratio is None:
         # Check if old key exists just in case or default
         samples_per_epoch = cfg.training.data.get("samples_per_epoch", None)
         if samples_per_epoch is not None:
              num_samples = min(samples_per_epoch, len(train_dataset))
              train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False, num_samples=num_samples)
              shuffle_train = False
    elif epoch_sample_ratio < 1.0:
         num_samples = int(len(train_dataset) * epoch_sample_ratio)
         num_samples = max(1, num_samples) # At least one
         train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False, num_samples=num_samples)
         shuffle_train = False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.data.batch_size, 
        shuffle=shuffle_train,
        sampler=train_sampler,
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
        num_layers=model_cfg.num_layers,
        lstm_layers=model_cfg.lstm_layers,
        use_fourier_encoder=model_cfg.get('use_fourier_encoder', False),
        use_fourier_decoder=model_cfg.get('use_fourier_decoder', False),
        fourier_frequencies=model_cfg.get('fourier_frequencies', 128),
        fourier_scale=model_cfg.get('fourier_scale', 20.0),
        spatial_max=model_cfg.get('spatial_max', 100.0)
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
            inputs = batch[:-1]
            targets = batch[1:]
            batch_size = inputs[0]['action'].shape[0]
            
            # Init LSTM State (tuples)
            h = torch.zeros(params.lstm_layers, batch_size, params.h_dim).to(device)
            c = torch.zeros(params.lstm_layers, batch_size, params.h_dim).to(device)
            state = (h, c)
            
            optimizer.zero_grad()
            seq_loss = 0
            seq_ll = 0
            
            for step_in, step_target in zip(inputs, targets):
                obs_in = step_in['obs'].to(device)
                obs_target = step_target['obs'].to(device)
                done_in = step_in['done'].to(device)
                
                if cfg.training.model.get('use_actions', True):
                    act_in = step_in['action'].to(device) 
                else: 
                    act_in = torch.zeros_like(step_in['action'].to(device))

                output_step = model(state, act_in, obs=obs_in, query=obs_target)
                next_state = output_step.state
                pred_dist = output_step.prediction
                
                gt_val = obs_target.values
                if gt_val.dim() == 2:
                    gt_val = gt_val.unsqueeze(-1)
                
                if obs_target.mask is not None:
                     point_mask = obs_target.mask.float() 
                     ll_per_point = pred_dist.log_prob(gt_val).squeeze(-1)
                     ll = (ll_per_point * point_mask).sum(dim=-1) / torch.clamp(point_mask.sum(dim=-1), min=1.0)
                else:
                     ll = pred_dist.log_prob(gt_val).mean(dim=-1).mean(dim=-1)

                # print(np.unique(obs_target.mask.cpu().numpy(), return_counts=True))
                
                valid = (1.0 - done_in).squeeze()
                step_loss = -ll * valid
                seq_loss += step_loss.mean()
                seq_ll += (ll * valid).mean()
                
                state = (next_state[0] * (1.0 - done_in).view(1, -1, 1), 
                         next_state[1] * (1.0 - done_in).view(1, -1, 1))
            
            seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            optimizer.step()
            
            train_loss += seq_loss.item()
            train_ll += seq_ll.item()
            pbar.set_postfix({'loss': seq_loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ll = train_ll / len(train_loader)
        scheduler.step()
        
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        
        # Validation
        model.eval()
        val_ll = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[:-1]
                targets = batch[1:]
                batch_size = inputs[0]['action'].shape[0]
                h = torch.zeros(params.lstm_layers, batch_size, params.h_dim).to(device)
                c = torch.zeros(params.lstm_layers, batch_size, params.h_dim).to(device)
                state = (h, c)
                
                batch_ll = 0
                for step_in, step_target in zip(inputs, targets):
                    obs_in = step_in['obs'].to(device)
                    obs_target = step_target['obs'].to(device)
                    done_in = step_in['done'].to(device)
                    
                    if cfg.training.model.get('use_actions', True):
                        act_in = step_in['action'].to(device)
                    else:
                        act_in = torch.zeros_like(step_in['action'].to(device))
                        
                    output_step = model(state, act_in, obs=obs_in, query=obs_target)
                    next_state = output_step.state
                    pred_dist = output_step.prediction
                    
                    gt_val = obs_target.values
                    if gt_val.dim() == 2:
                        gt_val = gt_val.unsqueeze(-1)
                    if obs_target.mask is not None:
                         point_mask = (~obs_target.mask).float() 
                         ll_per_point = pred_dist.log_prob(gt_val).squeeze(-1)
                         ll = (ll_per_point * point_mask).sum(dim=-1) / torch.clamp(point_mask.sum(dim=-1), min=1.0)
                    else:
                         ll = pred_dist.log_prob(gt_val).mean(dim=-1).mean(dim=-1)
                    
                    valid = (1.0 - done_in).squeeze()
                    batch_ll += (ll * valid).mean()
                    state = (next_state[0] * (1.0 - done_in).view(1, -1, 1), 
                             next_state[1] * (1.0 - done_in).view(1, -1, 1))
                
                val_ll += batch_ll.item()

        avg_val_ll = val_ll / len(val_loader)
        writer.add_scalar("Val/LL", avg_val_ll, epoch)
        log.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val LL={avg_val_ll:.4f}")
        
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
        
        # Visualization
        if epoch % 1 == 0:
             log_visualization_to_tensorboard(model, val_loader, device, writer, epoch)
             evaluate_10_15_protocol(model, val_dataset, device, writer, epoch, model_type="model_free")
        
        # Checkpoint
        save_checkpoint(model, optimizer, epoch, -avg_val_ll, params, os.path.join(ckpt_dir, "last_model.pt"))
        if avg_val_ll > best_val_ll:
            best_val_ll = avg_val_ll
            save_checkpoint(model, optimizer, epoch, -avg_val_ll, params, os.path.join(ckpt_dir, "best_model.pt"))

    writer.close()

if __name__ == "__main__":
    train()
