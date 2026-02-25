"""
train_fno_3d.py — Training for FNO-3D (true 3D spectral convolution)
======================================================================
Same sliding-window strategy as train_fno_temporal.py:
  - Slide h_ctx-frame context window across the sequence
  - Compute NLL over h_pred future targets
  - backward() after each window to bound memory (no full-sequence graph)

The only differences from train_fno_temporal.py:
  - Imports FNO3d / FNO3dConfig
  - config_name = fno_3d_train
"""

import sys
import os
sys.path.append(os.getcwd())

import random
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.model_free.fno_3d import FNO3d, FNO3dConfig
from src.models.shared.datasets import SequentialDataset, dense_sequential_collate_fn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, loss, cfg_dict, path):
    torch.save({
        'epoch':                 epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
        'loss':                  loss,
        'hyper_parameters':      cfg_dict,
    }, path)


def make_times(t_offset: int, h_ctx: int, seq_len: int,
               device, B: int) -> torch.Tensor:
    """Relative times [t_offset .. t_offset+h_ctx-1] normalised to [0,1]. → (B, h_ctx)"""
    t = torch.arange(t_offset, t_offset + h_ctx, device=device, dtype=torch.float32)
    t = t / max(seq_len - 1, 1)
    return t.unsqueeze(0).expand(B, -1)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def log_vis(model, loader, cfg, H, W, device, writer, epoch,
            rollout_steps=(1, 5, 10, 15)):
    model.eval()
    ctx_obs, trg_obs, _ = next(iter(loader))
    ctx_obs = ctx_obs.to(device)

    T      = ctx_obs.xs.shape[1]
    h_ctx  = cfg.training.model.h_ctx
    seq_len = cfg.training.data.sequence_length
    cases  = min(2, ctx_obs.xs.shape[0])
    max_step = max(rollout_steps)

    n_cols = len(rollout_steps) * 3
    fig, axes = plt.subplots(cases, n_cols, figsize=(5 * n_cols, 4 * cases))
    if cases == 1:
        axes = axes[None, :]

    frames_all = ctx_obs.values[:, :, :, 0].view(ctx_obs.xs.shape[0], T, H, W)

    with torch.no_grad():
        for b_idx in range(cases):
            seed  = frames_all[b_idx:b_idx+1, :h_ctx]    # (1, h_ctx, H, W)
            preds = model.autoregressive_forecast(
                seed, seed_t_start=0, horizon=max_step, num_samples=1)

            for j, step in enumerate(rollout_steps):
                idx     = step - 1
                mu_img  = preds[idx]['mean'].astype('float32')
                std_img = preds[idx]['std'].astype('float32')

                abs_t   = h_ctx + idx
                gt_img  = (frames_all[b_idx, abs_t].cpu().numpy()
                           if abs_t < T else np.zeros((H, W), dtype=np.float32))
                err     = np.abs(gt_img - mu_img).mean()
                col     = j * 3

                ax = axes[b_idx, col]
                sc = ax.imshow(gt_img, vmin=0, vmax=1, cmap='rainbow', origin='lower')
                ax.set_title(f"GT +{step} b={b_idx}")
                plt.colorbar(sc, ax=ax)

                ax = axes[b_idx, col+1]
                sc2 = ax.imshow(mu_img, vmin=0, vmax=1, cmap='rainbow', origin='lower')
                ax.set_title(f"μ +{step}  err={err:.3f}")
                plt.colorbar(sc2, ax=ax)

                ax = axes[b_idx, col+2]
                sc3 = ax.imshow(std_img, vmin=0, vmax=1, cmap='hot', origin='lower')
                ax.set_title(f"σ +{step}  μσ={std_img.mean():.3f}")
                plt.colorbar(sc3, ax=ax)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_t = torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1)
    writer.add_image("Val/Rollout", img_t, epoch)
    plt.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@hydra.main(version_base=None,
            config_path="../../config/training",
            config_name="fno_3d_train")
def train(cfg: DictConfig):
    print(f"Training FNO-3D — {cfg.training.experiment_name}")
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    log_dir    = os.path.join(output_dir, "logs")
    ckpt_dir   = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    writer = SummaryWriter(log_dir=log_dir)

    # ---- Data ---------------------------------------------------------------
    try:
        root_dir = Path(hydra.utils.get_original_cwd())
    except Exception:
        root_dir = Path(os.getcwd())

    data_path = root_dir / cfg.training.data.data_path
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    h_ctx   = cfg.training.model.h_ctx
    h_pred  = cfg.training.model.h_pred
    seq_len = cfg.training.data.sequence_length
    max_ep  = cfg.training.data.get("max_samples", None)

    assert seq_len > h_ctx + h_pred, \
        f"sequence_length ({seq_len}) must be > h_ctx+h_pred ({h_ctx+h_pred})"

    train_ds = SequentialDataset(
        data_path=str(data_path), sequence_length=seq_len,
        forecast_horizon=h_pred, mode='train',
        train_split=cfg.training.data.train_split,
        max_episodes=max_ep, dense=True)
    val_ds = SequentialDataset(
        data_path=str(data_path), sequence_length=seq_len,
        forecast_horizon=h_pred, mode='val',
        train_split=cfg.training.data.train_split,
        max_episodes=max_ep, dense=True)

    H, W = train_ds.H, train_ds.W
    print(f"Grid {H}×{W}  h_ctx={h_ctx}  h_pred={h_pred}  "
          f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.data.batch_size,
        shuffle=True, collate_fn=dense_sequential_collate_fn,
        num_workers=cfg.training.data.num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.data.batch_size,
        shuffle=False, collate_fn=dense_sequential_collate_fn,
        num_workers=cfg.training.data.num_workers)

    # ---- Model --------------------------------------------------------------
    mc = cfg.training.model
    fno_cfg = FNO3dConfig(
        h_ctx     = h_ctx,
        h_pred    = h_pred,
        modes_t   = mc.get("modes_t",  4),
        modes_h   = mc.get("modes_h",  8),
        modes_w   = mc.get("modes_w",  8),
        width     = mc.get("width",   32),
        n_layers  = mc.get("n_layers", 4),
        use_grid  = mc.get("use_grid", True),
        use_time  = mc.get("use_time", True),
        seq_len_ref = seq_len,
        min_std   = mc.get("min_std", 1e-4),
    )
    model = FNO3d(fno_cfg).to(device)
    print(f"FNO3d params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer ----------------------------------------------------------
    opt_cfg   = cfg.training.optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt_cfg.max_epochs, eta_min=opt_cfg.min_lr)

    def criterion(dist, gt):
        return -dist.log_prob(gt).mean()

    # ---- Training loop ------------------------------------------------------
    best_val = float('inf')

    for epoch in range(opt_cfg.max_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            ctx_obs, trg_obs, _ = batch
            ctx_obs = ctx_obs.to(device)
            trg_obs = trg_obs.to(device)

            T      = ctx_obs.xs.shape[1]
            B_size = ctx_obs.xs.shape[0]

            # (B, T, P, 1) → (B, T, H, W)
            frames  = ctx_obs.values[:, :, :, 0].view(B_size, T, H, W)
            # (B, T, P, h_pred) → (B, T, H, W, h_pred)
            targets = trg_obs.values.view(B_size, T, H, W, h_pred)

            n_win = T - h_ctx - h_pred + 1
            if n_win <= 0:
                continue

            batch_loss = 0.0
            optimizer.zero_grad()

            for t in range(h_ctx - 1, T - h_pred):
                ctx_w = frames[:, t - h_ctx + 1 : t + 1]    # (B, h_ctx, H, W)
                times  = make_times(t - h_ctx + 1, h_ctx, seq_len, device, B_size)

                dists = model(ctx_w, times)   # List[Normal] of h_pred, (B,H,W,1)

                step_loss = sum(
                    criterion(dists[h], targets[:, t, :, :, h].unsqueeze(-1))
                    for h in range(h_pred)
                ) / (h_pred * n_win)

                step_loss.backward()     # free activations immediately
                batch_loss += step_loss.item() * n_win

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            optimizer.step()

            avg = batch_loss / n_win
            train_loss += avg
            pbar.set_postfix({'nll': f"{avg:.4f}"})

        avg_train = train_loss / len(train_loader)
        scheduler.step()

        writer.add_scalar("Train/NLL", avg_train, epoch)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)

        # ---- Validation -------------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ctx_obs, trg_obs, _ = batch
                ctx_obs = ctx_obs.to(device)
                trg_obs = trg_obs.to(device)

                T      = ctx_obs.xs.shape[1]
                B_size = ctx_obs.xs.shape[0]
                frames  = ctx_obs.values[:, :, :, 0].view(B_size, T, H, W)
                targets = trg_obs.values.view(B_size, T, H, W, h_pred)

                n_win = T - h_ctx - h_pred + 1
                if n_win <= 0:
                    continue

                batch_nll = 0.0
                for t in range(h_ctx - 1, T - h_pred):
                    ctx_w  = frames[:, t - h_ctx + 1 : t + 1]
                    times  = make_times(t - h_ctx + 1, h_ctx, seq_len, device, B_size)
                    dists  = model(ctx_w, times)
                    for h in range(h_pred):
                        gt_h = targets[:, t, :, :, h].unsqueeze(-1)
                        batch_nll += criterion(dists[h], gt_h).item()
                val_loss += batch_nll / (h_pred * n_win)

        avg_val = val_loss / len(val_loader)
        writer.add_scalar("Val/NLL", avg_val, epoch)
        log.info(f"Epoch {epoch}: Train={avg_train:.4f}  Val={avg_val:.4f}")

        if epoch % cfg.training.visualizer.visualize_every == 0:
            log_vis(model, val_loader, cfg, H, W, device, writer, epoch)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        save_checkpoint(model, optimizer, epoch, avg_val, cfg_dict,
                        os.path.join(ckpt_dir, "last_model.pt"))
        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(model, optimizer, epoch, avg_val, cfg_dict,
                            os.path.join(ckpt_dir, "best_model.pt"))

    writer.close()


if __name__ == "__main__":
    train()
