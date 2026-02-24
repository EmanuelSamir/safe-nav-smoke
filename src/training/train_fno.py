import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
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
from pathlib import Path

from src.models.model_free.fno import FNO2d, FNOConfig
from src.models.shared.datasets import SequentialDataset, dense_sequential_collate_fn
from src.models.shared.observations import slice_obs
from src.models.shared.schedulers import LinearBetaScheduler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, loss, cfg_dict, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'hyper_parameters': cfg_dict,
    }, path)


def obs_to_grid(obs_step, H: int, W: int):
    """Dense Obs slice -> (B, H, W, 1): squeeze time dim, reshape flat pixel dim."""
    v = obs_step.values[:, 0, :, 0]   # (B, P)
    return v.view(v.shape[0], H, W, 1)


def obs_to_grid_2ch(obs_step, H: int, W: int, std_grid=None):
    """
    Dense Obs slice -> (B, H, W, 2): channel 0 = mean (GT), channel 1 = std.
    std_grid : (B, H, W, 1) tensor or None.  Defaults to zeros (teacher-force).
    """
    mean = obs_to_grid(obs_step, H, W)   # (B, H, W, 1)
    std  = std_grid if std_grid is not None else torch.zeros_like(mean)
    return torch.cat([mean, std], dim=-1)  # (B, H, W, 2)


def values_at_t(trg, t: int, h: int, H: int, W: int):
    """
    Extract ground truth for future sub-step h at sequence position t.
    trg.values : (B, T, P, forecast_horizon)  — last dim is H for multistep, 1 for H=1.
    Returns     : (B, H_grid, W_grid, 1)
    """
    B = trg.values.shape[0]
    return trg.values[:, t, :, h].view(B, H, W, 1)


def nll_loss(dist, gt: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood averaged over all pixels and batch.
    dist : Normal(mu, sigma)  shapes (B, H, W, 1)
    gt   : (B, H, W, 1)
    """
    return -dist.log_prob(gt).mean()


def grid_to_obs_values(grid, obs_template):
    """
    Flatten a (B, H, W, 1) grid back to (B, 1, P, 1) to match Obs layout.
    obs_template is used only to borrow xs/ys/mask (not mutated).
    """
    B = grid.shape[0]
    flat = grid.view(B, -1)            # (B, P)
    return flat.unsqueeze(1).unsqueeze(-1)  # (B, 1, P, 1)


def log_visualization_to_tensorboard(model, loader, H, W, device, writer, epoch,
                                     rollout_steps=(1, 5, 10, 15)):
    """
    Visualises a true autoregressive rollout for the val set.
    For each of 2 batch items:  seed = frame t=0,  run model.autoregressive_forecast
    for max(rollout_steps) steps, then compare predicted μ, σ vs GT at steps 5/10/15.
    """
    model.eval()
    ctx, trg, _ = next(iter(loader))
    ctx = ctx.to(device)
    trg = trg.to(device)

    cases     = min(2, ctx.xs.shape[0])
    max_step  = max(rollout_steps)
    n_cols    = len(rollout_steps) * 3     # GT | μ | σ

    fig, axes = plt.subplots(cases, n_cols, figsize=(5 * n_cols, 4 * cases))
    if cases == 1:
        axes = axes[None, :]

    with torch.no_grad():
        use_unc = model.use_uncertainty_input
        for b_idx in range(cases):
            # Seed: first frame (1, H, W, 1)
            seed = ctx.values[b_idx:b_idx+1, 0, :, 0].view(1, H, W, 1)

            # If model expects 2-channel input: pad seed with zero std
            x_t = (torch.cat([seed, torch.zeros_like(seed)], dim=-1)
                   if use_unc else seed)

            # Roll out: collect mu/sigma directly from forward()
            step_dists = {}   # step -> (mu_np, std_np)
            for s in range(1, max_step + 1):
                dists = model(x_t)   # List[Normal], params shape (1, H, W, 1)
                d = dists[0]
                mu_np  = d.mean[0, :, :, 0].cpu().float().numpy()    # (H, W)
                std_np = d.stddev[0, :, :, 0].cpu().float().numpy()  # (H, W)
                if s in rollout_steps:
                    step_dists[s] = (mu_np, std_np)
                # Propagate: uncertainty mode passes (mean, sigma); else just mean
                if use_unc:
                    x_t = torch.cat([d.mean, d.stddev], dim=-1)
                else:
                    x_t = d.mean

            for j, step in enumerate(rollout_steps):
                if step not in step_dists:
                    continue
                mu_img, std_img = step_dists[step]
                idx = step - 1

                # GT: trg.values[b, idx, :, 0] = frame at window-start + step
                if idx < trg.values.shape[1]:
                    gt_img = trg.values[b_idx, idx, :, 0].cpu().float().numpy().reshape(H, W)
                else:
                    gt_img = np.zeros((H, W), dtype=np.float32)

                err_img = np.abs(gt_img - mu_img)
                col = j * 3
                vmin, vmax = 0.0, 1.0

                ax = axes[b_idx, col]
                sc = ax.imshow(gt_img, vmin=vmin, vmax=vmax, cmap='rainbow', origin='lower')
                ax.set_title(f"GT +{step}  b={b_idx}")
                plt.colorbar(sc, ax=ax)

                ax = axes[b_idx, col + 1]
                sc2 = ax.imshow(mu_img, vmin=vmin, vmax=vmax, cmap='rainbow', origin='lower')
                ax.set_title(f"μ +{step}  err={err_img.mean():.3f}")
                plt.colorbar(sc2, ax=ax)

                ax = axes[b_idx, col + 2]
                sc3 = ax.imshow(std_img, vmin=0, vmax=1.0, cmap='hot', origin='lower')
                ax.set_title(f"σ +{step}  meanσ={std_img.mean():.3f}")
                plt.colorbar(sc3, ax=ax)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_tensor = torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1)
    writer.add_image("Val/Rollout_5_10_15", img_tensor, epoch)
    plt.close()



# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None,
            config_path="../../config/training",
            config_name="fno_train")
def train(cfg: DictConfig):

    # 1. Setup ---------------------------------------------------------------
    print(f"Training FNO2d with config: {cfg.training.experiment_name}")
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

    log_dir  = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # 2. Data ----------------------------------------------------------------
    try:
        root_dir = Path(hydra.utils.get_original_cwd())
    except Exception:
        root_dir = Path(os.getcwd())

    data_path = root_dir / cfg.training.data.data_path
    if not data_path.exists():
        log.error(f"Data not found at: {data_path}")
        return

    seq_len          = cfg.training.data.sequence_length
    max_episodes     = cfg.training.data.get("max_samples", None)
    forecast_horizon = cfg.training.model.get("forecast_horizon", 1)

    train_dataset = SequentialDataset(
        data_path=str(data_path),
        sequence_length=seq_len,
        forecast_horizon=forecast_horizon,
        mode='train',
        train_split=cfg.training.data.train_split,
        max_episodes=max_episodes,
        dense=True,
    )
    val_dataset = SequentialDataset(
        data_path=str(data_path),
        sequence_length=seq_len,
        forecast_horizon=forecast_horizon,
        mode='val',
        train_split=cfg.training.data.train_split,
        max_episodes=max_episodes,
        dense=True,
    )

    H, W = train_dataset.H, train_dataset.W
    print(f"Grid: {H}×{W}  H={forecast_horizon}  "
          f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.data.batch_size,
        shuffle=True,
        collate_fn=dense_sequential_collate_fn,
        num_workers=cfg.training.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.data.batch_size,
        shuffle=True,
        collate_fn=dense_sequential_collate_fn,
        num_workers=cfg.training.data.num_workers,
    )

    # 3. Model ---------------------------------------------------------------
    model_cfg = cfg.training.model
    fno_cfg = FNOConfig(
        modes1=model_cfg.modes1,
        modes2=model_cfg.modes2,
        width=model_cfg.width,
        use_grid=model_cfg.get("use_grid", True),
        n_layers=model_cfg.get("n_layers", 4),
        min_std=model_cfg.get("min_std", 1e-4),
        forecast_horizon=forecast_horizon,
        use_uncertainty_input=model_cfg.get("use_uncertainty_input", False),
    )
    model = FNO2d(fno_cfg).to(device)
    use_unc = fno_cfg.use_uncertainty_input
    print(f"FNO2d params: {sum(p.numel() for p in model.parameters()):,}  "
          f"H={forecast_horizon}  use_unc={use_unc}")

    # 4. Optimizer -----------------------------------------------------------
    opt_cfg   = cfg.training.optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt_cfg.max_epochs, eta_min=opt_cfg.min_lr)

    sampling_scheduler = LinearBetaScheduler(
        beta_start=cfg.training.sampling.beta_start,
        beta_end=cfg.training.sampling.beta_end,
        num_steps=cfg.training.sampling.warmup_epochs,
    )

    # NLL loss: -log p(y | x)
    def criterion(dist, gt):
        return -dist.log_prob(gt).mean()

    # 5. Training loop -------------------------------------------------------
    best_val_mse = float('inf')

    for epoch in range(opt_cfg.max_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            ctx, trg, _ = batch
            ctx = ctx.to(device)
            trg = trg.to(device)
            # ctx.values: (B, T, P, 1)
            # trg.values: (B, T, P, forecast_horizon)

            B_size = ctx.xs.shape[0]
            T_eff  = ctx.xs.shape[1]

            optimizer.zero_grad()
            loss_total  = 0.0
            ratio_force = 0.0

            # Seed from first frame
            ctx_frame_0 = ctx.values[:, 0, :, 0].view(B_size, H, W, 1)  # (B, H, W, 1)
            x_t = (torch.cat([ctx_frame_0, torch.zeros_like(ctx_frame_0)], dim=-1)
                   if use_unc else ctx_frame_0)

            for t in range(T_eff):
                dists = model(x_t)   # List[Normal] length forecast_horizon

                # Loss over all H sub-steps
                step_loss = 0.0
                for h in range(forecast_horizon):
                    gt_h = values_at_t(trg, t, h, H, W)   # (B, H, W, 1)
                    step_loss += criterion(dists[h], gt_h)
                step_loss /= forecast_horizon
                loss_total += step_loss

                # Scheduled sampling — use dists[0] (prediccion t+1) as next input
                if t + 1 < T_eff:
                    beta = sampling_scheduler.update(epoch)
                    if random.random() < beta:
                        if use_unc:
                            x_t = torch.cat(
                                [dists[0].mean, dists[0].stddev], dim=-1).detach()
                        elif cfg.training.sampling.get("use_mean", True):
                            x_t = dists[0].mean.detach()
                        else:
                            x_t = dists[0].sample().detach()
                        ratio_force += 1.0
                    else:
                        # Teacher-force: next real frame with zero std
                        ctx_frame = ctx.values[:, t+1, :, 0].view(B_size, H, W, 1)
                        x_t = (torch.cat([ctx_frame, torch.zeros_like(ctx_frame)], dim=-1)
                               if use_unc else ctx_frame)

            ratio_force /= max(T_eff - 1, 1)
            loss = loss_total / T_eff
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'mse': f"{loss.item():.5f}",
                              'force': f"{ratio_force:.2f}"})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        writer.add_scalar("Train/NLL", avg_train_loss, epoch)
        writer.add_scalar("Train/LR",
                          optimizer.param_groups[0]['lr'], epoch)

        # 6. Validation ------------------------------------------------------
        model.eval()
        val_mse = 0.0

        with torch.no_grad():
            for batch in val_loader:
                ctx, trg, _ = batch
                ctx = ctx.to(device)
                trg = trg.to(device)

                T_eff    = ctx.xs.shape[1]
                B_size   = ctx.xs.shape[0]
                batch_nll = 0.0

                ctx_frame_0 = ctx.values[:, 0, :, 0].view(B_size, H, W, 1)
                x_t = (torch.cat([ctx_frame_0, torch.zeros_like(ctx_frame_0)], dim=-1)
                       if use_unc else ctx_frame_0)

                for t in range(T_eff):
                    dists = model(x_t)
                    step_nll = 0.0
                    for h in range(forecast_horizon):
                        gt_h = values_at_t(trg, t, h, H, W)
                        step_nll += criterion(dists[h], gt_h).item()
                    batch_nll += step_nll / forecast_horizon

                    if t + 1 < T_eff:
                        ctx_frame = ctx.values[:, t+1, :, 0].view(B_size, H, W, 1)
                        x_t = (torch.cat([ctx_frame, torch.zeros_like(ctx_frame)], dim=-1)
                               if use_unc else ctx_frame)

                val_mse += batch_nll / T_eff

        avg_val_mse = val_mse / len(val_loader)
        writer.add_scalar("Val/NLL", avg_val_mse, epoch)
        log.info(f"Epoch {epoch}: Train NLL={avg_train_loss:.5f}  "
                 f"Val NLL={avg_val_mse:.5f}")

        # 7. Visualisation ---------------------------------------------------
        if epoch % cfg.training.visualizer.visualize_every == 0:
            log_visualization_to_tensorboard(
                model, val_loader, H, W, device, writer, epoch)

        # 8. Checkpoints -----------------------------------------------------
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        save_checkpoint(model, optimizer, epoch, avg_val_mse, cfg_dict,
                        os.path.join(ckpt_dir, "last_model.pt"))
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            save_checkpoint(model, optimizer, epoch, avg_val_mse, cfg_dict,
                            os.path.join(ckpt_dir, "best_model.pt"))

    writer.close()


if __name__ == "__main__":
    train()
