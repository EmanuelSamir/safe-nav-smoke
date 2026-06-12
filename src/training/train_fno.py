import os
import sys

sys.path.append(os.getcwd())

import logging
import random
from io import BytesIO
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from src.models.fno import FNO, FNOConfig
from src.models.shared.datasets import SequentialDataset, dense_sequential_collate_fn

log = logging.getLogger(__name__)


class TrainingDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_path: str
    batch_size: int = 8
    train_split: float = 0.9
    max_samples: Optional[int] = None
    sequence_length: int = 30
    num_workers: int = 0
    downsample_factor: int = 1


class TrainingLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "nll"
    beta: Optional[float] = None
    m_samples: int = 3
    normalize_l2: bool = True


class TrainingOptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float = 1e-3
    min_lr: float = 1e-4
    max_epochs: int = 250
    grad_clip: float = 1.0


class TrainingCheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    save_top_k: int = 3
    monitor: str = "val_nll"
    mode: str = "min"


class TrainingVisualizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visualize_every: int = 10


class FNOTrainingConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_name: str
    seed: int = 42
    data: TrainingDataConfig
    model: FNOConfig
    loss: TrainingLossConfig
    optimizer: TrainingOptimizerConfig
    checkpoint: TrainingCheckpointConfig
    visualizer: TrainingVisualizerConfig


class FNOTrainingGlobalSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    training: FNOTrainingConfigSchema


def save_checkpoint(model, optimizer, epoch, loss, cfg_dict, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "hyper_parameters": cfg_dict,
        },
        path,
    )


def make_times(t_offset: int, h_ctx: int, seq_len: int, device, B: int) -> torch.Tensor:
    """Relative times [t_offset .. t_offset+h_ctx-1] normalised to [0,1]. → (B, h_ctx)"""
    t = torch.arange(t_offset, t_offset + h_ctx, device=device, dtype=torch.float32)
    t = t / max(seq_len - 1, 1)
    return t.unsqueeze(0).expand(B, -1)


def clip_grad_norm_(parameters, max_norm):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    total_norm = 0.0
    for p in parameters:
        g = p.grad.detach()
        if g.is_complex():
            param_norm = torch.sum(g.real ** 2 + g.imag ** 2)
        else:
            param_norm = torch.sum(g ** 2)
        total_norm += param_norm
    total_norm = total_norm.sqrt()

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    return total_norm


def log_vis(model, loader, t_cfg: FNOTrainingConfigSchema, H, W, x_size, y_size, device, writer, epoch, rollout_steps=(1, 5, 10, 15)):
    model.eval()
    ctx_obs, trg_obs, _ = next(iter(loader))ma
    ctx_obs = ctx_obs.to(device)

    T = ctx_obs.xs.shape[1]
    h_ctx = t_cfg.model.h_ctx
    seq_len = t_cfg.data.sequence_length
    cases = min(2, ctx_obs.xs.shape[0])
    max_step = max(rollout_steps)

    n_cols = len(rollout_steps) * 3
    fig, axes = plt.subplots(cases, n_cols, figsize=(5 * n_cols, 4 * cases))
    if cases == 1:
        axes = axes[None, :]

    frames_all = ctx_obs.values[:, :, :, 0].view(ctx_obs.xs.shape[0], T, H, W)

    with torch.no_grad():
        for b_idx in range(cases):
            seed = frames_all[b_idx : b_idx + 1, :h_ctx]  # (1, h_ctx, H, W)
            preds = model.autoregressive_forecast(
                seed, seed_t_start=0, horizon=max_step, num_samples=1, mode="mean"
            )

            for j, step in enumerate(rollout_steps):
                idx = step - 1
                if idx >= len(preds):
                    break
                mu_img = preds[idx]["mean"][0].astype("float32")
                std_img = preds[idx]["std"][0].astype("float32")

                abs_t = h_ctx + idx
                gt_img = (
                    frames_all[b_idx, abs_t].cpu().numpy()
                    if abs_t < T
                    else np.zeros((H, W), dtype=np.float32)
                )
                err = np.abs(gt_img - mu_img).mean()
                col = j * 3

                extent = [0, x_size, 0, y_size]

                ax = axes[b_idx, col]
                sc = ax.imshow(gt_img, vmin=0, vmax=1, cmap="rainbow", origin="lower",
                               extent=extent, aspect="auto")
                ax.set_title(f"GT +{step} b={b_idx}")
                plt.colorbar(sc, ax=ax)

                ax = axes[b_idx, col + 1]
                sc2 = ax.imshow(mu_img, vmin=0, vmax=1, cmap="rainbow", origin="lower",
                                extent=extent, aspect="auto")
                ax.set_title(f"μ +{step}  err={err:.3f}")
                plt.colorbar(sc2, ax=ax)

                ax = axes[b_idx, col + 2]
                sc3 = ax.imshow(std_img, vmin=0, vmax=1, cmap="hot", origin="lower",
                                extent=extent, aspect="auto")
                ax.set_title(f"σ +{step}  μσ={std_img.mean():.3f}")
                plt.colorbar(sc3, ax=ax)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_t = torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1)
    writer.add_image("Val/Rollout", img_t, epoch)
    plt.close()


@hydra.main(version_base=None, config_path="../../configs/training", config_name="fno")
def train(cfg: DictConfig):
    # Pydantic configuration validation
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    if "training" in cfg_container and "model" in cfg_container["training"]:
        # Map sequence length to seq_len_ref
        cfg_container["training"]["model"]["seq_len_ref"] = cfg_container["training"]["data"]["sequence_length"]

    global_cfg = FNOTrainingGlobalSchema.model_validate(cfg_container)
    t_cfg = global_cfg.training

    print(f"Training FNO — {t_cfg.experiment_name}")
    torch.manual_seed(t_cfg.seed)
    np.random.seed(t_cfg.seed)
    random.seed(t_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    from hydra.core.hydra_config import HydraConfig

    output_dir = HydraConfig.get().runtime.output_dir
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    writer = SummaryWriter(log_dir=log_dir)

    # Data
    try:
        root_dir = Path(hydra.utils.get_original_cwd())
    except Exception:
        root_dir = Path(os.getcwd())

    data_path = root_dir / t_cfg.data.data_path
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    h_ctx = t_cfg.model.h_ctx
    h_pred = t_cfg.model.h_pred
    seq_len = t_cfg.data.sequence_length
    max_ep = t_cfg.data.max_samples

    assert seq_len > h_ctx + h_pred, (
        f"sequence_length ({seq_len}) must be > h_ctx+h_pred ({h_ctx + h_pred})"
    )

    train_ds = SequentialDataset(
        data_path=str(data_path),
        sequence_length=seq_len,
        forecast_horizon=h_pred,
        mode="train",
        train_split=t_cfg.data.train_split,
        max_episodes=max_ep,
        dense=True,
        downsample_factor=t_cfg.data.downsample_factor,
    )
    val_ds = SequentialDataset(
        data_path=str(data_path),
        sequence_length=seq_len,
        forecast_horizon=h_pred,
        mode="val",
        train_split=t_cfg.data.train_split,
        max_episodes=max_ep,
        dense=True,
        downsample_factor=t_cfg.data.downsample_factor,
    )

    H, W = train_ds.H, train_ds.W
    print(f"Grid {H}×{W}  h_ctx={h_ctx}  h_pred={h_pred}  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg.data.batch_size,
        shuffle=True,
        collate_fn=dense_sequential_collate_fn,
        num_workers=t_cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg.data.batch_size,
        shuffle=False,
        collate_fn=dense_sequential_collate_fn,
        num_workers=t_cfg.data.num_workers,
    )

    # Model
    fno_cfg = t_cfg.model
    model = FNO(fno_cfg).to(device)
    print(f"FNO params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    opt_cfg = t_cfg.optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt_cfg.max_epochs, eta_min=opt_cfg.min_lr
    )

    loss_cfg = t_cfg.loss
    print(f"Loss configuration: name={loss_cfg.name}, beta={loss_cfg.beta}, m_samples={loss_cfg.m_samples}, normalize_l2={loss_cfg.normalize_l2}")

    def beta_nll_loss(dist, gt, beta):
        var = dist.variance
        nll = 0.5 * (((dist.mean - gt) ** 2) / var + torch.log(var))
        weight = var.detach() ** beta
        loss = nll * weight
        return loss.mean()

    def energy_score_loss(dist, gt, m_samples, normalize_l2):
        # Generate M samples using rsample (reparameterization trick)
        samples = dist.rsample(torch.Size([m_samples]))  # (M, B, H, W, 1)

        # Term 1: 1/M * sum_{j=1}^M ||u_i^j - u_i||_H
        diff_gt = samples - gt.unsqueeze(0)  # (M, B, H, W, 1)
        if normalize_l2:
            # Root-mean-squared-difference over grid dimensions (-3, -2, -1)
            norm_gt = torch.sqrt(torch.mean(diff_gt ** 2, dim=(-3, -2, -1)) + 1e-8)  # (M, B)
        else:
            # Raw L2 norm over grid dimensions
            norm_gt = torch.sqrt(torch.sum(diff_gt ** 2, dim=(-3, -2, -1)) + 1e-8)  # (M, B)
        term1 = norm_gt.mean(dim=0)  # (B,)

        # Term 2: 1/(2*M*(M-1)) * sum_{j=1}^M sum_{h=1}^M ||u_i^j - u_i^h||_H
        samples1 = samples.unsqueeze(1)  # (M, 1, B, H, W, 1)
        samples2 = samples.unsqueeze(0)  # (1, M, B, H, W, 1)
        diff_pairwise = samples1 - samples2  # (M, M, B, H, W, 1)
        if normalize_l2:
            norm_pairwise = torch.sqrt(torch.mean(diff_pairwise ** 2, dim=(-3, -2, -1)) + 1e-8)  # (M, M, B)
        else:
            norm_pairwise = torch.sqrt(torch.sum(diff_pairwise ** 2, dim=(-3, -2, -1)) + 1e-8)  # (M, M, B)
        term2 = norm_pairwise.sum(dim=(0, 1)) / (2.0 * m_samples * (m_samples - 1))  # (B,)

        loss = term1 - term2
        return loss.mean()

    def criterion(dist, gt):
        if loss_cfg.name == "energy_score":
            return energy_score_loss(dist, gt, m_samples=loss_cfg.m_samples, normalize_l2=loss_cfg.normalize_l2)
        elif loss_cfg.name == "nll":
            if loss_cfg.beta is not None:
                return beta_nll_loss(dist, gt, beta=loss_cfg.beta)
            return -dist.log_prob(gt).mean()
        else:
            raise ValueError(f"Unknown loss function: {loss_cfg.name}")

    # Training loop
    best_val = float("inf")

    for epoch in range(opt_cfg.max_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            ctx_obs, trg_obs, _ = batch
            ctx_obs = ctx_obs.to(device)
            trg_obs = trg_obs.to(device)

            T = ctx_obs.xs.shape[1]
            B_size = ctx_obs.xs.shape[0]

            # (B, T, P, 1) → (B, T, H, W)
            frames = ctx_obs.values[:, :, :, 0].view(B_size, T, H, W)
            # (B, T, P, h_pred) → (B, T, H, W, h_pred)
            targets = trg_obs.values.view(B_size, T, H, W, h_pred)

            n_win = T - h_ctx - h_pred + 1
            if n_win <= 0:
                continue

            batch_loss = 0.0
            optimizer.zero_grad()

            for t in range(h_ctx - 1, T - h_pred):
                ctx_w = frames[:, t - h_ctx + 1 : t + 1]  # (B, h_ctx, H, W)
                times = make_times(t - h_ctx + 1, h_ctx, seq_len, device, B_size)

                dists = model(ctx_w, times)  # List[Normal] of h_pred, (B,H,W,1)

                step_loss = sum(
                    criterion(dists[h], targets[:, t, :, :, h].unsqueeze(-1)) for h in range(h_pred)
                ) / (h_pred * n_win)

                step_loss.backward()
                batch_loss += step_loss.item() * n_win

            clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            optimizer.step()

            avg = batch_loss / n_win
            train_loss += avg
            pbar.set_postfix({loss_cfg.name: f"{avg:.4f}"})

        avg_train = train_loss / len(train_loader)
        scheduler.step()

        writer.add_scalar("Train/Loss", avg_train, epoch)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_nll = 0.0
        val_es = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ctx_obs, trg_obs, _ = batch
                ctx_obs = ctx_obs.to(device)
                trg_obs = trg_obs.to(device)

                T = ctx_obs.xs.shape[1]
                B_size = ctx_obs.xs.shape[0]
                frames = ctx_obs.values[:, :, :, 0].view(B_size, T, H, W)
                targets = trg_obs.values.view(B_size, T, H, W, h_pred)

                n_win = T - h_ctx - h_pred + 1
                if n_win <= 0:
                    continue

                batch_loss = 0.0
                batch_nll = 0.0
                batch_es = 0.0
                for t in range(h_ctx - 1, T - h_pred):
                    ctx_w = frames[:, t - h_ctx + 1 : t + 1]
                    times = make_times(t - h_ctx + 1, h_ctx, seq_len, device, B_size)
                    dists = model(ctx_w, times)
                    for h in range(h_pred):
                        gt_h = targets[:, t, :, :, h].unsqueeze(-1)
                        batch_loss += criterion(dists[h], gt_h).item()
                        batch_nll += -dists[h].log_prob(gt_h).mean().item()
                        batch_es += energy_score_loss(dists[h], gt_h, m_samples=loss_cfg.m_samples, normalize_l2=loss_cfg.normalize_l2).item()
                val_loss += batch_loss / (h_pred * n_win)
                val_nll += batch_nll / (h_pred * n_win)
                val_es += batch_es / (h_pred * n_win)

        avg_val = val_loss / len(val_loader)
        avg_val_nll = val_nll / len(val_loader)
        avg_val_es = val_es / len(val_loader)

        writer.add_scalar("Val/Loss", avg_val, epoch)
        writer.add_scalar("Val/NLL", avg_val_nll, epoch)
        writer.add_scalar("Val/EnergyScore", avg_val_es, epoch)
        log.info(f"Epoch {epoch}: Train={avg_train:.4f}  Val={avg_val:.4f}  Val_NLL={avg_val_nll:.4f}  Val_ES={avg_val_es:.4f}")

        if epoch % t_cfg.visualizer.visualize_every == 0:
            log_vis(model, val_loader, t_cfg, H, W, train_ds.x_size, train_ds.y_size, device, writer, epoch)

        cfg_dict = t_cfg.model_dump()
        save_checkpoint(
            model, optimizer, epoch, avg_val, cfg_dict, os.path.join(ckpt_dir, "last_model.pt")
        )
        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(
                model, optimizer, epoch, avg_val, cfg_dict, os.path.join(ckpt_dir, "best_model.pt")
            )

    writer.close()


if __name__ == "__main__":
    train()
