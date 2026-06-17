import os
import sys

sys.path.append(os.getcwd())

import logging
import random
from io import BytesIO
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from pydantic import BaseModel, ConfigDict, Field

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


def make_times(t_offset: int, h_ctx: int, seq_len: int, device, B: int) -> torch.Tensor:
    """Relative times [t_offset .. t_offset+h_ctx-1] normalised to [0,1]. → (B, h_ctx)"""
    t = torch.arange(t_offset, t_offset + h_ctx, device=device, dtype=torch.float32)
    t = t / max(seq_len - 1, 1)
    return t.unsqueeze(0).expand(B, -1)


def log_vis(model, loader, t_cfg: FNOTrainingConfigSchema, H, W, x_size, y_size, device, writer, epoch, rollout_steps=(1, 5, 10, 15)):
    model.eval()
    ctx_obs, trg_obs, _ = next(iter(loader))
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


class FNODataModule(L.LightningDataModule):
    def __init__(self, t_cfg: FNOTrainingConfigSchema, data_path: Path):
        super().__init__()
        self.t_cfg = t_cfg
        self.data_path = data_path
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        h_ctx = self.t_cfg.model.h_ctx
        h_pred = self.t_cfg.model.h_pred
        seq_len = self.t_cfg.data.sequence_length
        max_ep = self.t_cfg.data.max_samples

        self.train_ds = SequentialDataset(
            data_path=str(self.data_path),
            sequence_length=seq_len,
            forecast_horizon=h_pred,
            mode="train",
            train_split=self.t_cfg.data.train_split,
            max_episodes=max_ep,
            dense=True,
            downsample_factor=self.t_cfg.data.downsample_factor,
        )
        self.val_ds = SequentialDataset(
            data_path=str(self.data_path),
            sequence_length=seq_len,
            forecast_horizon=h_pred,
            mode="val",
            train_split=self.t_cfg.data.train_split,
            max_episodes=max_ep,
            dense=True,
            downsample_factor=self.t_cfg.data.downsample_factor,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.t_cfg.data.batch_size,
            shuffle=True,
            collate_fn=dense_sequential_collate_fn,
            num_workers=self.t_cfg.data.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.t_cfg.data.batch_size,
            shuffle=False,
            collate_fn=dense_sequential_collate_fn,
            num_workers=self.t_cfg.data.num_workers,
        )


class FNOLightningModule(L.LightningModule):
    def __init__(self, t_cfg: FNOTrainingConfigSchema, H: int, W: int, x_size: float, y_size: float):
        super().__init__()
        self.save_hyperparameters(ignore=["H", "W", "x_size", "y_size"])
        self.t_cfg = t_cfg
        self.H = H
        self.W = W
        self.x_size = x_size
        self.y_size = y_size
        self.model = FNO(t_cfg.model)
        self.loss_cfg = t_cfg.loss
        self.opt_cfg = t_cfg.optimizer
        
        # Turn off automatic optimization to preserve manual backward step-by-step logic
        self.automatic_optimization = False

    def forward(self, ctx_w, times):
        return self.model(ctx_w, times)

    def beta_nll_loss(self, dist, gt, beta):
        var = dist.variance
        nll = 0.5 * (((dist.mean - gt) ** 2) / var + torch.log(var))
        weight = var.detach() ** beta
        loss = nll * weight
        return loss.mean()

    def energy_score_loss(self, dist, gt, m_samples, normalize_l2):
        samples = dist.rsample(torch.Size([m_samples]))  # (M, B, H, W, 1)
        diff_gt = samples - gt.unsqueeze(0)  # (M, B, H, W, 1)
        if normalize_l2:
            norm_gt = torch.sqrt(torch.mean(diff_gt ** 2, dim=(-3, -2, -1)) + 1e-8)  # (M, B)
        else:
            norm_gt = torch.sqrt(torch.sum(diff_gt ** 2, dim=(-3, -2, -1)) + 1e-8)  # (M, B)
        term1 = norm_gt.mean(dim=0)  # (B,)

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

    def criterion(self, dist, gt):
        if self.loss_cfg.name == "energy_score":
            return self.energy_score_loss(dist, gt, m_samples=self.loss_cfg.m_samples, normalize_l2=self.loss_cfg.normalize_l2)
        elif self.loss_cfg.name == "nll":
            if self.loss_cfg.beta is not None:
                return self.beta_nll_loss(dist, gt, beta=self.loss_cfg.beta)
            return -dist.log_prob(gt).mean()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_cfg.name}")

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        ctx_obs, trg_obs, _ = batch
        T = ctx_obs.xs.shape[1]
        B_size = ctx_obs.xs.shape[0]
        h_ctx = self.t_cfg.model.h_ctx
        h_pred = self.t_cfg.model.h_pred
        seq_len = self.t_cfg.data.sequence_length

        frames = ctx_obs.values[:, :, :, 0].view(B_size, T, self.H, self.W)
        targets = trg_obs.values.view(B_size, T, self.H, self.W, h_pred)

        n_win = T - h_ctx - h_pred + 1
        if n_win <= 0:
            return

        batch_loss = 0.0
        for t in range(h_ctx - 1, T - h_pred):
            ctx_w = frames[:, t - h_ctx + 1 : t + 1]  # (B, h_ctx, H, W)
            times = make_times(t - h_ctx + 1, h_ctx, seq_len, self.device, B_size)

            dists = self(ctx_w, times)  # List[Normal] of h_pred, (B,H,W,1)

            step_loss = sum(
                self.criterion(dists[h], targets[:, t, :, :, h].unsqueeze(-1)) for h in range(h_pred)
            ) / (h_pred * n_win)

            self.manual_backward(step_loss)
            batch_loss += step_loss.item() * n_win

        # Clip gradients using custom complex-supporting clipping
        clip_grad_norm_(self.parameters(), self.opt_cfg.grad_clip)
        opt.step()

        self.log("Train/Loss", batch_loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        ctx_obs, trg_obs, _ = batch
        T = ctx_obs.xs.shape[1]
        B_size = ctx_obs.xs.shape[0]
        h_ctx = self.t_cfg.model.h_ctx
        h_pred = self.t_cfg.model.h_pred
        seq_len = self.t_cfg.data.sequence_length

        frames = ctx_obs.values[:, :, :, 0].view(B_size, T, self.H, self.W)
        targets = trg_obs.values.view(B_size, T, self.H, self.W, h_pred)

        n_win = T - h_ctx - h_pred + 1
        if n_win <= 0:
            return

        batch_loss = 0.0
        batch_nll = 0.0
        batch_es = 0.0
        for t in range(h_ctx - 1, T - h_pred):
            ctx_w = frames[:, t - h_ctx + 1 : t + 1]
            times = make_times(t - h_ctx + 1, h_ctx, seq_len, self.device, B_size)
            dists = self(ctx_w, times)
            for h in range(h_pred):
                gt_h = targets[:, t, :, :, h].unsqueeze(-1)
                batch_loss += self.criterion(dists[h], gt_h).item()
                batch_nll += -dists[h].log_prob(gt_h).mean().item()
                batch_es += self.energy_score_loss(dists[h], gt_h, m_samples=self.loss_cfg.m_samples, normalize_l2=self.loss_cfg.normalize_l2).item()

        avg_loss = batch_loss / (h_pred * n_win)
        avg_nll = batch_nll / (h_pred * n_win)
        avg_es = batch_es / (h_pred * n_win)

        self.log("Val/Loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Val/NLL", avg_nll, on_step=False, on_epoch=True)
        self.log("Val/EnergyScore", avg_es, on_step=False, on_epoch=True)

        return avg_loss

    def on_train_epoch_end(self):
        # Step the scheduler at the end of each epoch in manual optimization mode
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        opt = self.optimizers()
        self.log("Train/LR", opt.param_groups[0]["lr"], on_epoch=True)

    def on_validation_epoch_end(self):
        # Replicates manual log reporting at the end of validation
        metrics = self.trainer.callback_metrics
        val_loss = metrics.get("Val/Loss")
        val_nll = metrics.get("Val/NLL")
        val_es = metrics.get("Val/EnergyScore")
        train_loss = metrics.get("Train/Loss")

        if val_loss is not None:
            epoch = self.trainer.current_epoch
            train_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
            log.info(f"Epoch {epoch}: Train={train_str}  Val={val_loss:.4f}  Val_NLL={val_nll:.4f}  Val_ES={val_es:.4f}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.opt_cfg.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.opt_cfg.max_epochs, eta_min=self.opt_cfg.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }


class FNOVisualizerCallback(L.Callback):
    def __init__(self, visualize_every: int):
        super().__init__()
        self.visualize_every = visualize_every

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.visualize_every != 0:
            return

        val_loader = trainer.val_dataloaders
        if not val_loader:
            return
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        logger = trainer.logger
        if logger is not None:
            writer = logger.experiment
            log_vis(
                model=pl_module.model,
                loader=val_loader,
                t_cfg=pl_module.t_cfg,
                H=pl_module.H,
                W=pl_module.W,
                x_size=pl_module.x_size,
                y_size=pl_module.y_size,
                device=pl_module.device,
                writer=writer,
                epoch=epoch
            )


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
    L.seed_everything(t_cfg.seed)

    from hydra.core.hydra_config import HydraConfig

    output_dir = HydraConfig.get().runtime.output_dir
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Data Setup for getting Grid HxW
    try:
        root_dir = Path(hydra.utils.get_original_cwd())
    except Exception:
        root_dir = Path(os.getcwd())

    data_path = root_dir / t_cfg.data.data_path
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    # Create temporary dataset just to inspect sizes (grid H, W, spatial size)
    temp_ds = SequentialDataset(
        data_path=str(data_path),
        sequence_length=t_cfg.data.sequence_length,
        forecast_horizon=t_cfg.model.h_pred,
        mode="val",
        train_split=t_cfg.data.train_split,
        max_episodes=1,
        dense=True,
        downsample_factor=t_cfg.data.downsample_factor,
    )
    H, W = temp_ds.H, temp_ds.W
    x_size, y_size = temp_ds.x_size, temp_ds.y_size

    # DataModule
    datamodule = FNODataModule(t_cfg, data_path)

    # Model
    model = FNOLightningModule(t_cfg, H, W, x_size, y_size)
    print(f"FNO params: {sum(p.numel() for p in model.model.parameters()):,}")

    # Checkpoint and Loggers
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=t_cfg.checkpoint.monitor.replace("val_", "Val/").replace("nll", "NLL").replace("loss", "Loss"), # Map val_nll to Val/NLL
        mode=t_cfg.checkpoint.mode,
        save_top_k=t_cfg.checkpoint.save_top_k,
        save_last=True,
    )

    tb_logger = TensorBoardLogger(save_dir=output_dir, name="", sub_dir="logs")

    # Trainer
    trainer = L.Trainer(
        max_epochs=t_cfg.optimizer.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, FNOVisualizerCallback(t_cfg.visualizer.visualize_every)],
        logger=tb_logger,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
