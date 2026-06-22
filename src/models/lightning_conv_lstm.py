import os
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

from src.models.conv_lstm import ConvLSTMModel
from src.models.shared.datasets import SequentialDataset, dense_sequential_collate_fn
from src.training.schemas import ConvLSTMTrainingConfigSchema

log = logging.getLogger(__name__)


def make_times(t_offset: int, h_ctx: int, seq_len: int, device, B: int) -> torch.Tensor:
    """Relative times [t_offset .. t_offset+h_ctx-1] normalised to [0,1]. → (B, h_ctx)"""
    t = torch.arange(t_offset, t_offset + h_ctx, device=device, dtype=torch.float32)
    t = t / max(seq_len - 1, 1)
    return t.unsqueeze(0).expand(B, -1)


def log_vis(
    model,
    loader,
    t_cfg: ConvLSTMTrainingConfigSchema,
    H,
    W,
    device,
    writer,
    epoch,
    rollout_steps=(1, 5, 10, 15),
):
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
                mu_img = preds[idx]["mean"].astype("float32")[0]
                std_img = preds[idx]["std"].astype("float32")[0]

                abs_t = h_ctx + idx
                gt_img = (
                    frames_all[b_idx, abs_t].cpu().numpy()
                    if abs_t < T
                    else np.zeros((H, W), dtype=np.float32)
                )
                err = np.abs(gt_img - mu_img).mean()
                col = j * 3

                ax = axes[b_idx, col]
                sc = ax.imshow(gt_img, vmin=0, vmax=1, cmap="rainbow", origin="lower")
                ax.set_title(f"GT +{step} b={b_idx}")
                plt.colorbar(sc, ax=ax)

                ax = axes[b_idx, col + 1]
                sc2 = ax.imshow(mu_img, vmin=0, vmax=1, cmap="rainbow", origin="lower")
                ax.set_title(f"μ +{step}  err={err:.3f}")
                plt.colorbar(sc2, ax=ax)

                ax = axes[b_idx, col + 2]
                sc3 = ax.imshow(std_img, vmin=0, vmax=1, cmap="hot", origin="lower")
                ax.set_title(f"σ +{step}  μσ={std_img.mean():.3f}")
                plt.colorbar(sc3, ax=ax)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_t = torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1)
    writer.add_image("Val/Rollout", img_t, epoch)
    plt.close()


class ConvLSTMDataModule(L.LightningDataModule):
    def __init__(self, t_cfg: ConvLSTMTrainingConfigSchema, data_path: Path):
        super().__init__()
        self.t_cfg = t_cfg
        self.data_path = data_path
        self.train_ds = None
        self.val_ds = None
        self.H = None
        self.W = None

    def setup(self, stage=None):
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
        self.H = self.train_ds.H
        self.W = self.train_ds.W

    def train_dataloader(self):
        workers = self.t_cfg.data.num_workers
        if workers == 0 and "SLURM_CPUS_PER_TASK" in os.environ:
            workers = max(1, int(os.environ["SLURM_CPUS_PER_TASK"]) - 1)

        return DataLoader(
            self.train_ds,
            batch_size=self.t_cfg.data.batch_size,
            shuffle=True,
            collate_fn=dense_sequential_collate_fn,
            num_workers=workers,
        )

    def val_dataloader(self):
        workers = self.t_cfg.data.num_workers
        if workers == 0 and "SLURM_CPUS_PER_TASK" in os.environ:
            workers = max(1, int(os.environ["SLURM_CPUS_PER_TASK"]) - 1)

        return DataLoader(
            self.val_ds,
            batch_size=self.t_cfg.data.batch_size,
            shuffle=False,
            collate_fn=dense_sequential_collate_fn,
            num_workers=workers,
        )


class ConvLSTMLightningModule(L.LightningModule):
    def __init__(self, t_cfg: ConvLSTMTrainingConfigSchema, H: int, W: int):
        super().__init__()
        self.save_hyperparameters(ignore=["H", "W"])
        self.t_cfg = t_cfg
        self.H = H
        self.W = W
        self.model = ConvLSTMModel(t_cfg.model)
        self.loss_cfg = t_cfg.loss
        self.opt_cfg = t_cfg.optimizer

        self.automatic_optimization = False

    def forward(self, frames, times=None):
        return self.model(frames, times)

    def beta_nll_loss(self, dist, gt, beta):
        var = dist.variance
        nll = 0.5 * (((dist.mean - gt) ** 2) / var + torch.log(var))
        weight = var.detach() ** beta
        loss = nll * weight
        return loss.mean()

    def criterion(self, dist, gt):
        if self.loss_cfg.beta is not None:
            return self.beta_nll_loss(dist, gt, beta=self.loss_cfg.beta)
        return -dist.log_prob(gt).mean()

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
                self.criterion(dists[h], targets[:, t, :, :, h].unsqueeze(-1))
                for h in range(h_pred)
            ) / (h_pred * n_win)

            self.manual_backward(step_loss)
            batch_loss += step_loss.item()

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.opt_cfg.grad_clip)
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
        for t in range(h_ctx - 1, T - h_pred):
            ctx_w = frames[:, t - h_ctx + 1 : t + 1]
            times = make_times(t - h_ctx + 1, h_ctx, seq_len, self.device, B_size)
            dists = self(ctx_w, times)
            for h in range(h_pred):
                gt_h = targets[:, t, :, :, h].unsqueeze(-1)
                batch_loss += self.criterion(dists[h], gt_h).item()

        avg_loss = batch_loss / (h_pred * n_win)

        self.log("Val/Loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Val/NLL", avg_loss, on_step=False, on_epoch=True)

        return avg_loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        opt = self.optimizers()
        self.log("Train/LR", opt.param_groups[0]["lr"], on_epoch=True)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        val_loss = metrics.get("Val/Loss")
        train_loss = metrics.get("Train/Loss")

        if val_loss is not None:
            epoch = self.trainer.current_epoch
            train_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
            log.info(
                f"Epoch {epoch}: Train={train_str}  Val={val_loss:.4f}"
            )

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
            },
        }


class ConvLSTMVisualizerCallback(L.Callback):
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
                device=pl_module.device,
                writer=writer,
                epoch=epoch,
            )
