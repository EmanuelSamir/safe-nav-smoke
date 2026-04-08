import os
import sys

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.getcwd())

from models.shared.smoke_world_dataset import SmokeWorldDataset
from models.smoke_fno import SmokeWorldModel


def warp_patch(patch: torch.Tensor, delta_loc: torch.Tensor, res: float = 0.2):
    """Warps/shifts a patch based on relative movement delta_loc (meters).
    patch: (B, C, H, W)
    delta_loc: (B, 2) - [dx, dy] in meters
    """
    B, C, H, W = patch.shape
    # Convert meters to normalized grid coordinates [-1, 1]
    # In a patch of H pixels, H*res meters.
    # A shift of dx meters is dx / (H*res) pixels of the half-width.
    dx_norm = -delta_loc[:, 0] / (W * res / 2)  # Negative because we pull from old frame
    dy_norm = -delta_loc[:, 1] / (H * res / 2)

    theta = torch.zeros(B, 2, 3, device=patch.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = dx_norm
    theta[:, 1, 2] = dy_norm

    grid = F.affine_grid(theta, patch.size(), align_corners=False)
    warped = F.grid_sample(patch, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped


class AutoregressiveFluidWorldModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.past_frames = cfg.data.get("past_frames", 4)

        # FNO Architecture: expects 3 + past_frames channels input, produces 2 channels (mu, log_var)
        self.model = SmokeWorldModel(
            in_channels=3 + self.past_frames,
            out_channels=2,
            latent_dim=cfg.model.latent_dim,
            modes=cfg.model.modes,
        )

        self.res = 0.2
        self.learning_rate = cfg.training.lr

    def get_curriculum_length(self, max_L: int) -> int:
        """Gradually increases the sequence length L during training."""
        curriculum_epochs = self.cfg.training.get("curriculum_epochs", 0)
        if curriculum_epochs > 0 and max_L > 1:
            progress = min(1.0, self.current_epoch / curriculum_epochs)
            return 1 + int(progress * (max_L - 1))
        return max_L

    def calculate_nll(self, mu, log_var, target):
        """Gaussian Negative Log-Likelihood."""
        # Clamp log var to prevent exponential precision explosions on unseen validation data
        log_var = torch.clamp(log_var, min=-8.0, max=5.0)
        precision = torch.exp(-log_var)
        return 0.5 * (precision * (target - mu) ** 2 + log_var + np.log(2 * np.pi))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # x_seq: (B, L, 7, 80, 80)
        # y_seq: (B, L, 1, 80, 80)
        x_seq, y_seq = batch
        B, max_L, _, H, W = x_seq.shape

        L = self.get_curriculum_length(max_L)

        total_loss = 0

        # Initialize memory with zeros (simulating real-world "fog of war" at sequence start)
        current_memory = torch.zeros(B, self.past_frames, H, W, device=x_seq.device)

        for t in range(L):
            # 1. Current context: O_t, V_t, u_t
            o_t = x_seq[:, t, 0:1]
            v_t = x_seq[:, t, 1:2]
            u_t = x_seq[:, t, 2:3]

            # 2. Input Stack: [O_t, memory, V_t, u_t]
            model_input = torch.cat([o_t, current_memory, v_t, u_t], dim=1)

            # 3. Predict mu_t, log_var_t
            output = self(model_input)
            mu_t = output[:, 0:1]
            log_var_t = output[:, 1:2]

            # 4. Calculate Loss (NLL)
            loss_t = self.calculate_nll(mu_t, log_var_t, y_seq[:, t]).mean()
            total_loss += loss_t

            # 5. Update Memory for t+1
            # Shift predictions:
            if self.past_frames == 1:
                current_memory = mu_t.detach()
            else:
                current_memory = torch.cat(
                    [mu_t.detach(), current_memory[:, 0 : self.past_frames - 1]], dim=1
                )

        loss = total_loss / L
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_seq, y_seq = batch
        B, max_L, _, H, W = x_seq.shape

        L = self.get_curriculum_length(max_L)

        total_loss = 0
        total_mse = 0

        # Initialize memory with zeros (simulating real-world "fog of war" at sequence start)
        current_memory = torch.zeros(B, self.past_frames, H, W, device=x_seq.device)

        if batch_idx == 0 and self.logger is not None:
            vis_data = []

        for t in range(L):
            o_t = x_seq[:, t, 0:1]
            v_t = x_seq[:, t, 1:2]
            u_t = x_seq[:, t, 2:3]
            model_input = torch.cat([o_t, current_memory, v_t, u_t], dim=1)
            output = self(model_input)
            mu_t, log_var_t = output[:, 0:1], output[:, 1:2]

            loss_t = self.calculate_nll(mu_t, log_var_t, y_seq[:, t]).mean()
            mse_t = F.mse_loss(mu_t, y_seq[:, t])

            total_loss += loss_t
            total_mse += mse_t

            if batch_idx == 0 and self.logger is not None:
                vis_data.append(
                    {
                        "o": o_t[0, 0].detach().cpu().numpy(),
                        "mem": current_memory[0].detach().cpu().numpy(),
                        "mu": mu_t[0, 0].detach().cpu().numpy(),
                        "std": torch.exp(0.5 * log_var_t)[0, 0].detach().cpu().numpy(),
                        "gt": y_seq[0, t, 0].detach().cpu().numpy(),
                    }
                )

            if self.past_frames == 1:
                current_memory = mu_t
            else:
                current_memory = torch.cat(
                    [mu_t, current_memory[:, 0 : self.past_frames - 1]], dim=1
                )

        loss = total_loss / L
        mse = total_mse / L
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)

        if batch_idx == 0 and self.logger is not None:
            from io import BytesIO

            import matplotlib.pyplot as plt
            from PIL import Image

            steps_to_plot = min(L, 4)
            step_indices = np.linspace(0, L - 1, steps_to_plot, dtype=int)
            cols = 4 + self.past_frames

            fig, axes = plt.subplots(steps_to_plot, cols, figsize=(4 * cols, 4 * steps_to_plot))
            if steps_to_plot == 1:
                axes = np.expand_dims(axes, 0)

            for i, t in enumerate(step_indices):
                d = vis_data[t]

                # 1. Cone
                ax = axes[i, 0]
                masked_o = np.ma.masked_where(d["o"] == 0, d["o"])
                cmap_obs = plt.get_cmap("rainbow").copy()
                cmap_obs.set_bad(color="black")
                im = ax.imshow(masked_o, vmin=0, vmax=1, cmap=cmap_obs, origin="lower")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if i == 0:
                    ax.set_title("O_t (Cone)")
                ax.set_ylabel(f"t={t}")

                # 2. Memory
                for m_idx in range(self.past_frames):
                    ax = axes[i, 1 + m_idx]
                    im = ax.imshow(d["mem"][m_idx], vmin=0, vmax=1, cmap="rainbow", origin="lower")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    if i == 0:
                        ax.set_title(f"Mem -{m_idx + 1}")

                # 3. Prediction mu
                ax = axes[i, 1 + self.past_frames]
                im = ax.imshow(d["mu"], vmin=0, vmax=1, cmap="rainbow", origin="lower")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if i == 0:
                    ax.set_title("Pred (mu)")

                # 4. Standard Dev
                ax = axes[i, 2 + self.past_frames]
                im = ax.imshow(d["std"], vmin=0, cmap="hot", origin="lower")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if i == 0:
                    ax.set_title("Uncertainty (std)")

                # 5. Ground Truth
                ax = axes[i, 3 + self.past_frames]
                im = ax.imshow(d["gt"], vmin=0, vmax=1, cmap="rainbow", origin="lower")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if i == 0:
                    ax.set_title("GT")

                for ax_obj in axes[i]:
                    ax_obj.set_xticks([])
                    ax_obj.set_yticks([])

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_t = torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1)
            self.logger.experiment.add_image("Val/Sequence", img_t, self.global_step)
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


@hydra.main(version_base=None, config_path="configs/training", config_name="smoke_world_train")
def train(cfg: DictConfig):
    # Get Hydra output directory
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        output_dir = "outputs"

    # 1. Dataset & Dataloaders
    train_ds = SmokeWorldDataset(
        data_path=cfg.data.data_path,
        seq_len=cfg.data.get("seq_len", 8),
        past_frames=cfg.data.get("past_frames", 4),
        sample_ratio=cfg.data.sample_ratio,
        mode="train",
    )
    val_ds = SmokeWorldDataset(
        data_path=cfg.data.data_path,
        seq_len=cfg.data.get("seq_len", 8),
        past_frames=cfg.data.get("past_frames", 4),
        sample_ratio=cfg.data.sample_ratio,
        mode="val",
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)

    # 2. Model
    model = AutoregressiveFluidWorldModel(cfg)

    # 3. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="smoke-world-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. Logger & Trainer
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="tb_logs")

    # Check for Apple Silicon (mps) or CUDA
    accelerator = "auto"
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=1,
        limit_train_batches=cfg.data.sample_ratio,
        limit_val_batches=cfg.data.sample_ratio,
    )

    # 5. Start training
    print("Starting Autoregressive Fluid World Model training...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
