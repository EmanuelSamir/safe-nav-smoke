import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from models.dreamer_v1_vae import DreamerV1VAE


class Dreamer1DDataset(Dataset):
    def __init__(self, data_path, seq_len=10, max_range=8.0, sample_ratio=1.0):
        # 1. Select only needed columns to avoid loading 12GB of 2D maps
        full_ds = load_from_disk(data_path)
        needed_cols = ["obs_readings", "action", "terminated", "truncated"]
        self.dataset = full_ds.select_columns(needed_cols)

        self.seq_len = seq_len
        self.max_range = max_range
        self.sample_ratio = sample_ratio

        # 2. Extract into memory for lightning fast access
        print(f"Pre-loading {len(self.dataset)} rows into memory...")
        self.all_readings = (
            np.array(self.dataset["obs_readings"], dtype=np.float32) / self.max_range
        )
        self.all_actions = np.array(self.dataset["action"], dtype=np.float32)
        terminated = np.array(self.dataset["terminated"])
        truncated = np.array(self.dataset["truncated"])

        # 3. Find valid sequence starts
        self.valid_starts = []
        ep_start = 0
        for i in range(len(terminated)):
            if terminated[i] or truncated[i]:
                if (i - ep_start + 1) >= seq_len:
                    self.valid_starts.extend(range(ep_start, i - seq_len + 2))
                ep_start = i + 1

        # 4. Apply sample_ratio
        if self.sample_ratio < 1.0:
            num_samples = int(len(self.valid_starts) * self.sample_ratio)
            indices = np.random.choice(len(self.valid_starts), max(1, num_samples), replace=False)
            self.valid_starts = [self.valid_starts[i] for i in sorted(indices)]

        print(f"Dataset ready with {len(self.valid_starts)} valid sequences.")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        readings = self.all_readings[start_idx : start_idx + self.seq_len]
        actions = self.all_actions[start_idx : start_idx + self.seq_len]
        return torch.from_numpy(readings), torch.from_numpy(actions)


class DreamerV1Module(pl.LightningModule):
    def __init__(self, action_dim=2, in_features=64, kl_scale=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model = DreamerV1VAE(in_features=in_features, act_dim=action_dim)
        self.kl_scale = kl_scale

    def training_step(self, batch, batch_idx):
        obs, acts = batch
        h, s, priors, posteriors = self.model(obs, acts)

        # 1. Reconstruction Loss
        recon_obs = self.model.decoder(h, s)
        loss_recon = F.mse_loss(recon_obs, obs)

        # 2. KL Divergence Loss
        loss_kl = 0
        for p, q in zip(priors, posteriors):
            loss_kl += torch.distributions.kl.kl_divergence(q, p).mean()
        loss_kl /= len(priors)

        loss = loss_recon + self.kl_scale * loss_kl

        self.log("train/loss", loss)
        self.log("train/loss_recon", loss_recon)
        self.log("train/loss_kl", loss_kl)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, acts = batch
        B, T, _ = obs.shape
        h, s, priors, posteriors = self.model(obs, acts)

        recon_obs = self.model.decoder(h, s)
        loss_recon = F.mse_loss(recon_obs, obs)

        loss_kl = 0
        for p, q in zip(priors, posteriors):
            loss_kl += torch.distributions.kl.kl_divergence(q, p).mean()
        loss_kl /= len(priors)

        loss = loss_recon + self.kl_scale * loss_kl

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/loss_recon", loss_recon)
        self.log("val/loss_kl", loss_kl)

        # Visualization (Rotate batches every epoch to see different data)
        # Assuming we want to log only once per validation epoch
        if batch_idx == (self.current_epoch % 10) and self.logger is not None:
            P = min(5, T // 2)
            if P > 0 and T > P:
                num_samples = min(3, B)
                # Select a few samples (e.g., first, middle, last)
                sample_indices = [0, B // 2, B - 1] if B >= 3 else list(range(B))
                
                fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
                if num_samples == 1: axes = np.expand_dims(axes, axis=0)
                
                vmin, vmax = 0, 1
                
                for i, idx in enumerate(sample_indices):
                    # 1. Ghost of the past (Context)
                    z_obs = obs[idx:idx+1, :P]
                    z_acts = acts[idx:idx+1, :P]
                    h_past, s_past, _, _ = self.model(z_obs, z_acts)
                    
                    # 2. Imagination (Future)
                    future_acts = acts[idx:idx+1, P:]
                    h_imag, s_imag = self.model.rollout(h_past[:, -1], s_past[:, -1], future_acts)
                    
                    # 3. Decode
                    o_imag = self.model.decoder(h_imag, s_imag)
                    
                    # Clean data for plotting
                    gt_scan = np.nan_to_num(obs[idx].cpu().numpy())
                    recon_scan = np.nan_to_num(recon_obs[idx].detach().cpu().numpy())
                    imag_data = np.nan_to_num(o_imag[0].detach().cpu().numpy())

                    imag_scan = np.zeros_like(gt_scan)
                    imag_scan[:P] = recon_scan[:P]
                    imag_scan[P:] = imag_data

                    # Row i
                    axes[i, 0].imshow(gt_scan, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                    axes[i, 0].set_title(f"GT (Batch {batch_idx}, Sample {idx})")
                    axes[i, 1].imshow(recon_scan, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                    axes[i, 1].set_title("Posterior Recon")
                    axes[i, 2].imshow(imag_scan, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                    axes[i, 2].set_title("Mental Rollout")
                    axes[i, 2].axhline(y=P - 0.5, color="white", linestyle="--")

                plt.tight_layout()
                self.logger.experiment.add_figure(
                    "Val/Mental_Rollout_Diversity", fig, global_step=self.global_step
                )
                plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


@hydra.main(version_base="1.3", config_path="configs/training", config_name="smoke_world_train")
def main(cfg: DictConfig):
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        output_dir = "outputs"

    pl.seed_everything(42)
    dataset = Dreamer1DDataset(
        data_path=cfg.data.data_path,
        seq_len=cfg.data.get("seq_len", 16),
        sample_ratio=cfg.data.get("sample_ratio", 1.0),
    )
    split = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0
    )

    model = DreamerV1Module(action_dim=2, in_features=64)

    tb_logger = TensorBoardLogger(save_dir=output_dir, name="dreamer_v1_logs")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)

    accelerator = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        accelerator = "cuda"

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback, LearningRateMonitor()],
        logger=tb_logger,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
