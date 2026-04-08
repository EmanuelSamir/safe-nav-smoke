import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from models.dreamer_v4_fluid import FluidWorldModelDreamer


class Dreamer1DDataset(Dataset):
    def __init__(self, data_path, seq_len=10, max_range=8.0):
        self.dataset = load_from_disk(data_path)
        self.seq_len = seq_len
        self.max_range = max_range

        terminated = self.dataset["terminated"]
        truncated = self.dataset["truncated"]
        self.valid_starts = []
        ep_start = 0
        for i in range(len(terminated)):
            if terminated[i] or truncated[i]:
                if (i - ep_start + 1) >= seq_len:
                    self.valid_starts.extend(range(ep_start, i - seq_len + 2))
                ep_start = i + 1

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        block = self.dataset[start_idx : start_idx + self.seq_len]

        # Extract 1D readings [seq_len, 64] & normalize to [0, 1] for Sigmoid Decoder
        readings = np.array(block["obs_readings"]) / self.max_range
        # Actions [seq_len, 2] (Assumes v, w are already suitable inputs)
        actions = np.array(block["action"])

        return torch.from_numpy(readings).float(), torch.from_numpy(actions).float()


@hydra.main(version_base="1.3", config_path="configs/training", config_name="smoke_world_train")
def main(cfg: DictConfig):
    # Get Hydra output directory
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        output_dir = "outputs"

    pl.seed_everything(42)

    # 1. Dataset
    print(f"Loading dataset from {cfg.data.data_path}...")
    dataset = Dreamer1DDataset(data_path=cfg.data.data_path, seq_len=cfg.data.get("seq_len", 10))
    print(f"Dataset generated {len(dataset)} valid sequences.")

    # 2. Strict Contiguous Split (Avoids temporal leakage from random overlaps)
    split = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # 3. Model
    model = FluidWorldModelDreamer(action_dim=2, in_features=64)

    # 4. Logger & Callbacks
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="tb_logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="dreamer-{epoch:02d}-{val/loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 5. Trainer
    accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        accelerator = "cuda"

    sample_ratio = cfg.data.get("sample_ratio", 1.0)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=1,
        limit_train_batches=sample_ratio,
        limit_val_batches=sample_ratio,
    )

    # 6. Train
    print("Starting Training Loop for Dreamer V4 Latent World Model...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
