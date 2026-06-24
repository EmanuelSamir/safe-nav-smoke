import os
import sys

sys.path.append(os.getcwd())

import logging
from pathlib import Path

import datetime
import yaml
import lightning as L

from src.models.lightning_conv_lstm import (
    ConvLSTMDataModule,
    ConvLSTMLightningModule,
    ConvLSTMVisualizerCallback,
)
from projects.2_training.schema import ConvLSTMTrainingConfig

log = logging.getLogger(__name__)


def train():
    config_path = os.path.join(os.path.dirname(__file__), "../../configs/training/conv_lstm.yaml")
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    t_cfg = ConvLSTMTrainingConfig(**yaml_data)

    print(f"Training ConvLSTM — {t_cfg.experiment_name}")
    L.seed_everything(t_cfg.seed)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "outputs", "training", t_cfg.experiment_name, timestamp)

    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        yaml.dump(t_cfg.model_dump(), f)

    root_dir = Path(os.getcwd())

    data_path = root_dir / t_cfg.data.data_path
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    # DataModule
    datamodule = ConvLSTMDataModule(t_cfg, data_path)
    datamodule.setup()

    # Model
    model = ConvLSTMLightningModule(t_cfg, datamodule.H, datamodule.W)
    print(f"ConvLSTM params: {sum(p.numel() for p in model.model.parameters()):,}")

    # Checkpoint and Loggers
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=t_cfg.checkpoint.monitor.replace("val_", "Val/")
        .replace("nll", "NLL")
        .replace("loss", "Loss"),  # Map val_nll to Val/NLL
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
        callbacks=[
            checkpoint_callback,
            ConvLSTMVisualizerCallback(t_cfg.visualizer.visualize_every),
        ],
        logger=tb_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        fast_dev_run=t_cfg.test,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
