import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
import logging
from pathlib import Path

import datetime
import yaml
import lightning as L

from src.models.lightning_fno import FNOLightningModule
from src.models.shared.base_lightning import BaseDataModule, BaseVisualizerCallback
from src.models.shared.schemas import FNOTrainingConfig

log = logging.getLogger(__name__)


import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="fno_config.yaml", help="Path to config file")
    args, _ = parser.parse_known_args()
    config_name = args.config
    config_path = os.path.join(os.path.dirname(__file__), config_name)
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    t_cfg = FNOTrainingConfig.model_validate(yaml_data)

    print(f"Training FNO — {t_cfg.experiment_name}")
    L.seed_everything(t_cfg.seed)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "outputs", t_cfg.project_name, t_cfg.sub_project_name, t_cfg.experiment_name, timestamp)

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
    datamodule = BaseDataModule(t_cfg, data_path)
    datamodule.setup()

    # Model
    model = FNOLightningModule(
        t_cfg, datamodule.H, datamodule.W, datamodule.x_size, datamodule.y_size
    )
    print(f"FNO params: {sum(p.numel() for p in model.model.parameters()):,}")

    # Checkpoint and Loggers
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger

    monitor_metric = t_cfg.checkpoint.monitor.replace("val_", "Val/").replace("nll", "NLL").replace("loss", "Loss")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=monitor_metric,
        mode=t_cfg.checkpoint.mode,
        save_top_k=t_cfg.checkpoint.save_top_k,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=500,
        mode=t_cfg.checkpoint.mode,
        verbose=True
    )

    tb_logger = TensorBoardLogger(save_dir=output_dir, name="", sub_dir="logs")

    # Trainer
    trainer = L.Trainer(
        max_epochs=t_cfg.optimizer.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping, BaseVisualizerCallback(t_cfg.visualizer.visualize_every)],
        logger=tb_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        fast_dev_run=t_cfg.test,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
