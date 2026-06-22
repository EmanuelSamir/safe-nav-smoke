import os
import sys

sys.path.append(os.getcwd())

import logging
from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from src.models.lightning_fno import FNODataModule, FNOLightningModule, FNOVisualizerCallback
from src.training.schemas import FNOTrainingConfigSchema

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/training", config_name="fno")
def train(cfg: DictConfig):
    # Merge with structured schema to validate and convert to dataclass
    schema = OmegaConf.structured(FNOTrainingConfigSchema)
    merged = OmegaConf.merge(schema, cfg)
    t_cfg = OmegaConf.to_object(merged)

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

    # Data Setup
    try:
        root_dir = Path(hydra.utils.get_original_cwd())
    except Exception:
        root_dir = Path(os.getcwd())

    data_path = root_dir / t_cfg.data.data_path
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    # DataModule
    datamodule = FNODataModule(t_cfg, data_path)
    datamodule.setup()

    # Model
    model = FNOLightningModule(
        t_cfg, datamodule.H, datamodule.W, datamodule.x_size, datamodule.y_size
    )
    print(f"FNO params: {sum(p.numel() for p in model.model.parameters()):,}")

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
        callbacks=[checkpoint_callback, FNOVisualizerCallback(t_cfg.visualizer.visualize_every)],
        logger=tb_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        fast_dev_run=t_cfg.test,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
