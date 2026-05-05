import os
import sys

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Add project root to path
sys.path.append(os.getcwd())

from hydra.core.hydra_config import HydraConfig
from pdd.lightning import PDDDataModule, PDDShortcutModule


@hydra.main(version_base="1.3", config_path="../configs/training", config_name="pdd_train")
def main(cfg: DictConfig):
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        output_dir = "outputs"

    # 1. Init DataModule
    dm = PDDDataModule(
        dataset_path=cfg.data.data_path,
        batch_size=cfg.training.batch_size,
        window_size=cfg.data.window_size,
        horizon=cfg.data.horizon,
        action_dim=cfg.model.action_dim,
        val_split=cfg.training.val_split,
    )

    # 2. Init Model
    model = PDDShortcutModule(
        lr=cfg.training.lr,
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.model.action_dim,
        horizon=cfg.data.horizon,
        ema_decay=cfg.model.ema_decay,
        M=cfg.model.M,
        sc_ratio=cfg.training.sc_ratio,
        training_mode=cfg.training.mode,
    )

    # Load Teacher Checkpoint if provided (Stage 2)
    if cfg.model.get("teacher_ckpt") is not None:
        print(f"Loading teacher checkpoint from {cfg.model.teacher_ckpt}...")
        import torch
        ckpt = torch.load(cfg.model.teacher_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)

    # 3. Loggers & Callbacks
    # Explicitly use the Hydra output directory
    logger = pl.loggers.TensorBoardLogger(save_dir=output_dir, name="pdd_logs", version="")
    
    # Selection of monitor based on mode
    mode_prefix = "teacher_" if cfg.training.mode == "teacher_pretrain" else ""
    monitor_val = f"val/{mode_prefix}total_loss"

    checkpoint_cb = ModelCheckpoint(
        monitor=monitor_val,
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="pdd-{step:06d}-" + "{" + monitor_val + ":.4f}",
        save_top_k=cfg.training.save_top_k,
        mode="min",
    )
    # Monitor name updated in the class above
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. Trainer (Step-based)
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        max_epochs=-1,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        num_sanity_val_steps=2,
    )

    # 5. Train
    print(f"Starting PDD training for {cfg.training.max_steps} steps...")
    print(f"Logs: {os.path.join(output_dir, 'pdd_logs')}")
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
