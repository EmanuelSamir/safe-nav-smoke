import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

from projects.single_agent.step_04_evaluation.schema import EvaluationConfig
from src.models.lightning_conv_lstm import ConvLSTMLightningModule
from src.models.lightning_fno import FNOLightningModule
from src.models.shared.datasets import SequentialDataset
from src.models.shared.base_lightning import make_times
from src.models.shared.schemas import (
    TrainingConfig, FNOTrainingConfig, ConvLSTMTrainingConfig,
    TrainingDataConfig, TrainingLossConfig, TrainingOptimizerConfig, 
    TrainingCheckpointConfig, TrainingVisualizerConfig,
    ModelConfig, FNOConfig, ConvLSTMConfig
)

torch.serialization.add_safe_globals([
    TrainingConfig, FNOTrainingConfig, ConvLSTMTrainingConfig,
    TrainingDataConfig, TrainingLossConfig, TrainingOptimizerConfig, 
    TrainingCheckpointConfig, TrainingVisualizerConfig,
    ModelConfig, FNOConfig, ConvLSTMConfig
])

def load_config() -> EvaluationConfig:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f) or {}
    return EvaluationConfig.model_validate(yaml_data)

def generate_rollouts():
    cfg = load_config()
    
    output_root = Path(os.getcwd()) / "outputs" / cfg.project_name / cfg.sub_project_name
    output_root.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset loading
    print(f"Loading dataset from {cfg.data_path}")
    dataset = SequentialDataset(
        data_path=cfg.data_path,
        sequence_length=cfg.sequence_length,
        forecast_horizon=cfg.forecast_horizon,
        mode=cfg.mode,
        max_episodes=cfg.max_episodes,
        dense=cfg.dense
    )
    
    for model_cfg in cfg.models:
        print(f"\nEvaluating model: {model_cfg.name}")
        ckpt_path = Path(os.getcwd()) / model_cfg.checkpoint_path
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue
            
        with open(Path(os.getcwd()) / model_cfg.config_path, "r") as f:
            t_cfg_data = yaml.safe_load(f)
            t_cfg = TrainingConfig.model_validate(t_cfg_data)
        
        # Load lightning module based on config type
        if t_cfg.model.type == "conv_lstm":
            model = ConvLSTMLightningModule.load_from_checkpoint(
                ckpt_path, t_cfg=t_cfg, H=dataset.H, W=dataset.W, x_size=dataset.x_size, y_size=dataset.y_size
            )
        else:
            model = FNOLightningModule.load_from_checkpoint(
                ckpt_path, t_cfg=t_cfg, H=dataset.H, W=dataset.W, x_size=dataset.x_size, y_size=dataset.y_size
            )
            
        model.to(device)
        model.eval()
        
        model_out_dir = output_root / model_cfg.name
        model_out_dir.mkdir(exist_ok=True)
        
        h_ctx = t_cfg.model.h_ctx
        max_horizon = cfg.max_horizon_eval
        
        # Generate predictions per episode
        for ep_idx in tqdm(range(len(dataset)), desc=f"Rollouts for {model_cfg.name}"):
            # Obtener el episodio completo directamente
            episode_data = dataset.smoke_data[ep_idx] # (T, H, W)
            frames = torch.from_numpy(episode_data).unsqueeze(0).to(device) # (1, T, H, W)
            T = frames.shape[1]
            
            # Find all possible starting points for the rollout
            time_steps = []
            means = []
            stds = []
            
            # Predict for each valid timestep
            with torch.no_grad():
                for t_idx in range(h_ctx - 1, T - max_horizon):
                    ctx_w = frames[:, t_idx - h_ctx + 1 : t_idx + 1] # (1, h_ctx, H, W)
                    
                    # Autoregressive rollout!
                    preds = model.model.autoregressive_forecast(
                        ctx_w, seed_t_start=0, horizon=max_horizon, num_samples=1, mode="mean"
                    )
                    
                    # Store means and stds for this timestep
                    t_means = np.stack([p["mean"][0] for p in preds]) # (H_max, H, W)
                    t_stds = np.stack([p.get("std", np.zeros_like(p["mean"]))[0] for p in preds]) # (H_max, H, W)
                    
                    time_steps.append(t_idx)
                    means.append(t_means)
                    stds.append(t_stds)
            
            if len(time_steps) == 0:
                continue
                
            np.savez_compressed(
                model_out_dir / f"ep_{ep_idx:04d}.npz",
                gt_full=frames[0].cpu().numpy(),
                time_steps=np.array(time_steps),
                mean=np.array(means),     # (N_times, H_max, H, W)
                std=np.array(stds),       # (N_times, H_max, H, W)
                sample=np.expand_dims(np.array(means), axis=2) # Dummy sample dim for script compatibility: (N_times, H_max, 1, H, W)
            )

if __name__ == "__main__":
    generate_rollouts()
