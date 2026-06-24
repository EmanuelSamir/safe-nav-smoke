from typing import Optional
from pydantic import model_validator
from src.utils.config_utils import StrictBaseModel

from src.models.fno import FNOConfig
from src.models.conv_lstm import ConvLSTMConfig


class TrainingDataConfig(StrictBaseModel):
    data_path: str = "data/physics_smoke"
    batch_size: int = 16
    train_split: float = 0.9
    max_samples: Optional[int] = None
    sequence_length: int = 30
    num_workers: int = 0
    downsample_factor: int = 1


class TrainingLossConfig(StrictBaseModel):
    name: Optional[str] = "nll"  # None for ConvLSTM, string for FNO
    beta: Optional[float] = None
    m_samples: int = 5
    normalize_l2: bool = True


class TrainingOptimizerConfig(StrictBaseModel):
    lr: float = 1.0e-3
    min_lr: float = 1.0e-4
    max_epochs: int = 250
    grad_clip: float = 1.0


class TrainingCheckpointConfig(StrictBaseModel):
    save_top_k: int = 3
    monitor: str = "val_nll"
    mode: str = "min"


class TrainingVisualizerConfig(StrictBaseModel):
    visualize_every: int = 5


class FNOTrainingConfig(StrictBaseModel):
    experiment_name: str = "fno"
    seed: int = 42
    data: TrainingDataConfig = TrainingDataConfig()
    model: FNOConfig = FNOConfig()
    loss: TrainingLossConfig = TrainingLossConfig()
    optimizer: TrainingOptimizerConfig = TrainingOptimizerConfig()
    checkpoint: TrainingCheckpointConfig = TrainingCheckpointConfig()
    visualizer: TrainingVisualizerConfig = TrainingVisualizerConfig()
    test: bool = False

    @model_validator(mode="after")
    def sync_sequence_length(self):
        if self.model is not None and self.data is not None:
            self.model.sequence_length = self.data.sequence_length
        return self


class ConvLSTMTrainingConfig(StrictBaseModel):
    experiment_name: str = "conv_lstm"
    seed: int = 42
    data: TrainingDataConfig = TrainingDataConfig(batch_size=8)
    model: ConvLSTMConfig = ConvLSTMConfig()
    loss: TrainingLossConfig = TrainingLossConfig(name=None)
    optimizer: TrainingOptimizerConfig = TrainingOptimizerConfig()
    checkpoint: TrainingCheckpointConfig = TrainingCheckpointConfig()
    visualizer: TrainingVisualizerConfig = TrainingVisualizerConfig(visualize_every=10)
    test: bool = False

    @model_validator(mode="after")
    def sync_sequence_length(self):
        if self.model is not None and self.data is not None:
            self.model.sequence_length = self.data.sequence_length
        return self
