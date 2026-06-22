from dataclasses import dataclass
from typing import Optional

from src.models.fno import FNOConfig
from src.models.conv_lstm import ConvLSTMConfig


@dataclass
class TrainingDataConfig:
    data_path: str
    batch_size: int
    train_split: float
    max_samples: Optional[int]
    sequence_length: int
    num_workers: int
    downsample_factor: int = 1


@dataclass
class TrainingLossConfig:
    name: Optional[str] = None  # None for ConvLSTM, string for FNO
    beta: Optional[float] = None
    m_samples: int = 10
    normalize_l2: bool = False


@dataclass
class TrainingOptimizerConfig:
    lr: float
    min_lr: float
    max_epochs: int
    grad_clip: float


@dataclass
class TrainingCheckpointConfig:
    save_top_k: int
    monitor: str
    mode: str


@dataclass
class TrainingVisualizerConfig:
    visualize_every: int


@dataclass
class FNOTrainingConfigSchema:
    experiment_name: str
    seed: int
    data: TrainingDataConfig
    model: FNOConfig
    loss: TrainingLossConfig
    optimizer: TrainingOptimizerConfig
    checkpoint: TrainingCheckpointConfig
    visualizer: TrainingVisualizerConfig
    test: bool = False

    def __post_init__(self):
        if self.model is not None and self.data is not None:
            self.model.sequence_length = self.data.sequence_length


@dataclass
class ConvLSTMTrainingConfigSchema:
    experiment_name: str
    seed: int
    data: TrainingDataConfig
    model: ConvLSTMConfig
    loss: TrainingLossConfig
    optimizer: TrainingOptimizerConfig
    checkpoint: TrainingCheckpointConfig
    visualizer: TrainingVisualizerConfig
    test: bool = False

    def __post_init__(self):
        if self.model is not None and self.data is not None:
            self.model.sequence_length = self.data.sequence_length
