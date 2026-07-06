from typing import List, Literal, Optional, Union
from typing_extensions import Annotated
from pydantic import Field, model_validator

from src.utils.config_utils import StrictBaseModel


# 1. Base Model Configurations
class ModelConfig(StrictBaseModel):
    h_ctx: int = 10
    h_pred: int = 5
    use_grid: bool = True
    use_time: bool = True
    min_std: float = 1e-4
    sequence_length: Optional[int] = 25
    is_probabilistic: bool = True


class FNOConfig(ModelConfig):
    type: Literal["fno"] = "fno"
    modes_t: int = 4
    modes_h: int = 8
    modes_w: int = 8
    width: int = 64
    n_layers: int = 4

    @model_validator(mode="after")
    def validate_modes(self):
        # if self.modes_t > self.h_ctx // 2:
        #     raise ValueError(f"modes_t ({self.modes_t}) must be <= h_ctx // 2 ({self.h_ctx // 2})")
        return self


class ConvLSTMConfig(ModelConfig):
    type: Literal["conv_lstm"] = "conv_lstm"
    hidden_dim: int = 32
    n_layers: int = 3
    kernel_size: int = 3

    @model_validator(mode="after")
    def validate_kernel(self):
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size ({self.kernel_size}) must be odd for symmetric padding.")
        return self


ModelConfigType = Annotated[
    Union[FNOConfig, ConvLSTMConfig], Field(discriminator="type")
]


# 2. Training Configurations
class TrainingDataConfig(StrictBaseModel):
    data_path: str = "data/physics_smoke"
    batch_size: int = 16
    train_split: float = 0.9
    max_samples: Optional[int] = None
    sequence_length: int = 30
    num_workers: int = 0
    downsample_factor: int = 1


class TrainingLossConfig(StrictBaseModel):
    name: Optional[str] = "nll"  # None for ConvLSTM, string for FNO (e.g. 'nll', 'energy_score', 'mse', 'mae')
    beta: Optional[float] = None
    m_samples: int = 5
    normalize_l2: bool = True


class TrainingOptimizerConfig(StrictBaseModel):
    lr: float = 1.0e-3
    min_lr: float = 1.0e-4
    max_epochs: int = 500
    grad_clip: float = 1.0


class TrainingCheckpointConfig(StrictBaseModel):
    save_top_k: int = 3
    monitor: str = "val_nll"
    mode: str = "min"


class TrainingVisualizerConfig(StrictBaseModel):
    visualize_every: int = 5
    rollout_steps: List[int] = [1, 5, 10, 15]


class TrainingConfig(StrictBaseModel):
    project_name: str = "single_agent_experiment"
    sub_project_name: str = "training"
    experiment_name: str
    seed: int = 42
    data: TrainingDataConfig = TrainingDataConfig()
    loss: TrainingLossConfig = TrainingLossConfig()
    optimizer: TrainingOptimizerConfig = TrainingOptimizerConfig()
    checkpoint: TrainingCheckpointConfig = TrainingCheckpointConfig()
    visualizer: TrainingVisualizerConfig = TrainingVisualizerConfig()
    test: bool = False
    model: ModelConfigType

    @model_validator(mode="after")
    def sync_sequence_length(self):
        if self.model is not None and self.data is not None:
            self.model.sequence_length = self.data.sequence_length
        return self


class FNOTrainingConfig(TrainingConfig):
    experiment_name: str = "fno"
    model: FNOConfig = FNOConfig(type="fno")


class ConvLSTMTrainingConfig(TrainingConfig):
    experiment_name: str = "conv_lstm"
    data: TrainingDataConfig = TrainingDataConfig(batch_size=8)
    model: ConvLSTMConfig = ConvLSTMConfig(type="conv_lstm")
    loss: TrainingLossConfig = TrainingLossConfig(name=None)
    visualizer: TrainingVisualizerConfig = TrainingVisualizerConfig(visualize_every=10)
