from pydantic import Field
from src.utils.config_utils import StrictBaseModel
from typing import List

class ModelEvalConfig(StrictBaseModel):
    name: str
    checkpoint_path: str
    config_path: str  # Path to the config used during training

class EvaluationConfig(StrictBaseModel):
    project_name: str = "single_agent_experiment"
    sub_project_name: str = "saved_rollouts"
    data_path: str = "data/structured_smoke_slow"
    max_episodes: int = 100
    
    # Dataset params
    sequence_length: int = 30
    forecast_horizon: int = 20
    mode: str = "test"
    dense: bool = True
    
    # Analysis params
    max_horizon_eval: int = 20

    models: List[ModelEvalConfig] = Field(default_factory=list)
