from typing import List
from pydantic import Field

from src.utils.config_utils import StrictBaseModel
from src.env.simulator.schemas import SmokeConfig

class DataCollectionConfig(StrictBaseModel):
    num_episodes: int = 100
    episode_steps: int = 200
    output_path: str = "data/structured_smoke_slow"

    smoke_params: SmokeConfig = Field(default_factory=SmokeConfig)

    test: bool = False

    blob_spread_range: List[float] = [1.5, 3.0]
    blob_intensity: float = 1.0

    writer_batch_size: int = 50
