from typing import List
from pydantic import Field

from src.utils.config_utils import StrictBaseModel
from src.env.simulator.schemas import SmokeConfig

class DataCollectionConfig(StrictBaseModel):
    num_episodes: int = 500
    episode_steps: int = 100
    output_path: str = "data/physics_smoke_slow"

    smoke_params: SmokeConfig = Field(default_factory=SmokeConfig)

    test: bool = False

    num_blobs_range: List[int] = [4, 8]
    blob_min_dist: float = 5.0
    blob_spread_range: List[float] = [1.0, 3.0]
    blob_intensity: float = 1.0
    
    max_spawn_attempts: int = 100
    spawn_margin: float = 2.0

    writer_batch_size: int = 50
