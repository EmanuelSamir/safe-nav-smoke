from typing import List

from src.utils.config_utils import StrictBaseModel


class DataCollectionConfig(StrictBaseModel):
    num_episodes: int = 500
    episode_steps: int = 100
    output_path: str = "data/physics_smoke_slow"

    x_size: float = 35.0
    y_size: float = 35.0
    resolution: float = 0.2
    dt: float = 0.1

    wind_speed: float = 5.0
    emission_rate: float = 1.8
    diffusion_rate: float = 0.0
    decay_rate: float = 1.5
    buoyancy: float = 1.2

    test: bool = False


class PhysicsCollectionConfig(DataCollectionConfig):
    num_blobs_range: List[int] = [4, 8]
    blob_min_dist: float = 5.0
    blob_spread_range: List[float] = [1.0, 3.0]
    blob_intensity: float = 1.0
