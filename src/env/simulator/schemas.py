from typing import Optional, Union

from pydantic import Field
from typing_extensions import Literal

from src.utils.config_utils import StrictBaseModel


# Sensor schema
class BaseSensorConfig(StrictBaseModel):
    sensor_type: str = "base"


class GlobalSensorConfig(BaseSensorConfig):
    sensor_type: Literal["global"] = "global"
    density_reading_per_unit_length: float = 0.0


class DownwardsSensorConfig(BaseSensorConfig):
    sensor_type: Literal["downwards"] = "downwards"
    density_reading_per_unit_length: float = 0.0
    x_fov_size: float = 6.0
    y_fov_size: float = 6.0


class Camera1DSensorConfig(BaseSensorConfig):
    sensor_type: Literal["camera_1d"] = "camera_1d"
    fov_size_degrees: float = 90.0
    num_rays: int = 64
    step_size: float = 0.5
    opacity_threshold: float = 2.0
    max_range: float = 8.0


SensorConfigType = Union[GlobalSensorConfig, DownwardsSensorConfig, Camera1DSensorConfig]


# Simulation config
class BaseSimConfig(StrictBaseModel):
    # These fields represent the basic geometry and timing of the simulator.
    # They are optional because Playback config loads them from the dataset,
    # while Smoke config will provide them directly.
    resolution: Optional[float] = None
    dt: Optional[float] = None
    x_size: Optional[float] = None
    y_size: Optional[float] = None


class PlaybackConfig(BaseSimConfig):
    data_path: str


class BlobConfig(StrictBaseModel):
    x_pos: float = 10.0
    y_pos: float = 10.0
    intensity: float = 1.0
    spread_rate: float = 1.0


class SmokeConfig(BaseSimConfig):
    resolution: float = 0.2
    dt: float = 0.1
    x_size: float = 30.0
    y_size: float = 30.0
    # Physics and specific smoke generation parameters
    velocity_iterations: int = 4
    pressure_iterations: int = 20
    mac_cormack: bool = True
    buoyancy_alpha: float = 0.05
    buoyancy_beta: float = 0.5
    average_wind_speed: float = 5.0
    smoke_decay_rate: float = 1.5
    smoke_emission_rate: float = 1.8
    smoke_diffusion_rate: float = 0.0
    inflow_bank_count: int = 5
    buoyancy_factor: float = 1.2

    blobs: list[BlobConfig] = Field(default_factory=lambda: [BlobConfig()])
