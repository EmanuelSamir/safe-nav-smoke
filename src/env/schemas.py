from typing import List, Optional, Union

from typing_extensions import Literal

from src.utils.config_utils import StrictBaseModel


class EnvConfig(StrictBaseModel):
    num_agents: int = 6
    world_x_size: float = 30.0
    world_y_size: float = 30.0
    clock: float = 0.1
    max_steps: int = 200
    collision_radius: float = 0.8
    goal_radius: float = 0.5
    initial_locations: Optional[List[List[float]]] = None
    goal_locations: Optional[List[List[float]]] = None
    render: Literal["none", "rgb_array", "human"] = "none"
    render_save_every: int = 2
    terminate_on_collision: bool = False
    collision_penalty: float = -10.0
    smoke_density_threshold: Optional[float] = 0.5
    save_transitions: bool = False

    # Extra parameters in smoke_env.yaml configuration
    test: Optional[bool] = False
    num_episodes: Optional[int] = 1
    save_transitions_path: Optional[str] = None


class BaseSensorConfig(StrictBaseModel):
    sensor_type: str = "base"
    world_x_size: Optional[float] = None
    world_y_size: Optional[float] = None


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


class PlaybackConfig(StrictBaseModel):
    data_path: str


class SmokeParams(StrictBaseModel):
    # Defining fields that are used in SmokeParams based on the codebase
    resolution: float = 0.2
    dt: float = 0.1
    x_size: float = 30.0
    y_size: float = 30.0
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


class BlobParams(StrictBaseModel):
    x_pos: float = 10.0
    y_pos: float = 10.0
    intensity: float = 1.0
    spread_rate: float = 1.0
