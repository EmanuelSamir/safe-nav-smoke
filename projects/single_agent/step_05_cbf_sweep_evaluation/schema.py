from typing import Literal, List
from pydantic import Field, model_validator
from src.utils.config_utils import StrictBaseModel
from src.env.schemas import EnvConfig
from src.agents.schemas import RobotConfig
from src.env.simulator.schemas import PlaybackConfig, GlobalSensorConfig
from src.controllers.schemas import CBFSmokeConfig

class SweepConfig(StrictBaseModel):
    """Configuration schema for parameter sweeps."""
    
    env: EnvConfig = Field(default_factory=EnvConfig)
    robot: RobotConfig
    sensor: GlobalSensorConfig = Field(default_factory=GlobalSensorConfig)
    simulator: PlaybackConfig
    
    cbf_base: CBFSmokeConfig
    
    sweep_param: Literal["smoke_threshold", "k1", "k2", "rho", "margin", "gamma"]
    sweep_values: List[float]
    
    @model_validator(mode="after")
    def validate_sweep(self) -> "SweepConfig":
        assert len(self.sweep_values) > 0, "sweep_values must not be empty"
        assert hasattr(self.cbf_base, self.sweep_param), f"cbf_base has no parameter '{self.sweep_param}'"
        return self
