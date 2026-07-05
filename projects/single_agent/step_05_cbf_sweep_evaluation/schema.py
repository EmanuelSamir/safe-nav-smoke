from typing import Literal, List, Union
from pydantic import Field, model_validator
from src.utils.config_utils import StrictBaseModel
from src.env.schemas import EnvConfig
from src.agents.schemas import DubinsConfig
from src.env.simulator.schemas import PlaybackConfig, GlobalSensorConfig, SmokeConfig
from src.controllers.schemas import CBFSmokeConfig

class SweepConfig(StrictBaseModel):
    """Configuration schema for parameter sweeps."""
    
    project_name: str = "single_agent_experiment"
    sub_project_name: str = "cbf_sweep"
    test_mode: bool = False
    
    env: EnvConfig = Field(default_factory=EnvConfig)
    robot: DubinsConfig = Field(default_factory=DubinsConfig)
    sensor: GlobalSensorConfig = Field(default_factory=GlobalSensorConfig)
    simulator: Union[PlaybackConfig, SmokeConfig] = Field(default_factory=SmokeConfig)
    
    cbf_base: CBFSmokeConfig
    
    sweep_param: Literal["smoke_threshold", "k1", "k2", "rho", "discrete_epsilon", "gamma"]
    sweep_values: List[float]
    
    @model_validator(mode="after")
    def validate_sweep(self) -> "SweepConfig":
        assert len(self.sweep_values) > 0, "sweep_values must not be empty"
        assert hasattr(self.cbf_base, self.sweep_param), f"cbf_base has no parameter '{self.sweep_param}'"
        return self
