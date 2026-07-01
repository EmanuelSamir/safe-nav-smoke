from typing import Literal, Optional, Union
import os

from pydantic import Field, model_validator

from src.utils.config_utils import StrictBaseModel
from src.env.schemas import EnvConfig
from src.agents.schemas import RobotConfig
from src.env.simulator.schemas import PlaybackConfig, SmokeConfig, GlobalSensorConfig
from src.controllers.schemas import MPPIConfig, CBFSmokeConfig

class IntegrationConfig(StrictBaseModel):
    """Configuration schema for the full simulation integration."""
    
    project_name: str = "single_agent_experiment"
    sub_project_name: str = "integration"
    
    experiment_mode: Literal["no_risk", "cbf", "persistent", "fno"]
    
    # Core components
    env: EnvConfig = Field(default_factory=EnvConfig)
    robot: RobotConfig
    sensor: GlobalSensorConfig = Field(default_factory=GlobalSensorConfig)
    simulator: PlaybackConfig
    
    # Controllers
    mppi: Optional[MPPIConfig] = None
    cbf: Optional[CBFSmokeConfig] = None
    
    # Neural Predictor (FNO)
    fno_checkpoint: Optional[str] = None
    fno_config: Optional[str] = None
    fno_cvar_alpha: float = 0.95
    
    @model_validator(mode="after")
    def validate_dependencies(self) -> "IntegrationConfig":
        """Asserts that all modes have their required parameters."""
        
        # 1. Controller Config Asserts
        if self.experiment_mode == "cbf":
            assert self.cbf is not None, "cbf config must be provided when experiment_mode='cbf'"
        else:
            assert self.mppi is not None, f"mppi config must be provided when experiment_mode='{self.experiment_mode}'"
            
        # 2. FNO Mode Asserts
        if self.experiment_mode == "fno":
            assert self.fno_checkpoint is not None, "fno_checkpoint must be provided when experiment_mode='fno'"
            assert self.fno_config is not None, "fno_config must be provided when experiment_mode='fno'"
            
            # Use absolute paths if possible, or assume they are relative to project root
            assert os.path.exists(self.fno_checkpoint) or os.path.exists(os.path.join(os.getcwd(), self.fno_checkpoint)), \
                f"FNO checkpoint not found: {self.fno_checkpoint}"
            assert os.path.exists(self.fno_config) or os.path.exists(os.path.join(os.getcwd(), self.fno_config)), \
                f"FNO config not found: {self.fno_config}"
                
        # 3. Environment Asserts
        assert self.env.num_agents == 1, "This integration project only supports num_agents=1"
        assert self.sensor.sensor_type == "global", "Only global sensor is supported in this integration"
        assert self.env.save_transitions is True, "save_transitions should be True to extract metrics later"
        
        return self
