from hydra.core.config_store import ConfigStore

from src.agents.basic_robot import RobotParams

# Import your configuration schemas (dataclasses) here:
from src.env.smoke_env import EnvConfig
from src.env.simulator.smoke import SmokeParams
from src.env.simulator.playback import PlaybackParams
from src.env.simulator.sensor import (
    GlobalSensorParams,
    DownwardsSensorParams,
    Camera1DSensorParams,
)
from src.models.conv_lstm import ConvLSTMConfig
from src.models.fno import FNOConfig
from src.training.schemas import ConvLSTMTrainingConfigSchema, FNOTrainingConfigSchema
from src.wrappers.smoke_forecast_wrapper import SmokeForecastWrapperConfig

# ==============================================================================
# HOW TO REGISTER NEW CONFIGURATION SCHEMAS (DATACLASSES):
# ==============================================================================
# When you add a new module or component (e.g. controllers, new simulators, etc.)
# that has its own configuration schema dataclass:
#
# 1. Import the configuration schema dataclass at the top of this file.
#    Example: `from src.controllers.my_controller import MyControllerConfig`
#
# 2. Register it below using the `cs.store()` method:
#    Example: `cs.store(group="controller", name="my_controller", node=MyControllerConfig)`
#
#    Note:
#    - 'group' corresponds to the subfolder/category name in your configs.
#    - 'name' corresponds to the YAML filename (without .yaml) or the config name.
#    - 'node' is the dataclass that acts as the validation schema.
# ==============================================================================

cs = ConfigStore.instance()

# Register Environment schema
cs.store(group="env", name="smoke_env", node=EnvConfig)

# Register Agent/Robot schema
cs.store(group="agent", name="dubins", node=RobotParams)

# Register Simulator/Smoke schema
cs.store(group="env/simulator", name="smoke", node=SmokeParams)
cs.store(group="env/simulator", name="playback", node=PlaybackParams)

# Register Sensor schemas
cs.store(group="env/sensors", name="global", node=GlobalSensorParams)
cs.store(group="env/sensors", name="downwards", node=DownwardsSensorParams)
cs.store(group="env/sensors", name="camera1d", node=Camera1DSensorParams)

# Register Model schemas
cs.store(group="models", name="conv_lstm", node=ConvLSTMConfig)
cs.store(group="models", name="fno", node=FNOConfig)

# Register Training schemas
cs.store(group="training", name="conv_lstm", node=ConvLSTMTrainingConfigSchema)
cs.store(group="training", name="fno", node=FNOTrainingConfigSchema)

# Register Wrapper schemas
cs.store(group="wrappers", name="smoke_forecast", node=SmokeForecastWrapperConfig)
