from typing import Dict, List, Optional

from pydantic import ConfigDict

from src.agents.schemas import RobotParams as AgentConfig
from src.controllers.schemas import ControllerConfigType
from src.env.schemas import EnvConfig, PlaybackConfig, SensorConfigType as SensorConfig
from src.utils.config_utils import StrictBaseModel

class RunConfig(StrictBaseModel):
    episodes: int = 100
    steps: Optional[int] = None
    device: Optional[str] = None
    controllers: Optional[List[str]] = None
    output_dir: Optional[str] = None
    render: str = "none"
    test: bool = False


class BenchmarkConfig(StrictBaseModel):
    """
    Unified strict schema for the benchmark.
    Removes redundancy by defining env, agent, and simulator ONCE.
    Forces playback (no physics simulation allowed).
    """

    agent: AgentConfig = AgentConfig()
    env: EnvConfig = EnvConfig()
    sensor: SensorConfig
    playback: PlaybackConfig
    run: RunConfig = RunConfig()
    
    # Un diccionario con los controladores a evaluar. 
    # El YAML contendrá directamente sus parámetros específicos.
    controllers: Dict[str, ControllerConfigType] = {}
