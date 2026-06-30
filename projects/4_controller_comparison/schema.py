from typing import Dict, List, Optional

from pydantic import ConfigDict, Field

from typing import Any, Dict, List, Optional, Union
from src.agents.schemas import RobotConfig as AgentConfig
from src.utils.config_utils import StrictBaseModel
from src.env.schemas import EnvConfig
from src.env.simulator.schemas import PlaybackConfig, SensorConfigType as SensorConfig

from typing import Annotated, Literal

class BenchmarkMPPIConfig(StrictBaseModel):
    num_samples: int
    horizon: int
    alpha_noise_sigma: float = 10.0
    noise_abs_cost: bool = False
    lambda_: float = 1.0
    device: str = "cpu"

class BenchmarkSafetyConfig(StrictBaseModel):
    safe_margin: float
    r_sense: float

class BenchmarkSolverConfig(StrictBaseModel):
    interaction_radius: float
    spatial_resolution: Union[float, List[float]]
    angular_cells: int
    target_time: float
    dt: float
    superlevel_set_epsilon: float
    safe_margin: float

class BenchmarkNominalConfig(StrictBaseModel):
    type: Literal["nominal"]
    mppi: BenchmarkMPPIConfig

class BenchmarkCBFConfig(StrictBaseModel):
    type: Literal["cbf"]
    mode: Optional[str] = None
    dt: Optional[float] = None
    k1: Optional[float] = None
    k2: Optional[float] = None
    rho: Optional[float] = None
    L: Optional[float] = None
    safety: Optional[BenchmarkSafetyConfig] = None
    mppi: BenchmarkMPPIConfig

class BenchmarkHJConfig(StrictBaseModel):
    type: Literal["hj"]
    mode: Optional[str] = None
    control_type: Optional[str] = None
    dt: Optional[float] = None
    action_min: Optional[List[float]] = None
    action_max: Optional[List[float]] = None
    safety: Optional[BenchmarkSafetyConfig] = None
    solver: Optional[BenchmarkSolverConfig] = None
    mppi: BenchmarkMPPIConfig

ControllerConfigType = Annotated[Union[BenchmarkNominalConfig, BenchmarkCBFConfig, BenchmarkHJConfig], Field(discriminator="type")]

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

    agent: AgentConfig
    env: EnvConfig
    sensor: SensorConfig
    playback: PlaybackConfig
    run: RunConfig = Field(default_factory=RunConfig)
    
    controllers: Dict[str, ControllerConfigType]
    # El YAML contendrá directamente sus parámetros específicos.
    controllers: Dict[str, ControllerConfigType] = {}
