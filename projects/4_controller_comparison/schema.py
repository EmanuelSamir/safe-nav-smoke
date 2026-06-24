from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """
    TODO(FUTURE MIGRATION):
    Currently, these Pydantic schemas live in the project folder as a "validation gate"
    (Option 1) because the core modules in `src/` still use standard Python `@dataclass`.
    When the full migration to Pydantic is completed for the entire repository,
    these schemas (AgentConfig, EnvConfig, SensorConfig, etc.) should be moved directly
    into their respective modules (e.g., `src/env/smoke_env.py`), replacing the old
    dataclasses. Once that is done, this `schema.py` should only contain `BenchmarkConfig`
    as a Composition Root that imports the base schemas from `src/`.
    """
    model_config = ConfigDict(extra="forbid")


class AgentConfig(StrictBaseModel):
    action_min: List[float]
    action_max: List[float]
    action_dim: int
    state_dim: int


class EnvConfig(StrictBaseModel):
    num_agents: int
    world_x_size: float
    world_y_size: float
    clock: float
    max_steps: int
    collision_radius: float
    goal_radius: float
    initial_locations: List[List[float]]
    goal_locations: List[List[float]]
    render: str = "none"
    render_save_every: int = 1
    terminate_on_collision: bool = False
    collision_penalty: float = 10.0
    smoke_density_threshold: float = 0.5
    save_transitions: bool = False


class SensorConfig(StrictBaseModel):
    sensor_type: str = "global"
    density_reading_per_unit_length: float = 0.0


class PlaybackConfig(StrictBaseModel):
    # Esto fuerza estructuralmente a que SOLO se use Playback en este benchmark
    data_path: str


class MPPIConfig(StrictBaseModel):
    horizon: int
    num_samples: int
    lambda_: float
    device: str = "cpu"
    alpha_noise_sigma: float = 10.0
    noise_abs_cost: bool = False


class SafetyConfig(StrictBaseModel):
    d_safe: float
    r_sense: float


class SolverConfig(StrictBaseModel):
    safe_margin: float
    domain_cells: List[int]
    domain: List[List[float]]
    accuracy: str
    target_time: float
    dt: float
    epsilon: float


class NominalControllerConfig(StrictBaseModel):
    type: Literal["nominal"]
    mppi: MPPIConfig


class CBFControllerConfig(StrictBaseModel):
    type: Literal["cbf"]
    mode: str
    dt: float
    k1: float
    k2: float
    rho: float
    L: float
    safety: SafetyConfig
    mppi: MPPIConfig


class HJControllerConfig(StrictBaseModel):
    type: Literal["hj"]
    mode: str
    dt: float
    action_min: List[float]
    action_max: List[float]
    safety: SafetyConfig
    mppi: MPPIConfig
    solver: SolverConfig


ControllerConfigType = Union[NominalControllerConfig, CBFControllerConfig, HJControllerConfig]


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
    
    # Un diccionario con los controladores a evaluar. 
    # El YAML contendrá directamente sus parámetros específicos.
    controllers: Dict[str, ControllerConfigType]
