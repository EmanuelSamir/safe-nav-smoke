"""Pydantic Structured Config schemas for all multi-agent controllers.

Each dataclass is registered and validated automatically by Pydantic.
Fields without a default are required.
"""

from typing import List, Literal, Union

from pydantic import Field

from src.controllers.base.schemas import (
    AbsoluteHJSolverConfig,
    MPPIConfig,
    RelativeHJSolverConfig,
)
from src.utils.config_utils import StrictBaseModel


class SharedSafetyConfig(StrictBaseModel):
    """Safety contract shared across all controllers in an experiment.

    Making this a separate node ensures d_safe, r_sense, and neighbor_mode
    are set once per experiment, not independently per controller.
    """

    robot_radius: float = 0.4
    safe_margin: float = 0.2
    r_sense: float = 8.0
    neighbor_mode: Literal["nearest", "all"] = "nearest"


SafetyMode = Literal["filter", "dual-guard", "online_dual-guard", "penalty"]


class BaseMultiAgentConfig(StrictBaseModel):
    """Base configuration for multi-agent controllers."""

    safety: SharedSafetyConfig = Field(default_factory=SharedSafetyConfig)
    mppi: MPPIConfig
    mode: SafetyMode = "filter"
    dt: float = 0.1


class CBFSmokeConfig(StrictBaseModel):
    """Full configuration for the single-agent CBF smoke controller (CBFSmokeController)."""

    smoke_threshold: float = 0.75
    k1: float = 5.0
    k2: float = 5.0
    rho: float = 5.0
    margin: float = 1.0
    gamma: float = 1.0
    R_diag: List[float] = [1.0, 1.0]


class MultiAgentCBFConfig(BaseMultiAgentConfig):
    """Full configuration for ``MultiAgentCBFController``."""

    k1: float = 1.5
    k2: float = 1.5
    rho: float = 5.0


class MultiAgentHJConfig(BaseMultiAgentConfig):
    """Full configuration for ``MultiAgentHJController``."""

    solver: Union[AbsoluteHJSolverConfig, RelativeHJSolverConfig] = Field(discriminator="mode")
    action_min: List[float] = [0.0, -4.0]
    action_max: List[float] = [6.0, 4.0]
    control_type: Literal["smooth", "bang_bang"] = "smooth"
