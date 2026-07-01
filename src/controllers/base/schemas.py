from typing import List, Literal, Optional

import numpy as np
import torch
from pydantic import ConfigDict
from typing import Any

from src.utils.config_utils import DeviceType, StrictBaseModel


class BaseFilterConfig(StrictBaseModel):
    """Base parameters for safety filters."""

    robot_radius: float = 0.4
    safe_margin: float = 0.2
    r_sense: float = 8.0
    dt: float = 0.1


class CBFFilterConfig(BaseFilterConfig):
    """Parameters for the High-Order Control Barrier Function (HOCBF) filter."""

    k1: float = 1.5
    k2: float = 1.5
    smoke_threshold: float = 0.75
    rho: float = 5.0


class HJFilterConfig(BaseFilterConfig):
    """Parameters for HJ Reachability / LRF projection."""

    action_min: List[float] = [0.0, -4.0]
    action_max: List[float] = [6.0, 4.0]
    control_type: Literal["smooth", "bang_bang"] = "smooth"


class MPPIConfig(StrictBaseModel):
    """Configuration parameters for the MPPI Controller."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nx: int
    noise_sigma: Optional[List[List[float]]] = None
    alpha_noise_sigma: Optional[float] = None
    num_samples: int = 100
    horizon: int = 10
    device: DeviceType = "cpu"
    lambda_: float = 3.0
    noise_mu: Optional[List[float]] = None
    u_min: Optional[List[float]] = None
    u_max: Optional[List[float]] = None
    u_init: Optional[List[float]] = None
    u_scale: float = 1.0
    u_per_command: int = 1
    step_dependent_dynamics: bool = True
    noise_abs_cost: bool = False
    cost_distance_weight: float = 1.0
    cost_risk_weight: float = 20.0
    cost_goal_reached: float = -100.0


class BaseHJSolverConfig(StrictBaseModel):
    """Base configuration for the Hamilton-Jacobi BRT solver."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    accuracy: Literal["low", "medium", "high", "very_high"] = "medium"
    target_time: float = -10.0
    dt: float = 0.05
    epsilon: float = 0.01
    dx: float = 0.1
    superlevel_set_epsilon: float = 0.0
    until_convergent: bool = True
    print_progress: bool = True
    safe_margin: float = 0.0


class AbsoluteHJSolverConfig(BaseHJSolverConfig):
    """Solver configuration for static obstacle avoidance maps."""
    mode: Literal["absolute"] = "absolute"
    map_width: float = 20.0
    map_height: float = 20.0
    spatial_resolution: float = 0.5
    angular_cells: int = 36


class RelativeHJSolverConfig(BaseHJSolverConfig):
    """Solver configuration for dynamic agent avoidance."""
    mode: Literal["relative"] = "relative"
    interaction_radius: float = 8.0
    spatial_resolution: float = 0.5
    angular_cells: int = 36
