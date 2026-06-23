"""Hydra Structured Config schemas for all multi-agent controllers.

Each dataclass is registered in ``src/__init__.py`` and validated automatically
by Hydra — fields without a default are required; if the YAML omits them,
Hydra raises a MissingMandatoryValue error at startup.

Design rules:
- No ``MISSING`` sentinel needed — Hydra Structured Configs handle it.
- ``action_min`` / ``action_max`` are ``list[float]`` (Hydra-serializable).
  Controllers convert them to tensors internally.
- Shared fields (d_safe, r_sense, neighbor_mode) live in ``SharedSafetyConfig``
  and are composed via Hydra's defaults list so they remain a single source
  of truth across experiments.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared safety parameters
# ---------------------------------------------------------------------------


@dataclass
class SharedSafetyConfig:
    """Safety contract shared across all controllers in an experiment.

    Making this a separate node ensures d_safe, r_sense, and neighbor_mode
    are set once per experiment, not independently per controller.

    Attributes:
        d_safe:        Minimum centre-to-centre clearance distance (m).
                       Used directly by CBF and as the BRT initial condition by HJ.
        r_sense:       Sensing radius — neighbours beyond this are excluded (m).
        neighbor_mode: Which neighbours enter the safety computation.
                       ``"nearest"`` — only the most dangerous (worst-case single neighbour).
                       ``"all"``     — all neighbours simultaneously (CBF only;
                                       HJ raises ValueError if set to "all").
    """

    d_safe: float
    r_sense: float
    neighbor_mode: str  # "nearest" | "all"


# ---------------------------------------------------------------------------
# MPPI parameters
# ---------------------------------------------------------------------------


@dataclass
class MPPIConfig:
    """MPPI planner parameters.

    Attributes:
        num_samples: Number of sampled trajectories K.
        horizon:     Planning horizon T (steps).
        lambda_:     Temperature — controls exploration/exploitation.
        device:      PyTorch device string (``"cpu"`` or ``"cuda"``).
    """

    num_samples: int
    horizon: int
    lambda_: float
    device: str


# ---------------------------------------------------------------------------
# HJ BRT solver parameters
# ---------------------------------------------------------------------------


@dataclass
class HJSolverConfig:
    """Configuration for the Hamilton-Jacobi BRT solver.

    Attributes:
        domain:       Grid bounds ``[[x_min, y_min, θ_min], [x_max, y_max, θ_max]]``.
        domain_cells: Grid resolution ``[Gx, Gy, Gθ]``.
        accuracy:     Solver accuracy level — ``"low"``, ``"medium"``, ``"high"``, ``"very_high"``.
        target_time:  BRT integration horizon (negative = backwards in time, s).
        dt:           Solver integration timestep (s).
        epsilon:      Early-stop convergence threshold on ΔV.
        dx:           Spatial resolution for signed-distance initialisation (m).
        safe_margin:  V < safe_margin ⟹ unsafe. Typically ``0.0`` (exact zero-level set).
    """

    domain: list
    domain_cells: list
    accuracy: str
    target_time: float
    dt: float
    epsilon: float
    dx: float
    safe_margin: float


@dataclass
class CBFSmokeConfig:
    """Full configuration for the single-agent CBF smoke controller (CBFSmokeController).

    Attributes:
        smoke_threshold: Smoke concentration threshold above which motion is penalised.
        k1:              Class-K gain for the 1st-order CBF constraint.
        k2:              Class-K gain for the 2nd-order CBF constraint.
        rho:             Penalty weight used in ADMM solver.
        margin:          Safety margin subtracted from the signed distance (m).
        R_diag:          Diagonal coefficients of the nominal control penalty matrix.
    """

    smoke_threshold: float
    k1: float
    k2: float
    rho: float
    margin: float
    R_diag: list[float]


# ---------------------------------------------------------------------------
# CBF controller config
# ---------------------------------------------------------------------------


@dataclass
class MultiAgentCBFConfig:
    """Full configuration for ``MultiAgentCBFController``.

    Attributes:
        safety:  Shared safety contract (composed from ``controller/safety`` group).
        mppi:    MPPI planner parameters (composed from ``controller/mppi`` group).
        mode:    Safety integration mode.
                 ``"filter"``  — QP projection at output step only.
                 ``"rollout"`` — DualGuard: projection during rollout + output.
                 ``"penalty"`` — Soft barrier penalty in the running cost.
        k1:      Class-K gain for the 1st-order CBF constraint.
        k2:      Class-K gain for the 2nd-order CBF constraint (reserved).
        L:       Longitudinal offset from vehicle centre to safety point (m).
        rho:     Penalty weight used in ``"penalty"`` mode.
        dt:      Timestep (referenced via interpolation).
    """

    safety: SharedSafetyConfig
    mppi: MPPIConfig

    mode: str  # "filter" | "rollout" | "penalty"

    # CBF-specific
    k1: float
    k2: float
    L: float
    rho: float
    dt: float


# ---------------------------------------------------------------------------
# HJ controller config
# ---------------------------------------------------------------------------


@dataclass
class MultiAgentHJConfig:
    """Full configuration for ``MultiAgentHJController``.

    Attributes:
        safety:  Shared safety contract (composed from ``controller/safety`` group).
                 ``neighbor_mode`` must be ``"nearest"`` — HJ only supports the
                 single most-dangerous neighbour.
        mppi:    MPPI planner parameters (composed from ``controller/mppi`` group).
        solver:  HJ BRT solver configuration (composed from ``controller/hj_solver`` group).
        mode:    Safety integration mode.
                 ``"filter"``         — LRF projection at output step only.
                 ``"rollout"``        — DualGuard: projection during rollout + output.
                 ``"online_rollout"`` — DualGuard with BRT re-solved every planning step.
                 ``"penalty"``        — Soft barrier penalty in the running cost.
        dt:      Timestep (referenced via interpolation).
        action_min: Minimum control bounds (referenced via interpolation).
        action_max: Maximum control bounds (referenced via interpolation).
    """

    safety: SharedSafetyConfig
    mppi: MPPIConfig
    solver: HJSolverConfig

    mode: str  # "filter" | "rollout" | "online_rollout" | "penalty"
    dt: float
    action_min: list[float]
    action_max: list[float]
