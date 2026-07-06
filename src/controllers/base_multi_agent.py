"""Base multi-agent MPPI controller with pluggable safety modes.

Provides:
- ``AgentMPPI``               — single-agent planner with risk-map cost.
- ``BaseMultiAgentController``— centralized orchestrator for N agents.
  Exposes ``_make_filter_fns`` to let subclasses inject safety filters in a
  unified, mode-aware way, avoiding duplicated shield logic.
"""

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np
import torch

from src.agents.dubins_robot import DubinsRobot
from src.agents.schemas import RobotConfig
from src.controllers.base.dual_guard import DualGuardShield
from src.controllers.base.mppi import MPPI
from src.controllers.schemas import BaseMultiAgentConfig, MPPIConfig, SafetyMode

logger = logging.getLogger(__name__)

# How neighbours enter the safety computation.
# - "nearest": only the single most dangerous neighbour (argmin h / V).
# - "all"    : all neighbours simultaneously (CBF only — HJ raises ValueError).
NeighborMode = Literal["nearest", "all"]


def get_value_in_map_from_coords(
    states_np: np.ndarray, coords: np.ndarray, map_data: np.ndarray
) -> np.ndarray:
    """Nearest-neighbour lookup.

    For each state in ``(K, 2)``, return the map value
    at the closest coordinate in ``coords`` of shape ``(N, 2)``.
    """
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    _, idxs = tree.query(states_np)
    return map_data[idxs]


class AgentMPPI(MPPI):
    """Single-agent MPPI planner with goal-reaching and smoke-risk cost.

    Args:
        params:          MPPI hyper-parameters.
        robot:           Dubins-car dynamics model.
        goal_thresh:     Distance (m) at which the goal is considered reached.
        dt:              Simulation timestep (s).
        running_cost_fn: Optional override for the running cost.  Signature:
                         ``(state, u, t) -> Tensor(K,)``.
    """

    def __init__(
        self,
        config: MPPIConfig,
        robot: DubinsRobot,
        goal_thresh: float = 0.1,
        dt: float = 0.1,
        running_cost_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, Optional[int]], torch.Tensor]
        ] = None,
    ):
        super().__init__(config=config)
        self.robot = robot
        self.goal_thresh = goal_thresh
        self.dt = dt
        self._goal: Optional[torch.Tensor] = None
        self._maps: deque = deque(maxlen=config.horizon)
        
        if running_cost_fn is not None:
            self.running_cost_fn = running_cost_fn
        else:
            self.running_cost_fn = self.running_cost
            
        self.terminal_state_cost_fn = self.terminal_state_cost

    def set_goal(self, goal_position) -> None:
        self._goal = torch.tensor(goal_position, dtype=self.dtype, device=self.device)

    def set_maps(self, maps: deque) -> None:
        self._maps = maps
        self._cached_maps = []
        import warnings
        from scipy.spatial import KDTree

        for coords, flatten_risk_map in maps:
            if torch.is_tensor(coords):
                coords_np = coords.cpu().detach().numpy()
            else:
                coords_np = np.array(coords)
                
            if torch.is_tensor(flatten_risk_map):
                risk_map_np = flatten_risk_map.cpu().detach().numpy()
            else:
                risk_map_np = np.array(flatten_risk_map)
                
            P = coords_np.shape[0]
            unique_y = np.unique(np.round(coords_np[:, 1], decimals=3))
            unique_x = np.unique(np.round(coords_np[:, 0], decimals=3))
            H = len(unique_y)
            W = len(unique_x)
            
            if H * W == P and H > 1 and W > 1:
                y_min, y_max = unique_y.min(), unique_y.max()
                x_min, x_max = unique_x.min(), unique_x.max()
                
                grid_tensor = torch.tensor(risk_map_np.reshape(1, 1, H, W), dtype=self.dtype, device=self.device)
                
                self._cached_maps.append({
                    "is_grid": True,
                    "grid_tensor": grid_tensor,
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max
                })
            else:
                warnings.warn("Using KDTree for map lookups. This will be slower.")
                tree = KDTree(coords_np)
                self._cached_maps.append({
                    "is_grid": False,
                    "tree": tree,
                    "risk_map_np": risk_map_np
                })

    def dynamics(
        self, state: torch.Tensor, u: torch.Tensor, t: Optional[int] = None
    ) -> torch.Tensor:
        return self.robot.dynamics(state, u)

    def _compute_risk_cost(self, states: torch.Tensor, t: Optional[int]) -> torch.Tensor:
        if not hasattr(self, '_cached_maps') or len(self._cached_maps) == 0 or t is None:
            return torch.zeros(states.shape[0], dtype=self.dtype, device=self.device)

        map_idx = min(t, len(self._cached_maps) - 1)
        cached = self._cached_maps[map_idx]
        
        if cached["is_grid"]:
            x = states[:, 0]
            y = states[:, 1]
            x_min, x_max = cached["x_min"], cached["x_max"]
            y_min, y_max = cached["y_min"], cached["y_max"]
            
            norm_x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
            norm_y = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
            
            norm_grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2)
            
            sampled = torch.nn.functional.grid_sample(
                cached["grid_tensor"], 
                norm_grid, 
                mode='nearest', 
                padding_mode='border', 
                align_corners=True
            )
            risk = sampled.view(-1)
        else:
            states_np = states[:, :2].detach().cpu().numpy()
            _, idxs = cached["tree"].query(states_np)
            risk_np = cached["risk_map_np"][idxs]
            risk = torch.tensor(risk_np, dtype=self.dtype, device=self.device)
            
        if t == 0:
            if cached["is_grid"]:
                max_density = cached["grid_tensor"].max().item()
            else:
                max_density = cached["risk_map_np"].max()
            print(f"[DEBUG] [t=0] Mapa max density: {max_density:.4f}, Trayectorias max risk: {risk.max().item():.4f}")
            
        return risk

    def running_cost(
        self, state: torch.Tensor, u: torch.Tensor, t: Optional[int] = None
    ) -> torch.Tensor:
        # We do NOT check self.running_cost_fn here because MPPI calls this directly
        # when running_cost_fn is set to self.running_cost.
        dist_cost = torch.norm(state[:, :2] - self._goal, dim=1)
        risk_cost = self._compute_risk_cost(state, t)
        return self.config.cost_distance_weight * dist_cost + self.config.cost_risk_weight * risk_cost

    def terminal_state_cost(self, states: torch.Tensor) -> Optional[torch.Tensor]:
        if self._goal is None:
            return None
        K, T, nx = states.shape
        goal_reached = torch.norm(states[:, :, :2] - self._goal, dim=2) < self.goal_thresh
        cost = torch.zeros(K, dtype=self.dtype, device=self.device)

        for k in range(K):
            if goal_reached[k].any():
                cost[k] = self.config.cost_goal_reached
        return cost


class BaseMultiAgentController:
    """Centralized orchestrator for N >= 1 decentralized MPPI agents.

    Subclasses implement ``_prepare_agent`` to inject safety filters.
    The helper ``_make_filter_fns`` provides a unified way to translate a
    *safety_fn* / *qp_fn* pair into the three supported safety modes:

    * ``"filter"``  — QP projection applied only at the **output** step.
      Rollout samples are unconstrained; safety is imposed on the executed
      action.  Fast and unbiased, but rollout cost landscape is unsafe.

    * ``"rollout"`` — **DualGuard / Shielded MPPI**: QP projection applied
      during **both** rollout and output.  Forces MPPI to explore within the
      safe set, yielding better cost estimates near obstacles.  More
      conservative and slightly slower due to per-step projection.

    * ``"penalty"`` — The safety value h(x) (or V(x) for HJ) is added as a
      soft penalty to the running cost.  No hard projection.  Allows MPPI to
      trade off between goal-reaching and safety via ``rho``.

    Args:
        num_agents:   Number of agents.
        robot_config: Robot configuration (passed to ``DubinsRobot``).
        mppi_config:  MPPI hyper-parameters shared across agents.
        goal_thresh:  Distance threshold for goal-reached detection.
        device:       PyTorch device string.
        dtype:        PyTorch floating-point dtype.
        dt:           Simulation timestep (s).
        r_sense:      Sensing radius — neighbours beyond this are ignored (m).
    """

    def __init__(
        self,
        num_agents: int,
        robot_config: RobotConfig,
        config: BaseMultiAgentConfig,
        goal_thresh: float = 0.1,
        dtype=torch.float32,
    ):
        self.num_agents = num_agents
        self.config = config
        self.device = config.mppi.device
        self.dtype = dtype
        self.dt = config.dt
        self.r_sense = config.safety.r_sense

        self.agents_controllers: Dict[str, AgentMPPI] = {
            f"agent_{i}": AgentMPPI(
                config=config.mppi,
                robot=DubinsRobot(robot_config),
                goal_thresh=goal_thresh,
                dt=self.dt,
            )
            for i in range(num_agents)
        }

        self.executor = ThreadPoolExecutor(max_workers=max(1, num_agents))
        self.last_commands: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_goals(self, goals: Dict[str, Any]) -> None:
        for key, goal_pos in goals.items():
            if key in self.agents_controllers:
                self.agents_controllers[key].set_goal(goal_pos)

    def set_maps(self, maps_deque: deque) -> None:
        for ctrl in self.agents_controllers.values():
            ctrl.set_maps(maps_deque)

    # Paleta de colores por agente (6 agentes)
    _AGENT_COLORS = [
        "#e63946",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#457b9d",
        "#a8dadc",
    ]

    def visualize_rollouts(
        self,
        ax,
        draw_samples: bool = True,
        sample_alpha: float = 0.12,
        sample_stride: int = 5,
    ) -> None:
        """Dibuja las trayectorias MPPI de cada agente en *ax*.

        Args:
            ax:            Matplotlib Axes donde dibujar.
            draw_samples:  Si True, dibuja las K trayectorias muestreadas en gris.
            sample_alpha:  Transparencia de las trayectorias de muestra.
            sample_stride: Dibuja 1 de cada N trayectorias para no saturar.
        """
        for agent_idx, (agent_key, ctrl) in enumerate(self.agents_controllers.items()):
            # synthetic_states se puebla después de cada ctrl.command()
            if ctrl.synthetic_states is None or ctrl.omega is None:
                continue

            # synthetic_states: (K, T, nx) - en device
            trajs = ctrl.synthetic_states.detach().cpu().numpy()  # (K, T, nx)
            omega = ctrl.omega.detach().cpu().numpy()  # (K,)

            # Prepend current state (t=0) to close the visual gap
            current_state = ctrl.state.detach().cpu().numpy()  # (nx,)
            current_state_repeated = np.tile(current_state, (trajs.shape[0], 1, 1))  # (K, 1, nx)
            trajs = np.concatenate([current_state_repeated, trajs], axis=1)  # (K, T+1, nx)

            color = self._AGENT_COLORS[agent_idx % len(self._AGENT_COLORS)]

            # Trayectorias muestreadas (submuestra para rendimiento)
            if draw_samples:
                for k in range(0, trajs.shape[0], sample_stride):
                    ax.plot(
                        trajs[k, :, 0],
                        trajs[k, :, 1],
                        color="gray",
                        alpha=sample_alpha,
                        linewidth=0.5,
                        zorder=2,
                    )

            # Trayectoria media ponderada por omega
            weighted = (omega[:, None, None] * trajs).sum(axis=0)  # (T, nx)
            ax.plot(
                weighted[:, 0],
                weighted[:, 1],
                color=color,
                linewidth=2.0,
                zorder=3,
                alpha=0.9,
            )

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Decentralized planning step with concurrent agent scheduling."""
        agents_states = {}
        for key, obs in current_obs.items():
            if obs is not None and key in self.agents_controllers:
                x, y = obs["location"]
                theta = obs["angle"]
                # Get last velocity from previous command, default to 0.0 if not available
                v = self.last_commands[key][0].item() if key in self.last_commands else 0.0
                agents_states[key] = torch.tensor([x, y, theta, v], dtype=self.dtype)

        def plan_agent(ego_key: str, ego_ctrl: AgentMPPI) -> Tuple[str, Optional[torch.Tensor]]:
            if ego_key not in agents_states:
                return ego_key, None

            ego_state_full = agents_states[ego_key]
            ego_state_core = ego_state_full[:3].to(device=self.device)
            ego_ctrl.state = ego_state_core

            ego_pos = ego_state_full[:2]
            neighbors = [
                state
                for k, state in agents_states.items()
                if k != ego_key and torch.norm(ego_pos - state[:2]) <= self.r_sense
            ]

            self._prepare_agent(ego_key, ego_ctrl, neighbors)
            command = ego_ctrl.command(ego_state_core)
            return ego_key, command

        commands: Dict[str, torch.Tensor] = {}

        if self.num_agents == 1:
            for key, ctrl in self.agents_controllers.items():
                k, cmd = plan_agent(key, ctrl)
                if cmd is not None:
                    commands[k] = cmd
                    self.last_commands[k] = cmd
        else:
            futures = [
                self.executor.submit(plan_agent, key, ctrl)
                for key, ctrl in self.agents_controllers.items()
            ]
            for future in futures:
                key, cmd = future.result()
                if cmd is not None:
                    commands[key] = cmd
                    self.last_commands[key] = cmd

        return commands

    # ------------------------------------------------------------------
    # Hook for subclasses
    # ------------------------------------------------------------------

    def _prepare_agent(self, ego_key: str, ego_ctrl: AgentMPPI, neighbors: list) -> None:
        """Configure safety filters for *ego_ctrl* given its *neighbors*.

        Subclasses call ``_make_filter_fns`` with mode-specific functions and
        assign the results to ``ego_ctrl.rollout_filter_fn``,
        ``ego_ctrl.output_filter_fn``, and/or ``ego_ctrl.running_cost_fn``.

        The *neighbors* list passed here already contains **only agents within
        r_sense**. The ``neighbor_mode`` argument to ``_make_filter_fns``
        further controls whether the safety computation considers ``"all"`` of
        them simultaneously or only the ``"nearest"`` (worst-case) one.
        """

    # ------------------------------------------------------------------
    # Shared safety-mode builder
    # ------------------------------------------------------------------

    def _make_filter_fns(
        self,
        ego_ctrl: AgentMPPI,
        mode: SafetyMode,
        *,
        safety_fn: Callable,
        qp_fn: Callable,
        penalty_fn: Optional[Callable] = None,
        safe_margin: float = 0.0,
        penalty_weight: float = 1.0,
    ) -> None:
        """Wire safety functions into *ego_ctrl* according to *mode*."""
        # Validate mode early — "online_dual-guard" is HJ-specific and must be
        # implemented by the subclass; base class does not handle it.
        if mode == "online_dual-guard":
            raise NotImplementedError(
                "'online_dual-guard' must be handled by the subclass before calling _make_filter_fns. "
                "Implement the BRT re-solve logic in _prepare_agent and then call _make_filter_fns "
                "with mode='dual-guard' using the freshly computed value function."
            )
        """Wire safety functions into *ego_ctrl* according to *mode*.

        Args:
            ego_ctrl:       The agent planner to configure.
            mode:           One of ``"filter"``, ``"dual-guard"``, ``"penalty"``.
            safety_fn:      ``(state, t) -> Tensor(K,)`` — safety index/value.
                            Positive = safe, negative = unsafe.
            qp_fn:          ``(state, u, t) -> Tensor(K, nu)`` — QP/LRF
                            projection of *u* onto the safe control set.
            penalty_fn:     Optional override for the penalty-mode cost term.
                            ``(state, u, t) -> Tensor(K,)``.  If ``None``,
                            defaults to ``penalty_weight * clamp(-safety_fn, 0)^2``.
            safe_margin:    Safety threshold — states with ``safety_fn < margin``
                            are treated as unsafe.
            penalty_weight: Scaling factor ``rho`` for the penalty mode.
        """
        if mode == "filter":
            # ── Output-only projection ──────────────────────────────────────
            # Rollout explores freely; executed action is projected to safe set.
            ego_ctrl.rollout_filter_fn = None
            ego_ctrl.output_filter_fn = lambda state, u, t=0: qp_fn(state, u, t)

        elif mode == "dual-guard":
            # ── DualGuard / Shielded MPPI ───────────────────────────────────
            # Both rollout samples and the executed action are shielded.
            # MPPI explores within the safe set → better cost landscape near
            # obstacles, but slightly more conservative trajectories.
            nu = ego_ctrl.nu
            shield = DualGuardShield(
                safety_function=safety_fn,
                safe_control_function=lambda state, u, t=0: qp_fn(state, u, t),
                safe_margin=safe_margin,
            )
            ego_ctrl.rollout_filter_fn = shield
            ego_ctrl.output_filter_fn = shield

        elif mode == "penalty":
            # ── Soft-barrier penalty ────────────────────────────────────────
            # No hard projection. Safety value is added as a running cost term.
            # Useful when combining with learned costs or when hard constraints
            # cause infeasibility (e.g. dense crowds).
            ego_ctrl.rollout_filter_fn = None
            ego_ctrl.output_filter_fn = None

            # Prevent infinite wrapping of the cost function across timesteps
            if not hasattr(ego_ctrl, "_original_running_cost_fn"):
                ego_ctrl._original_running_cost_fn = ego_ctrl.running_cost_fn
            base_cost_fn = ego_ctrl._original_running_cost_fn

            def penalty_running_cost(
                state: torch.Tensor, u: torch.Tensor, t: Optional[int] = None
            ) -> torch.Tensor:
                # Base navigation cost
                if base_cost_fn is not None:
                    base = base_cost_fn(state, u, t)
                else:
                    dist = torch.norm(state[:, :2] - ego_ctrl._goal, dim=1)
                    risk = ego_ctrl._compute_risk_cost(state, t)
                    base = dist + 20.0 * risk

                # Safety penalty
                if penalty_fn is not None:
                    safety_cost = penalty_fn(state, u, t if t is not None else 0)
                else:
                    h = safety_fn(state, t if t is not None else 0)
                    # Use a heavy linear penalty (C_pen = 1000.0) to match the old CBF implementation
                    # and ensure safety costs strongly dominate the goal distance during violations
                    safety_cost = 100.0 * penalty_weight * torch.clamp(-h, min=0.0)

                safety_cost = safety_cost.to(base.device)
                return base + safety_cost

            ego_ctrl.running_cost_fn = penalty_running_cost

        else:
            raise ValueError(
                f"Unknown safety mode: {mode!r}. Choose 'filter', 'dual-guard', or 'penalty'."
            )

if __name__ == '__main__':
    # Test comparativo entre KDTree y grid_sample
    print("Corriendo test comparativo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear un grid falso de 10x10 (H=10, W=10)
    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    xx, yy = np.meshgrid(x, y) # indexing='xy' (row-major: Y changes slow, X changes fast)
    
    # Coordenadas (N, 2)
    coords = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    
    # Riesgo aleatorio
    risk_map = np.random.rand(100).astype(np.float32)
    
    # Estados de trayectoria aleatorios (X, Y) dentro y fuera de límites
    states_np = np.random.uniform(-2, 12, (50, 2)).astype(np.float32)
    states = torch.tensor(states_np, device=device)
    
    # 1. Metodo KDTree
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    _, idxs = tree.query(states_np)
    risk_kdtree = risk_map[idxs]
    risk_kdtree_tensor = torch.tensor(risk_kdtree, device=device)
    
    # 2. Metodo Grid Sample
    unique_y = np.unique(coords[:, 1])
    unique_x = np.unique(coords[:, 0])
    y_min, y_max = unique_y.min(), unique_y.max()
    x_min, x_max = unique_x.min(), unique_x.max()
    
    grid_tensor = torch.tensor(risk_map.reshape(1, 1, 10, 10), device=device)
    
    norm_x = 2.0 * (states[:, 0] - x_min) / (x_max - x_min) - 1.0
    norm_y = 2.0 * (states[:, 1] - y_min) / (y_max - y_min) - 1.0
    norm_grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2)
    
    sampled = torch.nn.functional.grid_sample(
        grid_tensor, 
        norm_grid, 
        mode='nearest', 
        padding_mode='border', 
        align_corners=True
    )
    risk_grid_sample = sampled.view(-1)
    
    # Comparar
    is_close = torch.allclose(risk_kdtree_tensor, risk_grid_sample, atol=1e-5)
    print(f"Results are the same? {'YES' if is_close else 'NO'}")
    if not is_close:
        print("Differences found:")
        print("KDTree:", risk_kdtree_tensor[:15])
        print("Grid Sample:", risk_grid_sample[:15])
