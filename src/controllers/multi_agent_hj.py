"""Multi-agent HJ Reachability safety controller.

Uses ``BaseMultiAgentController._make_filter_fns`` to wire the HJ
least-restrictive filter (LRF) into MPPI in one of three modes:

* ``"filter"``  — LRF projection applied only at the output step.
* ``"rollout"`` — DualGuard: LRF projection during rollout AND output.
* ``"penalty"`` — Soft barrier using V(x) < safe_margin as penalty.

Two value functions can be precomputed:
* ``solve_relative`` — game-theoretic BRT for dynamic neighbour avoidance.
* ``solve``          — static BRT for obstacle avoidance.

If both are precomputed, the relative (dynamic) filter takes precedence.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.agents.schemas import RobotConfig
from src.controllers.base.hj import HJSolver
from src.controllers.base.hj_safety import HJFilter, HJFilterConfig
from src.controllers.base.mppi import MPPIConfig
from src.controllers.base_multi_agent import AgentMPPI, BaseMultiAgentController
from src.controllers.schemas import MultiAgentHJConfig

logger = logging.getLogger(__name__)


class MultiAgentHJController(BaseMultiAgentController):
    """Unified HJ Reachability + LRF multi-agent controller.

    Args:
        num_agents:   Number of agents.
        robot_config: Robot configuration.
        config:       Configuration object containing all parameters.
    """

    def __init__(
        self,
        num_agents: int,
        robot_config: RobotConfig,
        config: MultiAgentHJConfig,
    ):
        super().__init__(
            num_agents=num_agents,
            robot_config=robot_config,
            config=config,
        )
        self.hj_solver_config = config.solver
        self.hj_mode = config.mode

        self.hj_filter_config = HJFilterConfig(
            robot_radius=config.safety.robot_radius,
            safe_margin=config.safety.safe_margin,
            r_sense=config.safety.r_sense,
            dt=config.dt,
            action_min=config.action_min,
            action_max=config.action_max,
            control_type=config.control_type,
        )

        _ref_robot = next(iter(self.agents_controllers.values())).robot

        if self.hj_solver_config.mode == "absolute":
            # Static obstacle value function (absolute frame)
            self._hj_solver = HJSolver(self.hj_solver_config, _ref_robot)
            self._hj_values: Optional[np.ndarray] = None
            self._hj_values_grad: Optional[list] = None

            self._hj_relative_solver = None
            self._hj_relative_grid = None
            self._hj_relative_values: Optional[np.ndarray] = None
            self._hj_relative_values_grad: Optional[list] = None
        else:
            # Relative (dynamic inter-agent) value function
            self._hj_solver = None
            self._hj_values: Optional[np.ndarray] = None
            self._hj_values_grad: Optional[list] = None

            self._hj_relative_solver = HJSolver(self.hj_solver_config, _ref_robot)
            self._hj_relative_grid = None
            self._hj_relative_values: Optional[np.ndarray] = None
            self._hj_relative_values_grad: Optional[list] = None

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def solve(self, grid_map: np.ndarray) -> None:
        """Precompute static BRT value function V(x) and its gradients."""
        if self._hj_solver is None:
            logger.warning("solve() called but controller is in relative mode. Ignoring.")
            return
        self._hj_values = self._hj_solver.solve(1.0 - grid_map, current_time=0.0)
        self._hj_values_grad = self._hj_solver.last_values_grad
        logger.info("Static HJ values and gradients precomputed.")

    def solve_relative(self) -> None:
        """Precompute relative HJI value function V_rel(x_rel) and its gradients."""
        if self._hj_relative_solver is None:
            logger.warning("solve_relative() called but controller is in absolute mode. Ignoring.")
            return
        collision_radius = 2.0 * self.config.safety.robot_radius + self.config.safety.safe_margin

        logger.info("Starting relative HJI BRT computation...")
        self._hj_relative_values = self._hj_relative_solver.solve(
            grid_map=None, current_time=0.0, collision_radius=collision_radius
        )
        self._hj_relative_values_grad = self._hj_relative_solver.last_values_grad
        self._hj_relative_grid = self._hj_relative_solver.grid
        logger.info("Relative HJI solved and gradients precomputed.")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.hj_mode == "online_dual-guard":
            # Solve relative game once per timestep for all agents
            self.solve_relative()
        return super().get_commands(current_obs)

    # ------------------------------------------------------------------
    # Safety filter injection
    # ------------------------------------------------------------------

    def _prepare_agent(self, ego_key: str, ego_ctrl: AgentMPPI, neighbors: list) -> None:
        if self._hj_relative_values is not None:
            self._prepare_lrf(ego_ctrl, neighbors)
        elif self._hj_values is not None:
            self._prepare_static(ego_ctrl)
        else:
            ego_ctrl.rollout_filter_fn = None
            ego_ctrl.output_filter_fn = None

    def _prepare_lrf(self, ego_ctrl: AgentMPPI, neighbors: list) -> None:
        """Wire the relative (dynamic) HJ LRF filter in the selected mode."""
        hj_filter = HJFilter(self.hj_filter_config)
        grid = self._hj_relative_grid
        values = self._hj_relative_values
        grad = self._hj_relative_values_grad

        # safety_fn: V(x_rel) — negative means inside the BRT (unsafe)
        def safety_fn(state: torch.Tensor, t: int = 0) -> torch.Tensor:
            return self._value_as_safety(
                hj_filter.lrf_filter(
                    state, torch.zeros_like(state[:, :2]), neighbors, grid, values, grad, t
                ),
                state,
                neighbors,
                grid,
                values,
                t,
            )

        def qp_fn(state: torch.Tensor, u: torch.Tensor, t: int = 0) -> torch.Tensor:
            return hj_filter.lrf_filter(state, u, neighbors, grid, values, grad, t)

        actual_mode = "dual-guard" if self.hj_mode == "online_dual-guard" else self.hj_mode

        self._make_filter_fns(
            ego_ctrl,
            actual_mode,
            safety_fn=safety_fn,
            qp_fn=qp_fn,
            safe_margin=0.0,  # LRF checks if V(x) < 0.0 (unsafe)
            penalty_weight=1.0,
        )

    def _prepare_static(self, ego_ctrl: AgentMPPI) -> None:
        """Wire the static obstacle HJ filter in the selected mode."""
        hj_filter = HJFilter(self.hj_filter_config)
        values = self._hj_values
        grad = self._hj_values_grad
        domain = self.hj_solver_config.domain
        cells = self.hj_solver_config.domain_cells

        def qp_fn(state: torch.Tensor, u: torch.Tensor, t: int = 0) -> torch.Tensor:
            return hj_filter.static_filter(state, u, values, grad, domain, cells, t)

        def safety_fn(state: torch.Tensor, t: int = 0) -> torch.Tensor:
            """Look up V(x) directly on the absolute grid using pure PyTorch."""
            device = state.device
            dtype = state.dtype

            lo = torch.tensor(domain[0], device=device, dtype=dtype)
            hi = torch.tensor(domain[1], device=device, dtype=dtype)
            c = torch.tensor(cells, device=device, dtype=torch.long)
            dx = (hi - lo) / c

            idx = torch.round((state - lo) / dx).long()
            idx = torch.clamp(idx, 0, c - 1)

            v_grid = (
                values.to(device)
                if torch.is_tensor(values)
                else torch.tensor(np.asarray(values), device=device)
            )
            v = v_grid[idx[:, 0], idx[:, 1], idx[:, 2]]

            return v

        actual_mode = "dual-guard" if self.hj_mode == "online_dual-guard" else self.hj_mode

        self._make_filter_fns(
            ego_ctrl,
            actual_mode,
            safety_fn=safety_fn,
            qp_fn=qp_fn,
            safe_margin=0.0,  # LRF checks if V(x) < 0.0 (unsafe)
            penalty_weight=1.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _value_as_safety(
        _unused_u: torch.Tensor,
        state: torch.Tensor,
        neighbors: list,
        grid,
        values,
        t: int,
    ) -> torch.Tensor:
        """Extract minimum V(x_rel) over neighbours as the safety scalar.

        Pure PyTorch implementation to avoid JAX overhead inside DualGuardShield.
        """
        if not neighbors:
            return 1000.0 * torch.ones(state.shape[0], dtype=state.dtype, device=state.device)

        device = state.device
        dtype = state.dtype
        K = state.shape[0]

        arrays = [
            n if torch.is_tensor(n) else torch.tensor(np.asarray(n), device=device, dtype=dtype)
            for n in neighbors
        ]
        neighbors_t = torch.stack(arrays).to(device=device, dtype=dtype)
        N = neighbors_t.shape[0]

        theta = state[:, 2:3]  # (K, 1)
        theta_j = neighbors_t[:, 2]  # (N,)

        dx = neighbors_t[:, 0] - state[:, 0:1]  # (K, N)
        dy = neighbors_t[:, 1] - state[:, 1:2]

        xr = torch.cos(theta) * dx + torch.sin(theta) * dy  # (K, N)
        yr = -torch.sin(theta) * dx + torch.cos(theta) * dy
        thr = (theta_j - theta) % (2.0 * torch.pi)  # (K, N)

        lo = torch.tensor(grid.domain.lo, device=device, dtype=dtype)
        hi = torch.tensor(grid.domain.hi, device=device, dtype=dtype)
        cells = torch.tensor(grid.shape, device=device, dtype=torch.long)
        dx_grid = (hi - lo) / cells

        states_rel_flat = torch.stack([xr.reshape(-1), yr.reshape(-1), thr.reshape(-1)], dim=-1)
        idx = torch.round((states_rel_flat - lo) / dx_grid).long()

        i0 = torch.clamp(idx[:, 0], 0, cells[0] - 1)
        i1 = torch.clamp(idx[:, 1], 0, cells[1] - 1)
        i2 = idx[:, 2] % cells[2]

        v_grid = (
            values.to(device)
            if torch.is_tensor(values)
            else torch.tensor(np.asarray(values), device=device)
        )
        v_kn = v_grid[i0, i1, i2].reshape(K, N)
        v_min, _ = torch.min(v_kn, dim=1)

        return v_min


if __name__ == "__main__":
    import numpy as np

    from src.agents.schemas import RobotConfig

    # RobotParams
    from src.controllers.base.schemas import RelativeHJSolverConfig

    robot_config = RobotConfig(
        name="dubins2d",
        device="cpu",
        action_dim=2,
        state_dim=3,
        action_max=[6.0, 4.0],
        action_min=[0.0, -4.0],
        state_max=[30.0, 30.0, 2 * np.pi],
        state_min=[-30.0, -30.0, 0.0],
        dt=0.1,
    )
    from src.controllers.schemas import SharedSafetyConfig

    mppi_config = MPPIConfig(
        nx=3,
        noise_sigma=torch.eye(2),
        num_samples=10,
        horizon=5,
        device="cpu",
        u_min=torch.tensor([0.0, -4.0]),
        u_max=torch.tensor([6.0, 4.0]),
    )
    hj_solver_config = RelativeHJSolverConfig(
        interaction_radius=10.0,
        spatial_resolution=0.5,
        angular_cells=18,
        accuracy="low",
        superlevel_set_epsilon=0.0,
    )
    shared_safety = SharedSafetyConfig(robot_radius=0.4, safe_margin=0.2, r_sense=8.0)

    for mode in ("filter", "dual-guard", "penalty"):
        config = MultiAgentHJConfig(
            safety=shared_safety,
            mppi=mppi_config,
            mode=mode,
            solver=hj_solver_config,
            action_min=[0.0, -4.0],
            action_max=[6.0, 4.0],
            dt=0.1,
        )
        controller = MultiAgentHJController(
            num_agents=2,
            robot_config=robot_config,
            config=config,
        )
        controller.solve_relative()
        controller.set_goals({"agent_0": [10.0, 0.0], "agent_1": [-10.0, 0.0]})
        obs = {
            "agent_0": {"location": [0.0, 0.0], "angle": 0.0},
            "agent_1": {"location": [0.5, 0.0], "angle": 0.0},
        }
        cmds = controller.get_commands(obs)
        print(f"[mode={mode}] Test passed. Commands:", {k: v.tolist() for k, v in cmds.items()})
