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

import jax
import jax.numpy as jnp
import numpy as np
import torch

from src.controllers.base.hj import HJSolver, HJSolverConfig
from src.controllers.base.hj_safety import HJFilter, HJFilterParams
from src.controllers.base.mppi import MPPIParams
from src.controllers.base.proper_models import RelativeDubinsDynamics
from src.controllers.base_multi_agent import AgentMPPI, BaseMultiAgentController, SafetyMode

logger = logging.getLogger(__name__)


class MultiAgentHJController(BaseMultiAgentController):
    """Unified HJ Reachability + LRF multi-agent controller.

    Args:
        num_agents:   Number of agents.
        robot_params: Robot configuration.
        mppi_params:  MPPI hyper-parameters.
        hj_params:    HJ filter parameters.
        hj_config:    HJ solver configuration (grid, accuracy, etc.).
        hj_mode:      Safety mode — ``"filter"``, ``"rollout"``, or ``"penalty"``.
        goal_thresh:  Distance threshold for goal-reached detection.
        device:       PyTorch device string.
        dtype:        PyTorch floating-point dtype.
        dt:           Simulation timestep (s).
    """

    def __init__(
        self,
        num_agents: int,
        robot_params: Any,
        mppi_params: MPPIParams,
        hj_params: HJFilterParams,
        hj_config: Any,  # HJSolverConfig
        hj_mode: SafetyMode = "filter",
        goal_thresh: float = 0.1,
        device: str = "cpu",
        dtype=torch.float32,
        dt: float = 0.1,
    ):
        super().__init__(
            num_agents=num_agents,
            robot_params=robot_params,
            mppi_params=mppi_params,
            goal_thresh=goal_thresh,
            device=device,
            dtype=dtype,
            dt=dt,
            r_sense=hj_params.r_sense,
        )
        self.hj_params = hj_params
        self.hj_config = hj_config
        self.hj_mode = hj_mode

        # Static obstacle value function (absolute frame)
        _ref_robot = next(iter(self.agents_controllers.values())).robot
        self._hj_solver = HJSolver(hj_config, _ref_robot)
        self._hj_values: Optional[np.ndarray] = None
        self._hj_values_grad: Optional[list] = None

        # Relative (dynamic inter-agent) value function
        self._hj_relative_grid = None
        self._hj_relative_values: Optional[np.ndarray] = None
        self._hj_relative_values_grad: Optional[list] = None

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def solve(self, grid_map: np.ndarray, time: float = 0.0) -> None:
        """Precompute static BRT value function V(x) and its gradients."""
        self._hj_values = self._hj_solver.solve(1.0 - grid_map, time=time)
        self._hj_values_grad = self._hj_solver.last_values_grad
        logger.info("Static HJ values and gradients precomputed.")

    def solve_relative(
        self,
        time: float = 0.0,
        target_time: float = -5.0,
        dt: float = 0.05,
        epsilon: float = 0.01,
    ) -> None:
        """Precompute relative HJI value function V_rel(x_rel) and its gradients."""
        import hj_reachability as hj

        domain_cells = self.hj_config.domain_cells
        domain = self.hj_config.domain

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(np.array(domain[0]), np.array(domain[1])),
            tuple(domain_cells),
            periodic_dims=2,
        )

        action_min = next(iter(self.agents_controllers.values())).robot.action_min
        action_max = next(iter(self.agents_controllers.values())).robot.action_max
        dynamics = RelativeDubinsDynamics(action_min, action_max)

        grid_states = grid.states
        initial_values = (
            np.sqrt(grid_states[..., 0] ** 2 + grid_states[..., 1] ** 2) - self.hj_params.d_safe
        )

        solver_settings = hj.SolverSettings.with_accuracy(
            self.hj_config.accuracy,
            hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        )
        problem = dict(solver_settings=solver_settings, dynamics=dynamics, grid=grid)

        target_time = -abs(target_time)
        times = np.linspace(time, target_time, int(abs(target_time - time) / dt))
        values = initial_values

        logger.info("Starting relative HJI BRT computation...")
        for i in range(1, len(times)):
            values_new = hj.step(
                **problem,
                time=times[i - 1],
                values=values,
                target_time=times[i],
                progress_bar=False,
            )
            diff = np.max(np.abs(values_new - values))
            values = values_new
            if diff < epsilon:
                break

        self._hj_relative_grid = grid
        self._hj_relative_values = np.array(values)
        self._hj_relative_values_grad = list(jnp.gradient(jnp.asarray(self._hj_relative_values)))
        logger.info("Relative HJI solved and gradients precomputed.")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.hj_mode == "online_rollout":
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
        hj = HJFilter(self.hj_params)
        grid = self._hj_relative_grid
        values = self._hj_relative_values
        grad = self._hj_relative_values_grad

        # safety_fn: V(x_rel) — negative means inside the BRT (unsafe)
        safety_fn = lambda state, t=0: self._value_as_safety(
            hj.lrf_filter(state, torch.zeros_like(state[:, :2]), neighbors, grid, values, grad, t),
            state,
            neighbors,
            grid,
            values,
            t,
        )

        qp_fn = lambda state, u, t=0: hj.lrf_filter(state, u, neighbors, grid, values, grad, t)

        actual_mode = "rollout" if self.hj_mode == "online_rollout" else self.hj_mode

        self._make_filter_fns(
            ego_ctrl,
            actual_mode,
            safety_fn=safety_fn,
            qp_fn=qp_fn,
            safe_margin=self.hj_params.safe_margin,
            penalty_weight=1.0,
        )

    def _prepare_static(self, ego_ctrl: AgentMPPI) -> None:
        """Wire the static obstacle HJ filter in the selected mode."""
        hj_filter = HJFilter(self.hj_params)
        values = self._hj_values
        grad = self._hj_values_grad
        domain = self.hj_config.domain
        cells = self.hj_config.domain_cells

        qp_fn = lambda state, u, t=0: hj_filter.static_filter(
            state, u, values, grad, domain, cells, t
        )

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
            
            v_grid = values.to(device) if torch.is_tensor(values) else torch.tensor(np.asarray(values), device=device)
            v = v_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
            
            return v

        actual_mode = "rollout" if self.hj_mode == "online_rollout" else self.hj_mode

        self._make_filter_fns(
            ego_ctrl,
            actual_mode,
            safety_fn=safety_fn,
            qp_fn=qp_fn,
            safe_margin=self.hj_params.safe_margin,
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

        arrays = [n if torch.is_tensor(n) else torch.tensor(np.asarray(n), device=device, dtype=dtype) for n in neighbors]
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

        v_grid = values.to(device) if torch.is_tensor(values) else torch.tensor(np.asarray(values), device=device)
        v_kn = v_grid[i0, i1, i2].reshape(K, N)
        v_min, _ = torch.min(v_kn, dim=1)

        return v_min


if __name__ == "__main__":
    import numpy as np

    from agents.basic_robot import RobotParams
    from controllers.base.hj import HJSolverConfig

    robot_params = RobotParams(
        name="dubins",
        action_dim=2,
        state_dim=3,
        action_max=[6.0, 4.0],
        action_min=[0.0, -4.0],
        state_max=[30.0, 30.0, 2 * np.pi],
        state_min=[-30.0, -30.0, 0.0],
        dt=0.1,
    )
    mppi_params = MPPIParams(
        nx=3,
        noise_sigma=torch.eye(2),
        num_samples=10,
        horizon=5,
        device="cpu",
        u_min=torch.tensor([0.0, -4.0]),
        u_max=torch.tensor([6.0, 4.0]),
    )
    hj_params = HJFilterParams(
        d_safe=2.4,
        safe_margin=0.0,
        r_sense=8.0,
        dt=0.1,
        action_min=torch.tensor([0.0, -4.0]),
        action_max=torch.tensor([6.0, 4.0]),
    )
    hj_config = HJSolverConfig(
        domain_cells=np.array([20, 20, 10]),
        domain=np.array([[-5.0, -5.0, 0.0], [5.0, 5.0, 2 * np.pi]]),
        accuracy="low",
        superlevel_set_epsilon=0.0,
    )

    for mode in ("filter", "rollout", "penalty"):
        controller = MultiAgentHJController(
            num_agents=2,
            robot_params=robot_params,
            mppi_params=mppi_params,
            hj_params=hj_params,
            hj_config=hj_config,
            hj_mode=mode,
        )
        controller.solve_relative()
        controller.set_goals({"agent_0": [10.0, 0.0], "agent_1": [-10.0, 0.0]})
        obs = {
            "agent_0": {"location": [0.0, 0.0], "angle": 0.0},
            "agent_1": {"location": [0.5, 0.0], "angle": 0.0},
        }
        cmds = controller.get_commands(obs)
        print(f"[mode={mode}] Test passed. Commands:", {k: v.tolist() for k, v in cmds.items()})
