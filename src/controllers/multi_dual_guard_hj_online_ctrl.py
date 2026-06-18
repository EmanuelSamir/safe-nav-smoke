"""Multi-agent decentralized DualGuard controller using HJ reachability solved in real-time (online).

Architecture mirrors MultiDualGuardHJCtrl, but solves the relative ISAACS equation
backwards in time on-the-fly at each time step.
"""

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import torch

from controllers.hj import HJSolver, HJSolverConfig
from controllers.mppi_ctrl import MPPICtrlParams
from controllers.multi_dual_guard_cbf_ctrl import _DualGuardMPPICtrl

logger = logging.getLogger(__name__)


class MultiDualGuardHJOnlineCtrl:
    """Centralized orchestrator for decentralized multi-agent safe navigation
    using HJ reachability solved online at each planning step.
    """

    def __init__(
        self,
        num_agents: int,
        robot_params: Any,
        robot_type: str,
        hj_config: HJSolverConfig,
        goal_thresh: float = 0.1,
        device: str = "cpu",
        dtype=torch.float32,
        mppi_params: MPPICtrlParams = MPPICtrlParams(),
        dt: float = 0.1,
        r_sense: float = 8.0,
        d_safe: float = 1.2,
    ):
        self.num_agents = num_agents
        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.r_sense = r_sense
        self.d_safe = d_safe

        # Per-agent controllers (DualGuard-backed MPPI)
        self.agents_controllers: Dict[str, _DualGuardMPPICtrl] = {
            f"agent_{i}": _DualGuardMPPICtrl(
                robot_params=robot_params,
                robot_type=robot_type,
                goal_thresh=goal_thresh,
                device=device,
                dtype=dtype,
                mppi_params=mppi_params,
                dt=dt,
            )
            for i in range(num_agents)
        }

        # Persistent executor
        self.executor = ThreadPoolExecutor(max_workers=num_agents)

        _ref_robot = self.agents_controllers["agent_0"].robot
        self._hj_solver = HJSolver(hj_config, _ref_robot)
        self._hj_values: Optional[np.ndarray] = None
        self._hj_values_grad: Optional[list] = None
        self._safe_margin: float = hj_config.superlevel_set_epsilon

        self._hj_relative_grid = None
        self._hj_relative_values: Optional[np.ndarray] = None
        self._hj_relative_values_grad: Optional[list] = None

    def set_goals(self, goals: Dict[str, Any]):
        for key, goal_pos in goals.items():
            if key in self.agents_controllers:
                self.agents_controllers[key].set_goal(goal_pos)

    def set_maps(self, maps_deque: deque):
        for ctrl in self.agents_controllers.values():
            ctrl.set_maps(maps_deque)

    def solve_relative(
        self,
        d_safe: float,
        time: float = 0.0,
        target_time: float = -5.0,
        dt: float = 0.05,
        epsilon: float = 0.01,
    ):
        """Solve the relative HJI value function online."""
        import hj_reachability as hj
        from controllers.hj import RelativeDubinsDynamics

        domain_cells = self._hj_solver.config.domain_cells
        domain = self._hj_solver.config.domain

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(np.array(domain[0]), np.array(domain[1])),
            tuple(domain_cells),
            periodic_dims=2,
        )

        action_min = self.agents_controllers["agent_0"].robot.action_min
        action_max = self.agents_controllers["agent_0"].robot.action_max
        dynamics = RelativeDubinsDynamics(action_min, action_max)

        grid_states = grid.states
        x_coords = grid_states[..., 0]
        y_coords = grid_states[..., 1]
        initial_values = np.sqrt(x_coords**2 + y_coords**2) - d_safe

        solver_settings = hj.SolverSettings.with_accuracy(
            self._hj_solver.config.accuracy,
            hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        )

        problem_definition = {
            "solver_settings": solver_settings,
            "dynamics": dynamics,
            "grid": grid,
        }

        times = np.linspace(time, target_time, int(abs(target_time - time) / dt))
        values = initial_values

        for i in range(1, len(times)):
            values_new = hj.step(
                **problem_definition,
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
        self._hj_relative_values_grad = np.gradient(self._hj_relative_values)

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Solve online HJI relative value function at each step
        self.solve_relative(
            d_safe=self.d_safe, time=0.0, target_time=-5.0, dt=0.05, epsilon=0.01
        )

        agents_states = {
            key: np.array([obs["location"][0], obs["location"][1], obs["angle"]], dtype=np.float32)
            for key, obs in current_obs.items()
            if obs is not None and key in self.agents_controllers
        }

        def plan_agent(ego_key: str, ego_ctrl: _DualGuardMPPICtrl) -> tuple:
            if ego_key not in agents_states:
                return ego_key, None

            ego_state = agents_states[ego_key]
            ego_ctrl.set_state(ego_state)

            ego_pos = ego_state[:2]
            neighbors = [
                state
                for k, state in agents_states.items()
                if k != ego_key and np.linalg.norm(ego_pos - state[:2]) <= self.r_sense
            ]

            ego_ctrl.planner.safety_function = lambda states, t=0: (
                self._hj_safety_function_batch_relative(states, neighbors, t)
            )
            ego_ctrl.planner.safe_control_function = lambda states, t=0: (
                self._hj_safe_control_batch_relative(states, neighbors, t)
            )
            ego_ctrl.planner.safe_margin = self._safe_margin

            return ego_key, ego_ctrl.get_command()

        commands: Dict[str, torch.Tensor] = {}

        if self.num_agents == 1:
            for key, ctrl in self.agents_controllers.items():
                k, cmd = plan_agent(key, ctrl)
                if cmd is not None:
                    commands[k] = cmd
        else:
            futures = [
                self.executor.submit(plan_agent, key, ctrl)
                for key, ctrl in self.agents_controllers.items()
            ]
            for future in futures:
                key, cmd = future.result()
                if cmd is not None:
                    commands[key] = cmd

        return commands

    def visualize_rollouts(self, ax, draw_samples: bool = False):
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for i, (key, ctrl) in enumerate(self.agents_controllers.items()):
            ctrl.visualize_rollouts(ax, color=colors[i % len(colors)], draw_samples=draw_samples)

    def _hj_safety_function_batch_relative(
        self, states: torch.Tensor, neighbors: list, t: int = 0
    ) -> torch.Tensor:
        if len(neighbors) == 0:
            return 10.0 * torch.ones(states.shape[0], dtype=states.dtype, device=states.device)

        states_np = states.detach().cpu().numpy()
        grid = self._hj_relative_grid
        v_list = []

        lo = grid.domain.lo
        dx_grid = np.array(grid.spacings)
        cells = self._hj_solver.config.domain_cells

        for neighbor in neighbors:
            dx = neighbor[0] - states_np[:, 0]
            dy = neighbor[1] - states_np[:, 1]
            theta = states_np[:, 2]

            xr = np.cos(theta) * dx + np.sin(theta) * dy
            yr = -np.sin(theta) * dx + np.cos(theta) * dy
            thr = (neighbor[2] - theta) % (2.0 * np.pi)

            states_rel = np.stack([xr, yr, thr], axis=-1)
            idx_np = np.round((states_rel - lo) / dx_grid).astype(int)
            idx_clipped = tuple(np.clip(idx_np[:, d], 0, cells[d] - 1) for d in range(3))
            v_neigh = self._hj_relative_values[idx_clipped]
            v_list.append(v_neigh)

        v_matrix = np.stack(v_list, axis=1)
        v_min = np.min(v_matrix, axis=1)

        return torch.tensor(v_min, dtype=states.dtype, device=states.device)

    def _hj_safe_control_batch_relative(
        self, states: torch.Tensor, neighbors: list, t: int = 0
    ) -> torch.Tensor:
        if len(neighbors) == 0:
            return torch.zeros((states.shape[0], 2), dtype=states.dtype, device=states.device)

        states_np = states.detach().cpu().numpy()
        K = states_np.shape[0]
        grid = self._hj_relative_grid
        v_list = []
        xr_list = []
        yr_list = []
        grad_x_list = []
        grad_y_list = []
        grad_theta_list = []

        lo = grid.domain.lo
        dx_grid = np.array(grid.spacings)
        cells = self._hj_solver.config.domain_cells

        for neighbor in neighbors:
            dx = neighbor[0] - states_np[:, 0]
            dy = neighbor[1] - states_np[:, 1]
            theta = states_np[:, 2]

            xr = np.cos(theta) * dx + np.sin(theta) * dy
            yr = -np.sin(theta) * dx + np.cos(theta) * dy
            thr = (neighbor[2] - theta) % (2.0 * np.pi)

            states_rel = np.stack([xr, yr, thr], axis=-1)
            idx_np = np.round((states_rel - lo) / dx_grid).astype(int)
            idx_clipped = tuple(np.clip(idx_np[:, d], 0, cells[d] - 1) for d in range(3))

            v_neigh = self._hj_relative_values[idx_clipped]
            v_list.append(v_neigh)
            xr_list.append(xr)
            yr_list.append(yr)

            grad_x_list.append(self._hj_relative_values_grad[0][idx_clipped])
            grad_y_list.append(self._hj_relative_values_grad[1][idx_clipped])
            grad_theta_list.append(self._hj_relative_values_grad[2][idx_clipped])

        v_matrix = np.stack(v_list, axis=1)
        crit_neigh_idx = np.argmin(v_matrix, axis=1)

        batch_idx = np.arange(K)

        xr_crit = np.stack(xr_list, axis=1)[batch_idx, crit_neigh_idx]
        yr_crit = np.stack(yr_list, axis=1)[batch_idx, crit_neigh_idx]
        grad_x_crit = np.stack(grad_x_list, axis=1)[batch_idx, crit_neigh_idx]
        grad_y_crit = np.stack(grad_y_list, axis=1)[batch_idx, crit_neigh_idx]
        grad_theta_crit = np.stack(grad_theta_list, axis=1)[batch_idx, crit_neigh_idx]

        action_max = self.agents_controllers["agent_0"].robot.action_max
        action_min = self.agents_controllers["agent_0"].robot.action_min

        ham_v_coeff = -grad_x_crit
        ham_w_coeff = yr_crit * grad_x_crit - xr_crit * grad_y_crit - grad_theta_crit

        safe_v = np.where(ham_v_coeff > 0, action_max[0], action_min[0])
        safe_w = np.where(ham_w_coeff > 0, action_max[1], action_min[1])

        u_safe = np.stack([safe_v, safe_w], axis=-1).astype(np.float32)
        return torch.tensor(u_safe, dtype=states.dtype, device=states.device)
