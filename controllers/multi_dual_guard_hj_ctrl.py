"""Multi-agent decentralized DualGuard controller using HJ reachability as the safety filter.

Architecture mirrors MultiDualGuardCBFCtrl:
    safety_function       = V(x)       — BRT value function   (V ≥ safe_margin ⟹ safe)
    safe_control_function = u_safe(x)  — least-restrictive filter (minimizes Hamiltonian)

Usage:
    ctrl = MultiDualGuardHJCtrl(num_agents, robot_params, robot_type, hj_config, ...)
    ctrl.set_goals(goals)
    ctrl.set_maps(maps_deque)
    ctrl.solve(grid_map)          # precompute BRT — must call before get_commands
    commands = ctrl.get_commands(obs)
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


class MultiDualGuardHJCtrl:
    """Centralized orchestrator for decentralized multi-agent safe navigation
    using HJ reachability as the DualGuard safety filter.

    Each agent holds an independent `_DualGuardMPPICtrl`. At every step,
    the shared precomputed value function V(x) is wired into each agent's
    DualGuard planner:
        safety_function       = V(x)           (grid lookup, scalar per state)
        safe_control_function = u_safe(x)      (least-restrictive filter)

    The value function is computed once (or on obstacle-map update) via solve().
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
    ):
        self.num_agents = num_agents
        self.device = device
        self.dtype = dtype
        self.dt = dt

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

        # Shared HJ solver — all agents share the same dynamics and obstacle map.
        # Uses the first agent's robot instance to build the JAX dynamics.
        _ref_robot = self.agents_controllers["agent_0"].robot
        self._hj_solver = HJSolver(hj_config, _ref_robot)
        self._hj_values: Optional[np.ndarray] = None
        self._hj_values_grad: Optional[list] = None
        self._safe_margin: float = hj_config.superlevel_set_epsilon

    # ======================================================
    #  EXTERNAL INTERFACE
    # ======================================================
    def set_goals(self, goals: Dict[str, Any]):
        """Set goal positions per agent. Format: {"agent_0": [gx, gy], ...}"""
        for key, goal_pos in goals.items():
            if key in self.agents_controllers:
                self.agents_controllers[key].set_goal(goal_pos)

    def set_maps(self, maps_deque: deque):
        """Broadcast a shared risk-map deque to all agent controllers."""
        for ctrl in self.agents_controllers.values():
            ctrl.set_maps(maps_deque)

    def solve(
        self,
        grid_map: np.ndarray,
        time: float = 0.0,
        target_time: float = -10.0,
        dt: float = 0.01,
        epsilon: float = 0.01,
    ):
        """Precompute the BRT value function V(x) for the given obstacle map.

        Must be called before get_commands(). Re-call on obstacle-map changes.

        Args:
            grid_map: 2-D binary occupancy map (0 = free, 1 = obstacle).
            time: Start time for BRT integration (typically 0).
            target_time: End time (negative = backwards in time).
            dt: Integration time step.
            epsilon: Early-stop convergence threshold.
        """
        self._hj_values = self._hj_solver.solve(
            grid_map, time=time, target_time=target_time, dt=dt, epsilon=epsilon
        )
        self._hj_values_grad = np.gradient(self._hj_values)
        logger.info("HJ value function solved; gradients precomputed.")

    def get_commands(
        self, current_obs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Decentralized HJ-shielded planning step.

        Args:
            current_obs: Observation dict keyed by agent id.
                         Each entry must contain 'location' (2,) and 'angle' (float).
                         Optional: 'velocity' (float) for unicycle agents.

        Returns:
            Dict mapping agent id -> control tensor.
        """
        if self._hj_values is None:
            raise RuntimeError(
                "Value function not available. Call solve(grid_map) first."
            )

        agents_physical = {
            key: self._extract_agent_physical(key, obs)
            for key, obs in current_obs.items()
            if obs is not None and key in self.agents_controllers
        }

        def plan_agent(ego_key: str, ego_ctrl: _DualGuardMPPICtrl) -> tuple:
            if ego_key not in agents_physical:
                return ego_key, None

            ego_ctrl.set_state(agents_physical[ego_key]["state_raw"])

            # Wire HJ safety functions into DualGuard for this step:
            #   safety_function  : V(x)       — positive outside the BRT (safe)
            #   safe_control     : u_safe(x)  — least-restrictive filter
            ego_ctrl.planner.safety_function = self._hj_safety_function_batch
            ego_ctrl.planner.safe_control_function = self._hj_safe_control_batch
            ego_ctrl.planner.safe_margin = self._safe_margin

            return ego_key, ego_ctrl.get_command()

        commands: Dict[str, torch.Tensor] = {}

        if self.num_agents == 1:
            for key, ctrl in self.agents_controllers.items():
                k, cmd = plan_agent(key, ctrl)
                if cmd is not None:
                    commands[k] = cmd
        else:
            with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
                futures = [
                    executor.submit(plan_agent, key, ctrl)
                    for key, ctrl in self.agents_controllers.items()
                ]
                for future in futures:
                    key, cmd = future.result()
                    if cmd is not None:
                        commands[key] = cmd

        return commands

    def visualize_rollouts(self, ax, draw_samples: bool = False):
        """Plot weighted-mean rollout paths for every agent."""
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for i, (key, ctrl) in enumerate(self.agents_controllers.items()):
            ctrl.visualize_rollouts(
                ax, color=colors[i % len(colors)], draw_samples=draw_samples
            )

    # ======================================================
    #  HJ BATCHED WRAPPERS  (torch Tensor <-> numpy grid)
    # ======================================================
    def _hj_safety_function_batch(
        self, states: torch.Tensor, t: int = 0
    ) -> torch.Tensor:
        """V(x) grid lookup for K states. Returns (K,) tensor.

        V(x) >= safe_margin => state is safe (outside the BRT).
        """
        states_np = states.detach().cpu().numpy()
        values = np.array(
            [
                self._hj_solver.check_if_safe(states_np[k])[1]
                for k in range(states_np.shape[0])
            ],
            dtype=np.float32,
        )
        return torch.tensor(values, dtype=states.dtype, device=states.device)

    def _hj_safe_control_batch(
        self, states: torch.Tensor, t: int = 0
    ) -> torch.Tensor:
        """Least-restrictive control u_safe(x) for K states. Returns (K, nu) tensor."""
        states_np = states.detach().cpu().numpy()
        K = states_np.shape[0]
        nu = self.agents_controllers["agent_0"].robot.robot_params.action_dim
        u_safe = np.zeros((K, nu), dtype=np.float32)

        for k in range(K):
            action, _, _ = self._hj_solver.compute_least_restrictive_control(
                states_np[k], self._hj_values, self._hj_values_grad
            )
            if action is not None:
                u_safe[k] = action[:nu]  # guard against dimension mismatch

        return torch.tensor(u_safe, dtype=states.dtype, device=states.device)

    # ======================================================
    #  INTERNAL HELPERS
    # ======================================================
    def _extract_agent_physical(
        self, key: str, obs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract the minimal physical snapshot needed per step."""
        pos = np.asarray(obs["location"], dtype=np.float32)
        theta = float(obs["angle"])

        if "velocity" in obs:
            speed = float(obs["velocity"])
            state_raw = np.array([pos[0], pos[1], theta, speed], dtype=np.float32)
        else:
            ctrl = self.agents_controllers[key]
            speed = float(ctrl.robot.robot_params.action_max[0]) / 2.0
            state_raw = np.array([pos[0], pos[1], theta], dtype=np.float32)

        return {
            "p_center": torch.tensor(pos, dtype=torch.float32),
            "v_center": torch.tensor(
                [speed * np.cos(theta), speed * np.sin(theta)], dtype=torch.float32
            ),
            "state_raw": state_raw,
        }
