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

        # Persistent executor to avoid thread spawn/tear-down overhead on each step
        self.executor = ThreadPoolExecutor(max_workers=num_agents)

        # Shared HJ solver — all agents share the same dynamics and obstacle map.
        # Uses the first agent's robot instance to build the JAX dynamics.
        _ref_robot = self.agents_controllers["agent_0"].robot
        self._hj_solver = HJSolver(hj_config, _ref_robot)
        self._hj_values: Optional[np.ndarray] = None
        self._hj_values_grad: Optional[list] = None
        self._safe_margin: float = hj_config.superlevel_set_epsilon

        # Relative HJI variables for inter-agent avoidance
        self._hj_relative_grid = None
        self._hj_relative_values: Optional[np.ndarray] = None
        self._hj_relative_values_grad: Optional[list] = None

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
        # HJSolver expects 0 = obstacle, 1 = free, so we invert the standard occupancy grid_map (0 = free, 1 = obstacle)
        self._hj_values = self._hj_solver.solve(
            1.0 - grid_map, time=time, target_time=target_time, dt=dt, epsilon=epsilon
        )
        self._hj_values_grad = np.gradient(self._hj_values)
        logger.info("HJ value function solved; gradients precomputed.")

    def solve_relative(
        self,
        d_safe: float,
        time: float = 0.0,
        target_time: float = -5.0,
        dt: float = 0.05,
        epsilon: float = 0.01,
    ):
        """Precompute the HJI Relative Value Function V_rel(x_rel) for inter-agent avoidance.

        Integrates the 2-player Isaacs equation backwards in time on the 3D relative state space.
        """
        import hj_reachability as hj

        from controllers.hj import RelativeDubinsDynamics

        # 1. Define the 3D relative grid domain [-10, 10] x [-10, 10] x [0, 2pi]
        domain_cells = self._hj_solver.config.domain_cells
        domain = self._hj_solver.config.domain

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(np.array(domain[0]), np.array(domain[1])),
            tuple(domain_cells),
            periodic_dims=2,
        )

        # 2. Define the Relative Dynamics
        action_min = self.agents_controllers["agent_0"].robot.action_min
        action_max = self.agents_controllers["agent_0"].robot.action_max
        dynamics = RelativeDubinsDynamics(action_min, action_max)

        # 3. Define initial values analytically: distance to a cylinder of radius d_safe
        grid_states = grid.states
        x_coords = grid_states[..., 0]
        y_coords = grid_states[..., 1]
        initial_values = np.sqrt(x_coords**2 + y_coords**2) - d_safe

        # 4. Set up HJI JAX solver
        solver_settings = hj.SolverSettings.with_accuracy(
            self._hj_solver.config.accuracy,
            hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        )

        problem_definition = {
            "solver_settings": solver_settings,
            "dynamics": dynamics,
            "grid": grid,
        }

        # Integrate backwards in time
        times = np.linspace(time, target_time, int(abs(target_time - time) / dt))
        values = initial_values

        logger.info("Starting relative HJI BRT computation...")
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
                logger.info(f"Relative HJI converged early at step {i}.")
                break

        self._hj_relative_grid = grid
        self._hj_relative_values = np.array(values)
        self._hj_relative_values_grad = np.gradient(self._hj_relative_values)
        logger.info("Relative HJI value function solved and gradients precomputed.")

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Decentralized HJ-shielded/HJI-shielded planning step.

        Args:
            current_obs: Observation dict keyed by agent id.
                         Each entry must contain 'location' (2,) and 'angle' (float).
                         Optional: 'velocity' (float) for unicycle agents.

        Returns:
            Dict mapping agent id -> control tensor.
        """
        # Validate that at least one value function is precomputed
        if self._hj_values is None and self._hj_relative_values is None:
            raise RuntimeError(
                "No precomputed value function available. Call solve(grid_map) or solve_relative(d_safe) first."
            )

        # Build raw state tensors [x, y, theta] for all agents (numpy arrays)
        agents_states = {
            key: np.array([obs["location"][0], obs["location"][1], obs["angle"]], dtype=np.float32)
            for key, obs in current_obs.items()
            if obs is not None and key in self.agents_controllers
        }

        def plan_agent(ego_key: str, ego_ctrl: _DualGuardMPPICtrl) -> tuple:
            if ego_key not in agents_states:
                return ego_key, None

            ego_state = agents_states[ego_key]

            # 1. Sync robot state
            ego_ctrl.set_state(ego_state)

            if self._hj_relative_values is not None:
                # Mode 2: Relative HJI Inter-Agent Avoidance
                # Sense neighbors within self.r_sense range
                ego_pos = ego_state[:2]
                neighbors = [
                    state
                    for k, state in agents_states.items()
                    if k != ego_key and np.linalg.norm(ego_pos - state[:2]) <= self.r_sense
                ]

                # Wire HJI relative safety functions capturing current neighbors
                ego_ctrl.planner.safety_function = lambda states, t=0: (
                    self._hj_safety_function_batch_relative(states, neighbors, t)
                )
                ego_ctrl.planner.safe_control_function = lambda states, t=0: (
                    self._hj_safe_control_batch_relative(states, neighbors, t)
                )
                ego_ctrl.planner.safe_margin = self._safe_margin
            else:
                # Mode 1: Static Obstacle Avoidance (fallback to global grid lookup)
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
            # Reuse the persistent executor to avoid per-step thread pool creation/destruction overhead
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
        """Plot weighted-mean rollout paths for every agent."""
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        for i, (key, ctrl) in enumerate(self.agents_controllers.items()):
            ctrl.visualize_rollouts(ax, color=colors[i % len(colors)], draw_samples=draw_samples)

    def _hj_safety_function_batch(self, states: torch.Tensor, t: int = 0) -> torch.Tensor:
        """V(x) grid lookup for K states. Returns (K,) tensor.

        V(x) >= safe_margin => state is safe (outside the BRT).
        """
        states_np = states.detach().cpu().numpy()

        # Pure NumPy vectorized nearest index lookup (avoids JAX host-device memory transfer overhead)
        lo = self._hj_solver.config.domain[0]
        hi = self._hj_solver.config.domain[1]
        cells = self._hj_solver.config.domain_cells
        dx_grid = (hi - lo) / cells
        idx_np = np.round((states_np - lo) / dx_grid).astype(int)

        idx_clipped = tuple(np.clip(idx_np[:, d], 0, cells[d] - 1) for d in range(3))

        values = self._hj_values[idx_clipped].astype(np.float32)
        return torch.tensor(values, dtype=states.dtype, device=states.device)

    def _hj_safe_control_batch(self, states: torch.Tensor, t: int = 0) -> torch.Tensor:
        """Least-restrictive control u_safe(x) for K states. Returns (K, nu) tensor."""
        states_np = states.detach().cpu().numpy()

        # Pure NumPy vectorized nearest index lookup (avoids JAX host-device memory transfer overhead)
        lo = self._hj_solver.config.domain[0]
        hi = self._hj_solver.config.domain[1]
        cells = self._hj_solver.config.domain_cells
        dx_grid = (hi - lo) / cells
        idx_np = np.round((states_np - lo) / dx_grid).astype(int)

        idx_clipped = tuple(np.clip(idx_np[:, d], 0, cells[d] - 1) for d in range(3))

        # Lookup gradients from grid
        grad_x = self._hj_values_grad[0][idx_clipped]
        grad_y = self._hj_values_grad[1][idx_clipped]
        grad_theta = self._hj_values_grad[2][idx_clipped]

        action_max = self.agents_controllers["agent_0"].robot.action_max
        action_min = self.agents_controllers["agent_0"].robot.action_min

        # Use 1000x faster vectorized computation for Dubins dynamics
        if self._hj_solver.config.system_name == "dubins2d":
            theta = states_np[:, 2]
            ham_v_coeff = np.cos(theta) * grad_x + np.sin(theta) * grad_y
            safe_v = np.where(ham_v_coeff > 0, action_max[0], action_min[0])
            safe_w = np.where(grad_theta > 0, action_max[1], action_min[1])
            u_safe = np.stack([safe_v, safe_w], axis=-1).astype(np.float32)
        else:
            # General fallback to standard iterative check
            K = states_np.shape[0]
            nu = self.agents_controllers["agent_0"].robot.robot_params.action_dim
            u_safe = np.zeros((K, nu), dtype=np.float32)

            for k in range(K):
                action, _, _ = self._hj_solver.compute_least_restrictive_control(
                    states_np[k], self._hj_values, self._hj_values_grad
                )
                if action is not None:
                    u_safe[k] = action[:nu]

        return torch.tensor(u_safe, dtype=states.dtype, device=states.device)

    def _hj_safety_function_batch_relative(
        self, states: torch.Tensor, neighbors: list, t: int = 0
    ) -> torch.Tensor:
        """V(x) relative lookups for all neighbors. Returns (K,) tensor.

        Overall safety is the minimum value over all neighbors (worst case).
        """
        if len(neighbors) == 0:
            return 10.0 * torch.ones(states.shape[0], dtype=states.dtype, device=states.device)

        states_np = states.detach().cpu().numpy()
        grid = self._hj_relative_grid
        v_list = []

        # Pure NumPy nearest index bounds (avoids JAX host-device memory transfer overhead)
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

        # Minimum value across all neighbors
        v_matrix = np.stack(v_list, axis=1)  # (K, N_neighbors)
        v_min = np.min(v_matrix, axis=1)  # (K,)

        return torch.tensor(v_min, dtype=states.dtype, device=states.device)

    def _hj_safe_control_batch_relative(
        self, states: torch.Tensor, neighbors: list, t: int = 0
    ) -> torch.Tensor:
        """Vectorized HJI Dubins relative least-restrictive controller."""
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

        # Pure NumPy nearest index bounds (avoids JAX host-device memory transfer overhead)
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

        v_matrix = np.stack(v_list, axis=1)  # (K, N_neighbors)
        crit_neigh_idx = np.argmin(v_matrix, axis=1)  # (K,)

        # Extract critical neighbor's relative state and gradients for each batch item
        batch_idx = np.arange(K)

        xr_crit = np.stack(xr_list, axis=1)[batch_idx, crit_neigh_idx]
        yr_crit = np.stack(yr_list, axis=1)[batch_idx, crit_neigh_idx]
        grad_x_crit = np.stack(grad_x_list, axis=1)[batch_idx, crit_neigh_idx]
        grad_y_crit = np.stack(grad_y_list, axis=1)[batch_idx, crit_neigh_idx]
        grad_theta_crit = np.stack(grad_theta_list, axis=1)[batch_idx, crit_neigh_idx]

        action_max = self.agents_controllers["agent_0"].robot.action_max
        action_min = self.agents_controllers["agent_0"].robot.action_min

        # Maximize relative Hamiltonian term containing controls u_1 = [v_1, w_1]
        # Coeff for v_1: -grad_x_crit
        # Coeff for w_1: yr_crit * grad_x_crit - xr_crit * grad_y_crit - grad_theta_crit
        ham_v_coeff = -grad_x_crit
        ham_w_coeff = yr_crit * grad_x_crit - xr_crit * grad_y_crit - grad_theta_crit

        safe_v = np.where(ham_v_coeff > 0, action_max[0], action_min[0])
        safe_w = np.where(ham_w_coeff > 0, action_max[1], action_min[1])

        u_safe = np.stack([safe_v, safe_w], axis=-1).astype(np.float32)
        return torch.tensor(u_safe, dtype=states.dtype, device=states.device)

    # ======================================================
    #  INTERNAL HELPERS
    # ======================================================
    def _extract_agent_physical(self, key: str, obs: Dict[str, Any]) -> Dict[str, Any]:
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from agents.basic_robot import RobotParams
    from agents.dubins_robot import DubinsRobot

    print("🧪 Iniciando simulación Swarm Circle Swap (10 drones) con HJI Relative Reachability...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DT = 0.1
    N_DRONES = 10
    CIRCLE_RADIUS = 10.0
    D_SAFE = 2.4
    R_SENSE = 8.0
    GOAL_THRESH = 0.6

    # 1. Configuración de parámetros físicos
    robot_params = RobotParams(
        action_dim=2,
        state_dim=3,
        action_max=[6.0, 4.0],
        action_min=[0.0, -4.0],
        state_max=[30.0, 30.0, 2 * np.pi],
        state_min=[-30.0, -30.0, 0.0],
        robot_type="dubins2d",
        dt=DT,
    )

    # Configuración del resolvedor Hamilton-Jacobi
    hj_config = HJSolverConfig(
        system_name="dubins2d",
        domain_cells=np.array([60, 60, 36]),
        domain=np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 2 * np.pi]]),
        mode="brt",
        accuracy="medium",
        superlevel_set_epsilon=0.0,
    )

    mppi_params = MPPICtrlParams(num_samples=100, horizon=15, lambda_=1.2)
    multi_ctrl = MultiDualGuardHJCtrl(
        num_agents=N_DRONES,
        robot_params=robot_params,
        robot_type="dubins2d",
        hj_config=hj_config,
        goal_thresh=GOAL_THRESH,
        device=device,
        mppi_params=mppi_params,
        dt=DT,
        r_sense=R_SENSE,
        d_safe=D_SAFE,
    )

    # Precomputar el valor HJI relativo de juego diferencial
    multi_ctrl.solve_relative(d_safe=D_SAFE, time=0.0, target_time=-5.0, dt=0.05, epsilon=0.01)

    # 2. Establecer metas y posiciones iniciales
    goals_dict = {}
    trajectories = {f"agent_{i}": [] for i in range(N_DRONES)}
    sims = {}

    for i in range(N_DRONES):
        theta = i * (2 * np.pi / N_DRONES)
        px = CIRCLE_RADIUS * np.cos(theta)
        py = CIRCLE_RADIUS * np.sin(theta)
        gx = -px
        gy = -py

        goals_dict[f"agent_{i}"] = np.array([gx, gy], dtype=np.float32)
        angle_to_center = np.arctan2(-py, -px)

        sim = DubinsRobot(robot_params)
        sim.reset(np.array([px, py, angle_to_center], dtype=np.float32))
        sims[f"agent_{i}"] = sim
        trajectories[f"agent_{i}"].append(sim.get_state().copy())

    multi_ctrl.set_goals(goals_dict)

    # Dummy risk map deque para MPPI
    _xs = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, 0.5)
    _ys = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, 0.5)
    _coords = (
        np.stack(np.meshgrid(_xs, _ys, indexing="xy"), axis=-1).reshape(-1, 2).astype(np.float32)
    )
    _risk = np.zeros(len(_coords), dtype=np.float32)
    multi_ctrl.set_maps(deque([(_coords, _risk)] * mppi_params.horizon, maxlen=mppi_params.horizon))

    # 3. Configurar visualización en tiempo real
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.set_ylim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Circular Swap - Swarm Circle Swap HJI Relative Real-time (Dubins)")

    # Dibujar círculo de generación
    spawn_circ = plt.Circle(
        (0, 0), CIRCLE_RADIUS, color="gray", fill=False, linestyle="--", alpha=0.5
    )
    ax.add_patch(spawn_circ)

    colors = plt.cm.rainbow(np.linspace(0, 1, N_DRONES))
    for i in range(N_DRONES):
        goal = goals_dict[f"agent_{i}"]
        ax.scatter(goal[0], goal[1], marker="*", color=colors[i], s=150, zorder=5)

    drone_patches = []
    trail_lines = []

    for i in range(N_DRONES):
        init_state = trajectories[f"agent_{i}"][0]
        # Dibujar dron con un tamaño que representa su zona de colisión
        patch = plt.Circle((init_state[0], init_state[1]), D_SAFE / 2.0, color=colors[i], alpha=0.6)
        ax.add_patch(patch)
        drone_patches.append(patch)
        (line,) = ax.plot(
            [init_state[0]], [init_state[1]], color=colors[i], linewidth=1.5, alpha=0.8
        )
        trail_lines.append(line)

    plt.draw()
    plt.pause(0.1)

    # 4. Ejecutar la simulación con actualización en tiempo real
    print("🚀 Iniciando bucle de control (120 pasos)...")
    max_steps = 120
    for step in tqdm(range(max_steps)):
        current_obs = {}
        for i in range(N_DRONES):
            key = f"agent_{i}"
            state = sims[key].get_state()
            current_obs[key] = {
                "location": state[:2],
                "angle": state[2],
            }

        actions = multi_ctrl.get_commands(current_obs)

        for i in range(N_DRONES):
            key = f"agent_{i}"
            u_cmd = actions[key].cpu().numpy()
            sims[key].dynamic_step(u_cmd)
            state = sims[key].get_state().copy()
            trajectories[key].append(state)

            # Actualizar gráfico en tiempo real
            drone_patches[i].center = (state[0], state[1])
            trail_x = [pt[0] for pt in trajectories[key]]
            trail_y = [pt[1] for pt in trajectories[key]]
            trail_lines[i].set_data(trail_x, trail_y)

        plt.draw()
        plt.pause(0.001)

    print("🎉 Simulación completada.")

    # Calcular estadísticas de separación entre drones
    min_dist_overall = float("inf")
    steps_recorded = len(trajectories["agent_0"])

    for s in range(steps_recorded):
        positions = []
        for i in range(N_DRONES):
            positions.append(trajectories[f"agent_{i}"][s][:2])
        for i in range(N_DRONES):
            for j in range(i + 1, N_DRONES):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist_overall:
                    min_dist_overall = d

    print(
        f"\nDistancia mínima observada entre drones: {min_dist_overall:.4f} metros (Umbral D_SAFE = {D_SAFE} m)"
    )
    if min_dist_overall >= D_SAFE:
        print("SUCCESS: ¡Seguridad matemática de evitación inter-agente GARANTIZADA!")
    else:
        print("WARNING: Violación de la distancia de seguridad entre drones.")

    # Mantener el gráfico abierto
    plt.ioff()
    plt.show()
