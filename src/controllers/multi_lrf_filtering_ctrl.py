"""Multi-agent decentralized LRF-filtering controller using HJ reachability as the safety filter.

Instead of a simple post-facto output safety filter, this controller incorporates LRFDualGuard.
By applying the Least Restrictive Filter (LRF) projection inside the rollout/forecasting phase,
the stochastically optimized MPPI plans are guaranteed to be safe and dynamically feasible.
"""

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import torch

from controllers.dual_guard import DualGuard
from controllers.hj import HJSolver, HJSolverConfig
from controllers.mppi_ctrl import MPPICtrl, MPPICtrlParams

logger = logging.getLogger(__name__)


# ==========================================================
#  LRF-SHIELDED DUALGUARD PLANNER
# ==========================================================
class LRFDualGuard(DualGuard):
    """Shielded MPPI planner that applies the Least Restrictive Filter (LRF)

    to both sampled rollout trajectories and the final executed control command.
    """

    def __init__(self, *args, lrf_filter_function=None, **kwargs):
        # Initialize DualGuard base class with a dummy safe control function
        super().__init__(
            *args,
            safety_function=kwargs.pop("safety_function", None),
            safe_control_function=lambda x, t=0: x[:, :self.nu] * 0.0,
            **kwargs,
        )
        self.lrf_filter_function = lrf_filter_function

    def _rollout_trajectories(self, actions):
        """Simulate K trajectories over horizon T while applying LRF to the nominal samples."""
        K, T, nu = actions.shape
        assert nu == self.nu, f"Action dimension mismatch: expected {self.nu}, got {nu}"

        # Initialize starting states
        if self.state.shape == (self.nx,):
            state = self.state.unsqueeze(0).repeat(K, 1)  # (K, nx)
        else:
            state = self.state.clone()

        cost_total = torch.zeros(K, device=self.device, dtype=self.dtype)
        all_states = []
        shielded_actions_list = []

        # Rollout dynamics for each step in the horizon
        for t in range(T):
            u_nominal_t = actions[:, t]  # (K, nu)

            # Apply LRF safety shield to nominal control samples
            if self.lrf_filter_function is not None:
                u_shielded_t = self.lrf_filter_function(state, u_nominal_t, t)
            else:
                u_shielded_t = u_nominal_t

            # Ensure actions stay within bounds
            u_shielded_t = self._bound_action(u_shielded_t)
            shielded_actions_list.append(u_shielded_t)

            # Propagate dynamics
            u_apply = self.u_scale * u_shielded_t
            next_state = self._apply_dynamics(state, u_apply, t)

            # Accumulate cost
            c_t = self._apply_cost(next_state, u_apply, t)
            cost_total += c_t

            all_states.append(next_state)
            state = next_state

        states_tensor = torch.stack(all_states, dim=1)  # (K, T, nx)
        shielded_actions = torch.stack(shielded_actions_list, dim=1)  # (K, T, nu)

        self.perturbed_actions = shielded_actions
        U_expanded = self.U.unsqueeze(0).expand(K, T, nu)
        self.noise = self.perturbed_actions - U_expanded

        if self.terminal_state_cost is not None:
            cost_total += self._apply_terminal_state_cost(states_tensor)

        return cost_total, states_tensor, self.perturbed_actions

    def command(self, state, shift_nominal_trajectory=True, info=None):
        """Computes optimal control command and applies output LRF safety filter at execution time."""
        # 1. Compute optimal nominal MPPI control (weighting is computed from LRF safe rollouts)
        action_opt = super(DualGuard, self).command(
            state, shift_nominal_trajectory=shift_nominal_trajectory, info=info
        )

        # 2. Apply LRF output safety filter to the optimized command at t=0
        if self.lrf_filter_function is not None:
            # Shape of self.state is (nx,) -> expand to (1, nx)
            u_filtered = self.lrf_filter_function(self.state.unsqueeze(0), action_opt.unsqueeze(0), 0)
            action = u_filtered.squeeze(0)
        else:
            action = action_opt

        return self._bound_action(action)


# ==========================================================
#  LRF-FILTERING BACKED MPPI CONTROLLER
# ==========================================================
class _LRFFilteringMPPICtrl(MPPICtrl):
    """MPPICtrl backed by LRFDualGuard.

    `lrf_filter_function` is injected dynamically at each step by MultiLRFFilteringCtrl.
    """

    def _init_mppi_planner(self) -> LRFDualGuard:
        sigma = self.params.alpha_noise_sigma * np.diag(
            self.robot.robot_params.action_max - self.robot.robot_params.action_min
        )
        return LRFDualGuard(
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            terminal_state_cost=self.terminal_state_cost,
            nx=self.robot.robot_params.state_dim,
            noise_sigma=torch.tensor(sigma, dtype=self.dtype, device=self.device),
            num_samples=self.params.num_samples,
            horizon=self.params.horizon,
            device=self.device,
            u_min=torch.tensor(
                self.robot.robot_params.action_min, dtype=self.dtype, device=self.device
            ),
            u_max=torch.tensor(
                self.robot.robot_params.action_max, dtype=self.dtype, device=self.device
            ),
            lambda_=self.params.lambda_,
            noise_abs_cost=self.params.noise_abs_cost,
            step_dependent_dynamics=True,
            lrf_filter_function=None,  # injected per planning step
            safe_margin=0.0,
        )


# ==========================================================
#  MULTI-AGENT DECENTRALIZED LRF-FILTERING MANAGER
# ==========================================================
class MultiLRFFilteringCtrl:
    """Centralized orchestrator for decentralized multi-agent safe navigation

    using precomputed Hamilton-Jacobi (HJ) lookup tables for LRF safety filtering.
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

        # Per-agent controllers (LRF-filtering-backed MPPI)
        self.agents_controllers: Dict[str, _LRFFilteringMPPICtrl] = {
            f"agent_{i}": _LRFFilteringMPPICtrl(
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

        self.executor = ThreadPoolExecutor(max_workers=num_agents)

        # Shared HJ solver — uses agent_0's robot instance
        _ref_robot = self.agents_controllers["agent_0"].robot
        self._hj_solver = HJSolver(hj_config, _ref_robot)
        self._hj_values: Optional[np.ndarray] = None
        self._hj_values_grad: Optional[list] = None
        self._safe_margin: float = max(hj_config.superlevel_set_epsilon, 0.8)

        self._hj_relative_grid = None
        self._hj_relative_values: Optional[np.ndarray] = None
        self._hj_relative_values_grad: Optional[list] = None

    def set_goals(self, goals: Dict[str, Any]):
        """Set goal positions per agent. Format: {"agent_0": [gx, gy], ...}"""
        for key, goal_pos in goals.items():
            if key in self.agents_controllers:
                self.agents_controllers[key].set_goal(goal_pos)

    def set_maps(self, maps_deque: deque):
        """Broadcast shared risk-map deque to all agent controllers."""
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
        """Precompute the static BRT value function V(x) and its gradients."""
        self._hj_values = self._hj_solver.solve(
            1.0 - grid_map, time=time, target_time=target_time, dt=dt, epsilon=epsilon
        )
        self._hj_values_grad = np.gradient(self._hj_values)
        logger.info("Static HJ values and gradients precomputed.")

    def solve_relative(
        self,
        d_safe: float,
        time: float = 0.0,
        target_time: float = -5.0,
        dt: float = 0.05,
        epsilon: float = 0.01,
    ):
        """Precompute the relative HJI value function V_rel(x_rel) and its gradients."""
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
                break

        self._hj_relative_grid = grid
        self._hj_relative_values = np.array(values)
        self._hj_relative_values_grad = np.gradient(self._hj_relative_values)
        logger.info("Relative HJI solved and gradients precomputed.")

    def _hj_lrf_filter_batch(
        self, states: torch.Tensor, u_nominal: torch.Tensor, neighbors: list, t: int = 0
    ) -> torch.Tensor:
        """Apply highly parallelized vectorized HJ LRF safety filtering to a batch of nominal actions."""
        states_np = states.detach().cpu().numpy()
        u_nom_np = u_nominal.detach().cpu().numpy()
        K = states_np.shape[0]
        device = states.device

        # If no HJ solver precomputed grid is available, nominal controls are unmodified
        if self._hj_values is None and self._hj_relative_values is None:
            return u_nominal

        u_filtered = u_nom_np.copy()

        # Mode 2: HJI Relative Inter-Agent Avoidance Lookup
        if self._hj_relative_values is not None:
            if len(neighbors) == 0:
                return u_nominal

            # Convert neighbor states to numpy and predict their future positions at step t
            neighbors_np = []
            for n in neighbors:
                n_np = n.cpu().numpy() if torch.is_tensor(n) else np.asarray(n)
                theta_j = n_np[2]
                v_nominal = 3.0  # nominal speed of neighbors
                dx_j = v_nominal * np.cos(theta_j) * (t * self.dt)
                dy_j = v_nominal * np.sin(theta_j) * (t * self.dt)
                neighbors_np.append(np.array([n_np[0] + dx_j, n_np[1] + dy_j, theta_j], dtype=np.float32))
            grid = self._hj_relative_grid
            lo = grid.domain.lo
            dx_grid = np.array(grid.spacings)
            cells = self._hj_solver.config.domain_cells

            v_list = []
            xr_list = []
            yr_list = []
            grad_x_list = []
            grad_y_list = []
            grad_theta_list = []

            # Vectorized projection lookups over all neighbors
            for neighbor in neighbors_np:
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

            # Select critical neighbor per sample
            v_matrix = np.stack(v_list, axis=1)  # (K, N_neighbors)
            v_min = np.min(v_matrix, axis=1)  # (K,)
            crit_neigh_idx = np.argmin(v_matrix, axis=1)  # (K,)

            batch_idx = np.arange(K)
            xr_crit = np.stack(xr_list, axis=1)[batch_idx, crit_neigh_idx]
            yr_crit = np.stack(yr_list, axis=1)[batch_idx, crit_neigh_idx]
            grad_x_crit = np.stack(grad_x_list, axis=1)[batch_idx, crit_neigh_idx]
            grad_y_crit = np.stack(grad_y_list, axis=1)[batch_idx, crit_neigh_idx]
            grad_theta_crit = np.stack(grad_theta_list, axis=1)[batch_idx, crit_neigh_idx]

            # Construct relative game Hamiltonian: V_dot = A * u >= 0
            A_v = -grad_x_crit
            A_w = yr_crit * grad_x_crit - xr_crit * grad_y_crit - grad_theta_crit

            # Active filter ONLY when state is near/inside unsafe set (v_min <= safe_margin)
            unsafe_mask = v_min <= self._safe_margin
            A_norm_sq = np.maximum(A_v**2 + A_w**2, 1e-6)

            A_u = A_v * u_nom_np[:, 0] + A_w * u_nom_np[:, 1]
            violation = -A_u

            # Least Restrictive safety projection
            proj_scale = np.where(unsafe_mask & (violation > 0), violation / A_norm_sq, 0.0)
            u_filtered[:, 0] += proj_scale * A_v
            u_filtered[:, 1] += proj_scale * A_w

        # Mode 1: Static Obstacle Avoidance Lookup
        elif self._hj_values is not None:
            lo = self._hj_solver.config.domain[0]
            hi = self.hj_solver.config.domain[1]
            cells = self._hj_solver.config.domain_cells
            dx_grid = (hi - lo) / cells

            idx_np = np.round((states_np - lo) / dx_grid).astype(int)
            idx_clipped = tuple(np.clip(idx_np[:, d], 0, cells[d] - 1) for d in range(3))

            v_val = self._hj_values[idx_clipped]
            grad_x = self._hj_values_grad[0][idx_clipped]
            grad_y = self._hj_values_grad[1][idx_clipped]
            grad_theta = self._hj_values_grad[2][idx_clipped]

            theta = states_np[:, 2]
            A_v = np.cos(theta) * grad_x + np.sin(theta) * grad_y
            A_w = grad_theta

            # Active filter ONLY when state is near/inside unsafe set
            unsafe_mask = v_val <= self._safe_margin
            A_norm_sq = np.maximum(A_v**2 + A_w**2, 1e-6)

            A_u = A_v * u_nom_np[:, 0] + A_w * u_nom_np[:, 1]
            violation = -A_u

            # Least Restrictive safety projection
            proj_scale = np.where(unsafe_mask & (violation > 0), violation / A_norm_sq, 0.0)
            u_filtered[:, 0] += proj_scale * A_v
            u_filtered[:, 1] += proj_scale * A_w

        return torch.tensor(u_filtered, dtype=states.dtype, device=device)

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Decentralized HJ LRF-filtering planning step."""
        if self._hj_values is None and self._hj_relative_values is None:
            raise RuntimeError("No precomputed HJ lookup available. Call solve() or solve_relative() first.")

        # Sync states in manager
        agents_states = {
            key: np.array([obs["location"][0], obs["location"][1], obs["angle"]], dtype=np.float32)
            for key, obs in current_obs.items()
            if obs is not None and key in self.agents_controllers
        }

        def plan_agent(ego_key: str, ego_ctrl: _LRFFilteringMPPICtrl) -> tuple:
            if ego_key not in agents_states:
                return ego_key, None

            ego_state = agents_states[ego_key]

            # 1. Local sensing
            ego_pos = ego_state[:2]
            neighbors = [
                torch.tensor(state, dtype=torch.float32)
                for k, state in agents_states.items()
                if k != ego_key and np.linalg.norm(ego_pos - state[:2]) <= self.r_sense
            ]

            # 2. Wire LRF batch projection function into the agent's LRFDualGuard planner
            ego_ctrl.planner.lrf_filter_function = lambda states, u_nominal, t=0: (
                self._hj_lrf_filter_batch(states, u_nominal, neighbors, t)
            )
            ego_ctrl.planner.safe_margin = self._safe_margin

            # 3. Sync robot state
            ego_ctrl.set_state(ego_state)

            # 4. Plan
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
        """Plot weighted-mean rollout paths for every agent."""
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        for i, (key, ctrl) in enumerate(self.agents_controllers.items()):
            ctrl.visualize_rollouts(ax, color=colors[i % len(colors)], draw_samples=draw_samples)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from agents.basic_robot import RobotParams
    from agents.dubins_robot import DubinsRobot

    print("🧪 Iniciando simulación Swarm Circle Swap (10 drones) con HJ LRF Filtering Controller...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DT = 0.1
    N_DRONES = 10
    CIRCLE_RADIUS = 10.0
    D_SAFE = 2.4
    R_SENSE = 8.0
    GOAL_THRESH = 0.6

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

    hj_config = HJSolverConfig(
        system_name="dubins2d",
        domain_cells=np.array([60, 60, 36]),
        domain=np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 2 * np.pi]]),
        mode="brt",
        accuracy="medium",
        superlevel_set_epsilon=0.0,
    )

    mppi_params = MPPICtrlParams(num_samples=100, horizon=15, lambda_=1.2)
    multi_ctrl = MultiLRFFilteringCtrl(
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

    multi_ctrl.solve_relative(d_safe=D_SAFE, time=0.0, target_time=-5.0, dt=0.05, epsilon=0.01)

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

    # Dummy risk map deque for MPPI
    _xs = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, 0.5)
    _ys = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, 0.5)
    _coords = (
        np.stack(np.meshgrid(_xs, _ys, indexing="xy"), axis=-1).reshape(-1, 2).astype(np.float32)
    )
    _risk = np.zeros(len(_coords), dtype=np.float32)
    multi_ctrl.set_maps(deque([(_coords, _risk)] * mppi_params.horizon, maxlen=mppi_params.horizon))

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.set_ylim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Circular Swap - Swarm Circle Swap HJ LRF Filtering Real-time (Dubins)")

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
        patch = plt.Circle((init_state[0], init_state[1]), D_SAFE / 2.0, color=colors[i], alpha=0.6)
        ax.add_patch(patch)
        drone_patches.append(patch)
        (line,) = ax.plot(
            [init_state[0]], [init_state[1]], color=colors[i], linewidth=1.5, alpha=0.8
        )
        trail_lines.append(line)

    plt.draw()
    plt.pause(0.1)

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

            drone_patches[i].center = (state[0], state[1])
            trail_x = [pt[0] for pt in trajectories[key]]
            trail_y = [pt[1] for pt in trajectories[key]]
            trail_lines[i].set_data(trail_x, trail_y)

        plt.draw()
        plt.pause(0.001)

    print("🎉 Simulación completada.")

    # Calculate separation statistics
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

    print(f"\nDistancia mínima de separación: {min_dist_overall:.4f} metros (D_SAFE = {D_SAFE})")
    if min_dist_overall >= D_SAFE:
        print("SUCCESS: ¡Seguridad matemática de evitación inter-agente GARANTIZADA!")
    else:
        print("WARNING: Violación de la distancia de seguridad entre drones.")

    plt.ioff()
    plt.show()
