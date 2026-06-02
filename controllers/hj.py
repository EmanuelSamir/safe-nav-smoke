from dataclasses import dataclass, field
from time import time as time_pkg

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
import skfmm
from hj_reachability import dynamics, sets
from matplotlib import pyplot as plt


# ============================================================
# CONFIG
# ============================================================
@dataclass
class HJSolverConfig:
    system_name: str  # e.g. "dubins2d", "dubins2d_fixed_velocity", "unicycle2d"
    domain_cells: np.ndarray  # e.g. [x_res, y_res, theta_res]
    domain: np.ndarray  # e.g. [[x_min, y_min, θ_min], [x_max, y_max, θ_max]]
    mode: str  # "brs" or "brt"
    accuracy: str  # "low", "medium", "high", "very_high"
    superlevel_set_epsilon: float = 0.0
    converged_values: np.ndarray | None = field(default_factory=lambda: None)
    until_convergent: bool = True
    print_progress: bool = True
    warm_start: bool = False
    action_bounds: np.ndarray = field(
        default_factory=lambda: (np.array([0.0, -4.0]), np.array([8.0, 4.0]))
    )
    disturbance_bounds: np.ndarray = field(
        default_factory=lambda: (np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    )


# ============================================================
# DYNAMICS: Generic Agent JAX Dynamics
# ============================================================
class AgentHJDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """Generic HJ Reachability Affine Dynamics that delegates to a Robot agent."""

    def __init__(self, robot, control_mode="max", disturbance_mode="min"):
        # Configure action space using the robot's physical configuration bounds
        control_space = sets.Box(
            jnp.array(robot.robot_params.action_min),
            jnp.array(robot.robot_params.action_max)
        )
        
        # Configure disturbance space (defaults to zero bounds)
        dist_min = robot.robot_params.action_min * 0.0
        dist_max = robot.robot_params.action_max * 0.0
        disturbance_space = sets.Box(jnp.array(dist_min), jnp.array(dist_max))
        
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)
        self.robot = robot

    def open_loop_dynamics(self, state, time):
        return self.robot.open_loop_dynamics_jnp(state, time)

    def control_jacobian(self, state, time):
        return self.robot.control_jacobian_jnp(state, time)

    def disturbance_jacobian(self, state, time):
        # Affine dynamics require a disturbance jacobian. Defaults to zero block.
        return jnp.zeros((state.shape[0], self.disturbance_space.ndim))


class RelativeDubinsDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """JAX dynamics for the relative state between two identical Dubins cars.

    States:
        xr, yr, thr
    Control (Ego):
        u = [v_1, w_1] -> [v, omega]
    Disturbance (Opponent):
        d = [v_2, w_2] -> [v, omega]

    Relative kinematics:
        d xr/dt   = -v_1 + v_2 * cos(thr) + w_1 * yr
        d yr/dt   = v_2 * sin(thr) - w_1 * xr
        d thr/dt  = w_2 - w_1
    """

    def __init__(self, action_min, action_max, control_mode="max", disturbance_mode="min"):
        control_space = sets.Box(jnp.array(action_min), jnp.array(action_max))
        disturbance_space = sets.Box(jnp.array(action_min), jnp.array(action_max))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.zeros_like(state)

    def control_jacobian(self, state, time):
        xr, yr, thr = state[0], state[1], state[2]
        return jnp.array([
            [-1.0, yr],
            [0.0, -xr],
            [0.0, -1.0]
        ])

    def disturbance_jacobian(self, state, time):
        xr, yr, thr = state[0], state[1], state[2]
        return jnp.array([
            [jnp.cos(thr), 0.0],
            [jnp.sin(thr), 0.0],
            [0.0, 1.0]
        ])


# ============================================================
# SOLVER
# ============================================================
class HJSolver:
    def __init__(self, config: HJSolverConfig, robot):
        self.config = config
        self.robot = robot
        self.problem_definition = None
        self.initial_values = None
        self.last_values = config.converged_values
        self.last_grid_map = None
        self.changed_grid_map = None
        self.processed_updates = []

    # ------------------ CORE BUILDERS ------------------
    def get_dynamics(self):
        return AgentHJDynamics(self.robot)

    def get_solver_settings(self, accuracy="low", mode="brt"):
        if mode not in ["brs", "brt"]:
            raise ValueError("Mode must be 'brs' or 'brt'.")
        if accuracy not in ["low", "medium", "high", "very_high"]:
            raise ValueError("Invalid accuracy level.")
        if mode == "brs":
            return hj.SolverSettings.with_accuracy(accuracy)
        return hj.SolverSettings.with_accuracy(
            accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
        )

    def get_domain_grid(self, domain, domain_cells):
        return hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(np.array(domain[0]), np.array(domain[1])),
            tuple(domain_cells),
            periodic_dims=2,
        )

    def get_problem_definition(self):
        return {
            "solver_settings": self.get_solver_settings(self.config.accuracy, self.config.mode),
            "dynamics": self.get_dynamics(),
            "grid": self.get_domain_grid(self.config.domain, self.config.domain_cells),
        }

    # ------------------ VALUE INITIALIZATION ------------------
    def compute_initial_values(self, grid_map: np.ndarray, dx: float = 0.1) -> np.ndarray:
        """Compute initial signed distance l(x) where 0 = obstacle.
        The shape of the output depends on the system dimension.
        """
        system = self.config.system_name

        # 1️⃣ Signed distance transform (2D map)
        dist = skfmm.distance(grid_map - 0.5, dx=dx)  # shape: (Ny, Nx)

        # 2️⃣ Extend along angular dimension (θ)
        num_theta = self.config.domain_cells[2]
        dist_3d = np.repeat(dist[:, :, np.newaxis], num_theta, axis=2)  # (Ny, Nx, Nθ)

        # 3️⃣ Extend along velocity dimension (v) — only for unicycle2d
        if "unicycle" in system:
            num_v = self.config.domain_cells[3]
            dist_4d = np.repeat(dist_3d[:, :, :, np.newaxis], num_v, axis=3)  # (Ny, Nx, Nθ, Nv)
            return dist_4d

        return dist_3d

    def compute_warm_start_values(self, grid_map: np.ndarray) -> np.ndarray:
        """Fuse previous V(x) with new obstacle map."""
        l_x = self.compute_initial_values(grid_map)
        warm_values = self.last_values.copy()
        changed = np.where(self.last_grid_map != grid_map)
        warm_values[changed] = l_x[changed]
        self.changed_grid_map = np.zeros_like(warm_values, dtype=np.uint8)
        self.changed_grid_map[changed] = 1
        return warm_values

    # ------------------ SOLVING ------------------
    def solve(self, grid_map, time=0.0, target_time=-10.0, dt=0.01, epsilon=0.01):
        if grid_map is None:
            raise ValueError("Grid map not provided.")

        self.initial_values = self.compute_initial_values(grid_map)

        if self.config.warm_start and self.last_values is not None:
            self.initial_values = self.compute_warm_start_values(grid_map)

        self.last_grid_map = grid_map

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition()

        times = np.linspace(time, target_time, int(abs(target_time - time) / dt))
        print("Starting BRT computation...") if self.config.print_progress else None

        values = self.initial_values
        start_t = time_pkg()

        for i in range(1, len(times)):
            values_new = hj.step(
                **self.problem_definition,
                time=times[i - 1],
                values=values,
                target_time=times[i],
                progress_bar=False,
            )
            diff = np.max(np.abs(values_new - values))
            values = values_new
            if self.config.print_progress:
                print(f"[{i}/{len(times)}] ΔV={diff:.4f}") if self.config.print_progress else None
            if self.config.until_convergent and diff < epsilon:
                print("Converged early.") if self.config.print_progress else None
                break

        self.last_values = np.array(values)
        print(f"Total time: {time_pkg() - start_t:.2f}s") if self.config.print_progress else None
        return self.last_values

    # ------------------ SAFETY CHECKS ------------------
    def _state_to_grid(self, state):
        grid = self.problem_definition["grid"]
        ind = np.clip(grid.nearest_index(state), 0, np.array(self.config.domain_cells) - 1)
        return np.array(ind, dtype=int)

    def check_if_safe(self, state, values=None):
        if values is None:
            values = self.last_values
        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition()
        idx = self._state_to_grid(state)
        idx = tuple(idx)
        v = values[idx]
        init_v = self.initial_values[idx] if self.initial_values is not None else None
        return v > self.config.superlevel_set_epsilon, v, init_v

    # ------------------ CONTROL COMPUTATION ------------------
    def compute_least_restrictive_control(self, state, values=None, values_grad=None):
        """Compute the least restrictive control action based on the values and values gradients.
        Delegates physically specific steering control actions to the respective agent instance.

        Args:
            state: The current state of the system.
            values: The values of the system.
            values_grad: The gradients of the values.

        Returns:
            action: The least restrictive control action.
            value: The value of the system.
            value_grad: The gradients of the values.
        """
        if values is None:
            if self.last_values is None:
                return None, None, None
            values = self.last_values

        if values_grad is None:
            values_grad = np.gradient(values)

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition()

        state = np.array(state)
        state_ind = self._state_to_grid(state)
        idx = tuple(state_ind)

        value = values[idx]
        grad_x = values_grad[0][idx]
        grad_y = values_grad[1][idx]
        grad_theta = values_grad[2][idx]

        # Extract gradient parts matching state dimensionality
        if "unicycle" in self.config.system_name:
            grad_v = values_grad[3][idx]
            value_grad_array = np.array([grad_x, grad_y, grad_theta, grad_v])
        else:
            value_grad_array = np.array([grad_x, grad_y, grad_theta])

        # Delegate physical steering safety action selection to the agent directly!
        action = self.robot.hj_safe_control(state, value_grad_array)

        return action, value, value_grad_array

    # ------------------ PLOTTING ------------------
    def plot_zero_level(self, grid_data, grid_map=None, title="HJ 0-Level Set"):
        x_res, y_res, _ = self.config.domain_cells
        x = np.linspace(self.config.domain[0][0], self.config.domain[1][0], x_res)
        y = np.linspace(self.config.domain[0][1], self.config.domain[1][1], y_res)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        if grid_map is not None:
            ax.imshow(
                grid_map, cmap="gray", origin="lower", extent=[x.min(), x.max(), y.min(), y.max()]
            )
        cs = ax.contour(X, Y, grid_data[:, :, 0].T, levels=[0], colors="red")
        ax.clabel(cs, fmt="%2.1f", colors="black", fontsize=8)
        ax.set_title(title)
        plt.show()
