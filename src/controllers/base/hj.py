from dataclasses import dataclass
from time import time as time_pkg

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
import skfmm
from hj_reachability import dynamics, sets


# ============================================================
# CONFIG
# ============================================================
@dataclass
class HJSolverConfig:
    domain_cells: np.ndarray  # e.g. [x_res, y_res, theta_res]
    domain: np.ndarray  # e.g. [[x_min, y_min, θ_min], [x_max, y_max, θ_max]]
    accuracy: str  # "low", "medium", "high", "very_high"
    target_time: float = -10.0
    dt: float = 0.01
    epsilon: float = 0.01
    dx: float = 0.1
    superlevel_set_epsilon: float = 0.0
    until_convergent: bool = True
    print_progress: bool = True


# ============================================================
# DYNAMICS: Generic Agent JAX Dynamics
# ============================================================
class AgentHJDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """Generic HJ Reachability Affine Dynamics that delegates to a Robot agent."""

    def __init__(self, robot, control_mode="max", disturbance_mode="min"):
        # Configure action space using the robot's physical configuration bounds
        control_space = sets.Box(
            jnp.array(robot.robot_params.action_min), jnp.array(robot.robot_params.action_max)
        )

        # Configure disturbance space (defaults to zero bounds)
        dist_min = jnp.array(robot.robot_params.action_min) * 0.0
        dist_max = jnp.array(robot.robot_params.action_max) * 0.0
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


# ============================================================
# SOLVER
class HJSolver:
    def __init__(self, config: HJSolverConfig, robot):
        self.config = config
        self.robot = robot
        self.problem_definition = None
        self.initial_values = None
        self.last_values = None
        self.last_values_grad = None

    # ------------------ CORE BUILDERS ------------------
    def get_dynamics(self):
        return AgentHJDynamics(self.robot)

    def get_solver_settings(self, accuracy="low"):
        if accuracy not in ["low", "medium", "high", "very_high"]:
            raise ValueError("Invalid accuracy level.")
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
            "solver_settings": self.get_solver_settings(self.config.accuracy),
            "dynamics": self.get_dynamics(),
            "grid": self.get_domain_grid(self.config.domain, self.config.domain_cells),
        }

    # ------------------ VALUE INITIALIZATION ------------------
    # ------------------ VALUE INITIALIZATION ------------------
    def compute_initial_values(self, grid_map: np.ndarray) -> np.ndarray:
        """Compute initial signed distance l(x) where 0 = obstacle.
        The shape of the output depends on the grid domain dimension.
        """
        # Signed distance transform (2D map)
        dist = skfmm.distance(grid_map - 0.5, dx=self.config.dx)  # shape: (Ny, Nx)

        # Dynamically repeat along any extra dimensions
        values = dist
        for axis_idx in range(2, len(self.config.domain_cells)):
            num_cells = self.config.domain_cells[axis_idx]
            values = np.repeat(values[..., np.newaxis], num_cells, axis=axis_idx)
        return values

    # ------------------ SOLVING ------------------
    def solve(self, grid_map, time=0.0):
        if grid_map is None:
            raise ValueError("Grid map not provided.")

        self.initial_values = self.compute_initial_values(grid_map)

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition()

        target_time = self.config.target_time
        dt = self.config.dt
        epsilon = self.config.epsilon

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
            # Avoid CPU-GPU sync bottleneck by using jnp
            diff = jnp.max(jnp.abs(values_new - values)).item()
            values = values_new
            if self.config.print_progress:
                print(f"[{i}/{len(times)}] ΔV={diff:.4f}")
            if self.config.until_convergent and diff < epsilon:
                print("Converged early.") if self.config.print_progress else None
                break

        self.last_values = values
        self.last_values_grad = jnp.gradient(self.last_values)
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
        v = float(values[idx])
        init_v = float(self.initial_values[idx]) if self.initial_values is not None else None
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
            if self.last_values_grad is None:
                self.last_values_grad = jnp.gradient(values)
            values_grad = self.last_values_grad

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition()

        state = np.array(state)
        state_ind = self._state_to_grid(state)
        idx = tuple(state_ind)

        value = float(values[idx])

        # Extract gradient components dynamically matching state space dimensionality
        dim = len(self.config.domain_cells)
        grad_list = []
        for d in range(dim):
            g = values_grad[d][idx]
            grad_list.append(g if isinstance(g, (float, int)) else float(g))
        value_grad_array = np.array(grad_list)

        # Delegate physical steering safety action selection to the agent directly!
        action = self.robot.hj_safe_control(state, value_grad_array)

        return action, value, value_grad_array

    # ------------------ PLOTTING ------------------
    def plot_zero_level(self, grid_data, grid_map=None, title="HJ 0-Level Set"):
        from matplotlib import pyplot as plt

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


if __name__ == "__main__":
    print("Running HJSolver initialization and sanity check...")

    class MockRobotParams:
        def __init__(self):
            self.action_min = [0.0, -4.0]
            self.action_max = [8.0, 4.0]

    class MockRobot:
        def __init__(self):
            self.robot_params = MockRobotParams()

        def open_loop_dynamics_jnp(self, state, time):
            return jnp.zeros_like(state)

        def control_jacobian_jnp(self, state, time):
            return jnp.array([[jnp.cos(state[2]), 0.0], [jnp.sin(state[2]), 0.0], [0.0, 1.0]])

        def hj_safe_control(self, state, value_grad_array):
            print("hj_safe_control called with grad:", value_grad_array)
            return np.array([4.0, 0.0])

    config = HJSolverConfig(
        domain_cells=np.array([20, 20, 18]),
        domain=np.array([[-10.0, -10.0, -np.pi], [10.0, 10.0, np.pi]]),
        accuracy="low",
        target_time=-0.1,
        dt=0.05,
    )

    robot = MockRobot()
    solver = HJSolver(config, robot)

    grid_map = np.ones((20, 20))
    grid_map[8:12, 8:12] = 0.0

    print("Testing solve()...")
    values = solver.solve(grid_map, time=0.0)
    print("Solved successfully. Shape of values:", values.shape)

    print("Testing check_if_safe()...")
    safe, val, init_val = solver.check_if_safe(np.array([0.0, 0.0, 0.0]))
    print(f"State [0, 0, 0] safe: {safe}, value: {val:.4f}, initial value: {init_val:.4f}")

    print("Testing compute_least_restrictive_control()...")
    action, val, grad = solver.compute_least_restrictive_control(np.array([0.0, 0.0, 0.0]))
    print(f"Computed action: {action}, value: {val:.4f}, gradient: {grad}")
    print("Sanity check passed!")
