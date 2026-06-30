import time
from typing import Dict, List, Optional, Tuple

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
import skfmm

from src.agents import Robot
from src.controllers.base.proper_models import AbsoluteDubinsDynamics, RelativeDubinsDynamics

# ============================================================
# CONFIG
# ============================================================
from src.controllers.base.schemas import AbsoluteHJSolverConfig, RelativeHJSolverConfig
from typing import Union


# ============================================================
# SOLVER
# ============================================================
class HJSolver:
    """Hamilton-Jacobi Reachability solver for static obstacle avoidance."""

    def __init__(self, config: Union[AbsoluteHJSolverConfig, RelativeHJSolverConfig], robot: Robot) -> None:
        self.config = config
        self.robot = robot
        self.problem_definition: Optional[Dict[str, dict]] = None
        self.initial_values: Optional[np.ndarray] = None
        self.last_values: Optional[np.ndarray] = None
        self.last_values_grad: Optional[List[jnp.ndarray]] = None

    @property
    def domain(self) -> np.ndarray:
        if self.config.mode == "absolute":
            return np.array([
                [-self.config.map_width / 2.0, -self.config.map_height / 2.0, 0.0],
                [self.config.map_width / 2.0, self.config.map_height / 2.0, 2 * np.pi]
            ])
        else:
            return np.array([
                [-self.config.interaction_radius, -self.config.interaction_radius, 0.0],
                [self.config.interaction_radius, self.config.interaction_radius, 2 * np.pi]
            ])

    @property
    def domain_cells(self) -> np.ndarray:
        if self.config.mode == "absolute":
            return np.array([
                int(self.config.map_width / self.config.spatial_resolution),
                int(self.config.map_height / self.config.spatial_resolution),
                self.config.angular_cells
            ])
        else:
            width = 2.0 * self.config.interaction_radius
            return np.array([
                int(width / self.config.spatial_resolution),
                int(width / self.config.spatial_resolution),
                self.config.angular_cells
            ])

    # ------------------ CORE BUILDERS ------------------
    def get_dynamics(self):
        # The robot instance is expected to have action_min and action_max properties
        # If it doesn't, we extract from its params.
        action_min = getattr(
            self.robot, "action_min", getattr(self.robot.params, "action_min", None)
        )
        action_max = getattr(
            self.robot, "action_max", getattr(self.robot.params, "action_max", None)
        )

        if action_min is None or action_max is None:
            raise ValueError("Robot must have action_min and action_max attributes or parameters.")

        if self.config.mode == "relative":
            return RelativeDubinsDynamics(action_min, action_max)
        else:
            return AbsoluteDubinsDynamics(action_min, action_max)

    def get_solver_settings(self, accuracy: str = "low") -> hj.SolverSettings:
        if accuracy not in ["low", "medium", "high", "very_high"]:
            raise ValueError(f"Invalid accuracy level: {accuracy}")
        return hj.SolverSettings.with_accuracy(
            accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
        )

    def get_domain_grid(self) -> hj.Grid:
        return hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(self.domain[0], self.domain[1]),
            tuple(self.domain_cells),
            periodic_dims=2,
        )

    def get_problem_definition(self) -> Dict[str, dict]:
        return {
            "solver_settings": self.get_solver_settings(self.config.accuracy),
            "dynamics": self.get_dynamics(),
            "grid": self.get_domain_grid(),
        }

    # ------------------ VALUE INITIALIZATION ------------------
    def compute_initial_values(self, grid_map: Optional[np.ndarray], collision_radius: Optional[float] = None) -> np.ndarray:
        """Compute initial signed distance l(x) where 0 = obstacle.
        The shape of the output depends on the grid domain dimension.
        """
        if self.config.mode == "relative":
            if collision_radius is None:
                raise ValueError("collision_radius must be provided for relative mode")
            grid = self.get_domain_grid()
            grid_states = grid.states
            return np.sqrt(grid_states[..., 0] ** 2 + grid_states[..., 1] ** 2) - collision_radius

        if grid_map is None:
            raise ValueError("grid_map must be provided for absolute mode")

        # Signed distance transform (2D map)
        dist = skfmm.distance(grid_map - 0.5, dx=self.config.dx)  # shape: (Ny, Nx)

        # Dynamically repeat along any extra dimensions
        values = dist
        for axis_idx in range(2, len(self.config.domain_cells)):
            num_cells = self.config.domain_cells[axis_idx]
            values = np.repeat(values[..., np.newaxis], num_cells, axis=axis_idx)
        return values

    # ------------------ SOLVING ------------------
    def solve(self, grid_map: Optional[np.ndarray] = None, current_time: float = 0.0, collision_radius: Optional[float] = None) -> np.ndarray:
        if self.config.mode == "absolute" and grid_map is None:
            raise ValueError("Grid map not provided.")

        self.initial_values = self.compute_initial_values(grid_map, collision_radius)

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition()
            
        self.grid = self.problem_definition["grid"]

        target_time = self.config.target_time
        dt = self.config.dt
        epsilon = self.config.epsilon

        times = np.linspace(current_time, target_time, int(abs(target_time - current_time) / dt))
        if self.config.print_progress:
            print("Starting BRT computation...")

        values = self.initial_values
        start_t = time.time()

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
                if self.config.print_progress:
                    print("Converged early.")
                break

        self.last_values = values
        self.last_values_grad = list(jnp.gradient(self.last_values))
        if self.config.print_progress:
            print(f"Total time: {time.time() - start_t:.2f}s")
        return self.last_values

    # ------------------ SAFETY CHECKS ------------------
    def _state_to_grid(self, state: np.ndarray) -> np.ndarray:
        grid = self.problem_definition["grid"]
        ind = np.clip(grid.nearest_index(state), 0, np.array(self.config.domain_cells) - 1)
        return np.array(ind, dtype=int)

    def check_if_safe(
        self, state: np.ndarray, values: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, Optional[float]]:
        if values is None:
            values = self.last_values
        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition()

        idx = self._state_to_grid(state)
        idx_tuple = tuple(idx)
        v = float(values[idx_tuple])
        init_v = float(self.initial_values[idx_tuple]) if self.initial_values is not None else None

        return v > self.config.superlevel_set_epsilon, v, init_v

    # ------------------ PLOTTING ------------------
    def plot_zero_level(
        self,
        grid_data: np.ndarray,
        grid_map: Optional[np.ndarray] = None,
        title: str = "HJ 0-Level Set",
    ) -> None:
        from matplotlib import pyplot as plt

        x_res, y_res, _ = self.domain_cells
        x = np.linspace(self.domain[0][0], self.domain[1][0], x_res)
        y = np.linspace(self.domain[0][1], self.domain[1][1], y_res)
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

    class MockRobotConfig:
        def __init__(self) -> None:
            self.action_min = [0.0, -4.0]
            self.action_max = [8.0, 4.0]

    class MockRobot:
        def __init__(self) -> None:
            self.action_min = np.array([0.0, -4.0])
            self.action_max = np.array([8.0, 4.0])
            self.params = MockRobotConfig()

    config = AbsoluteHJSolverConfig(
        map_width=20.0,
        map_height=20.0,
        spatial_resolution=1.0,
        angular_cells=18,
        accuracy="low",
        target_time=-0.1,
        dt=0.05,
    )

    robot = MockRobot()
    solver = HJSolver(config, robot)

    grid_map = np.ones((20, 20))
    grid_map[8:12, 8:12] = 0.0

    print("Testing solve()...")
    values = solver.solve(grid_map, current_time=0.0)
    print("Solved successfully. Shape of values:", values.shape)

    print("Testing check_if_safe()...")
    safe, val, init_val = solver.check_if_safe(np.array([0.0, 0.0, 0.0]))
    print(f"State [0, 0, 0] safe: {safe}, value: {val:.4f}, initial value: {init_val:.4f}")

    print("Displaying plot for absolute mode...")
    solver.plot_zero_level(values, grid_map, title="Absolute Mode - Static Obstacle BRT")

    print("\n--- Testing Relative Mode ---")
    relative_config = RelativeHJSolverConfig(
        interaction_radius=8.0,
        spatial_resolution=0.5,
        angular_cells=18,
        accuracy="low",
        target_time=-0.1,
        dt=0.05,
    )
    relative_solver = HJSolver(relative_config, robot)

    print("Testing solve() for relative mode...")
    rel_values = relative_solver.solve(grid_map=None, current_time=0.0, collision_radius=1.0)
    print("Solved successfully. Shape of values:", rel_values.shape)

    print("Displaying plot for relative mode...")
    relative_solver.plot_zero_level(rel_values, grid_map=None, title="Relative Mode - Dynamic Avoidance BRT")

    print("Sanity check and visual tests passed!")
