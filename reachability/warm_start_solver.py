import numpy as np
import skfmm
import skimage
import hj_reachability as hj
import jax.numpy as jnp
from hj_reachability import dynamics, sets

from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from time import time as time_pkg
from skimage.transform import resize

from learning.base_model import BaseModel
from learning.gaussian_process import GaussianProcess
from itertools import product
import matplotlib.pyplot as plt
from simulator.static_smoke import StaticSmoke, SmokeBlobParams
from src.failure_map_builder import FailureMapBuilder, FailureMapParams
from simulator.dynamic_smoke import DynamicSmoke, DynamicSmokeParams, DownwardsSensorParams, SmokeBlobParams
from src.utils import *
import numpy as np
import skfmm
import skimage
import jax.numpy as jnp
import hj_reachability as hj
from hj_reachability import dynamics, sets
from dataclasses import dataclass
from time import time as time_pkg
from skimage.transform import resize
from matplotlib import pyplot as plt
from scipy.ndimage import sobel

from src.utils import get_index_bounds


# ============================================================
# CONFIG
# ============================================================
@dataclass
class WarmStartSolverConfig:
    system_name: str              # "dubins2d"
    domain_cells: np.ndarray      # e.g. [x_res, y_res, theta_res]
    domain: np.ndarray            # e.g. [[x_min, y_min, θ_min], [x_max, y_max, θ_max]]
    mode: str                     # "brs" or "brt"
    accuracy: str                 # "low", "medium", "high", "very_high"
    superlevel_set_epsilon: float = 0.0
    converged_values: np.ndarray | None = field(default_factory=lambda: None)
    until_convergent: bool = True
    print_progress: bool = True
    warm_start: bool = False
    action_bounds: np.ndarray = field(default_factory=lambda: (np.array([0.0, -4.0]), np.array([8.0, 4.0])))
    disturbance_bounds: np.ndarray = field(default_factory=lambda: (np.array([0.0, 0.0]), np.array([0.0, 0.0])))

# ============================================================
# DYNAMICS: Dubins3D
# ============================================================
class Dubins2D(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, action_bounds, disturbance_bounds, control_mode="max", disturbance_mode="min"):
        assert len(action_bounds) == 2 and len(disturbance_bounds) == 2, "Action and disturbance bounds must be of length 2"
        assert action_bounds[0].shape == action_bounds[1].shape == (2,), "Action bounds must be a 2D array"
        assert disturbance_bounds[0].shape == disturbance_bounds[1].shape == (2,), "Disturbance bounds must be a 2D array"
        control_space = sets.Box(jnp.array(action_bounds[0]), jnp.array(action_bounds[1]))
        disturbance_space = sets.Box(jnp.array(disturbance_bounds[0]), jnp.array(disturbance_bounds[1]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.zeros_like(state)

    def control_jacobian(self, state, time):
        _, _, psi = state
        return jnp.array([
            [jnp.cos(psi), 0.0],
            [jnp.sin(psi), 0.0],
            [0.0, 1.0],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])


class Dubins2D_fixed_velocity(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, action_bounds, disturbance_bounds, control_mode="max", disturbance_mode="min"):
        assert len(action_bounds) == 2 and len(disturbance_bounds) == 2, "Action and disturbance bounds must be of length 2"
        assert action_bounds[0].shape == (2,), "Action bounds must be a 2D array"
        assert disturbance_bounds[0].shape == (2,), "Disturbance bounds must be a 2D array"
        control_space = sets.Box(jnp.array([action_bounds[0][1]]), jnp.array([action_bounds[1][1]]))
        disturbance_space = sets.Box(jnp.array([disturbance_bounds[0][1]]), jnp.array([disturbance_bounds[1][1]]))
        self.v = action_bounds[1][0]
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, psi = state
        return jnp.array([self.v * jnp.cos(psi), self.v * jnp.sin(psi), 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[0], [0], [1]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.], [0.], [1.]])

class Unicycle2D(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, action_bounds, disturbance_bounds, control_mode="max", disturbance_mode="min"):
        assert len(action_bounds) == 2 and len(disturbance_bounds) == 2, "Action and disturbance bounds must be of length 2"
        assert action_bounds[0].shape == action_bounds[1].shape == (2,), "Action bounds must be a 2D array"
        assert disturbance_bounds[0].shape == disturbance_bounds[1].shape == (2,), "Disturbance bounds must be a 2D array"
        control_space = sets.Box(jnp.array(action_bounds[0]), jnp.array(action_bounds[1]))
        disturbance_space = sets.Box(jnp.array(disturbance_bounds[0]), jnp.array(disturbance_bounds[1]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, psi, vel = state
        return jnp.array([
            vel * jnp.cos(psi),
            vel * jnp.sin(psi),
            0.0,
            0.0,
        ])

    def control_jacobian(self, state, time):
        _, _, psi, vel = state
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])

    def disturbance_jacobian(self, state, time):
        x, y, psi, v = state
        return jnp.array([
            [jnp.cos(psi), 0.0],
            [jnp.sin(psi), 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])

# ============================================================
# SOLVER
# ============================================================
class WarmStartSolver:
    def __init__(self, config: WarmStartSolverConfig):
        self.config = config
        self.problem_definition = None
        self.initial_values = None
        self.last_values = config.converged_values
        self.last_grid_map = None
        self.changed_grid_map = None
        self.processed_updates = []

    # ------------------ CORE BUILDERS ------------------
    def get_dynamics(self, system_name: str):
        if system_name == "dubins2d":
            return Dubins2D(action_bounds=self.config.action_bounds, disturbance_bounds=self.config.disturbance_bounds)
        elif system_name == "dubins2d_fixed_velocity":
            return Dubins2D_fixed_velocity(action_bounds=self.config.action_bounds, disturbance_bounds=self.config.disturbance_bounds)
        elif system_name == "unicycle2d":
            return Unicycle2D(action_bounds=self.config.action_bounds, disturbance_bounds=self.config.disturbance_bounds)
        else:
            raise ValueError(f"Unsupported system '{system_name}'. Expected 'dubins2d' or 'dubins2d_fixed_velocity' or 'unicycle2d'.")

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
            "dynamics": self.get_dynamics(self.config.system_name),
            "grid": self.get_domain_grid(self.config.domain, self.config.domain_cells),
        }

    # ------------------ VALUE INITIALIZATION ------------------
    def compute_initial_values(self, grid_map: np.ndarray, dx: float = 0.1) -> np.ndarray:
        """
        Compute initial signed distance l(x) where 0 = obstacle.
        The shape of the output depends on the system dimension.
        """
        system = self.config.system_name

        if system not in ["dubins2d", "dubins2d_fixed_velocity", "unicycle2d"]:
            raise NotImplementedError(f"System '{system}' not implemented.")

        # 1️⃣ Signed distance transform (2D map)
        # grid_map assumed: 1 = free, 0 = obstacle
        dist = skfmm.distance(grid_map - 0.5, dx=dx)  # shape: (Ny, Nx)

        # 2️⃣ Extend along angular dimension (θ)
        num_theta = self.config.domain_cells[2]
        dist_3d = np.repeat(dist[:, :, np.newaxis], num_theta, axis=2)  # (Ny, Nx, Nθ)

        # 3️⃣ Extend along velocity dimension (v) — only for unicycle2d
        if system == "unicycle2d":
            num_v = self.config.domain_cells[3]
            dist_4d = np.repeat(dist_3d[:, :, :, np.newaxis], num_v, axis=3)  # (Ny, Nx, Nθ, Nv)
            return dist_4d

        # 4️⃣ Otherwise return 3D for Dubins
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
        idx = self._state_to_grid(state)
        idx = tuple(idx)
        v = values[idx]
        init_v = self.initial_values[idx] if self.initial_values is not None else None
        return v > self.config.superlevel_set_epsilon, v, init_v

    # ------------------ CONTROL COMPUTATION ------------------
    def compute_least_restrictive_control(
        self, state, values=None, values_grad=None
    ):
        """
        Compute the least restrictive control action based on the values and values gradients.
        If values and values_grad are not provided, use the last values and values gradients.
        If the problem definition is not set, set it.
        If last values are not set, return None.
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
            # values_grad = [sobel(values, axis=i, mode='nearest') for i in range(values.ndim)]

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition(
                self.system, self.domain, values.shape, self.mode, self.accuracy
            )

        state = np.array(state)
        state_ind = self._state_to_grid(state)
        idx = tuple(state_ind)

        value = values[idx]
        grad_x = values_grad[0][idx]
        grad_y = values_grad[1][idx]
        grad_theta = values_grad[2][idx]

        value, value_grad = np.array(value), np.array([grad_x, grad_y, grad_theta])

        if self.config.system_name == "unicycle2d":
            gx_body = np.cos(state[2]) * grad_x + np.sin(state[2]) * grad_y
            grad_v = values_grad[3][idx]

            if gx_body > 0 and grad_v > 0:
                safe_a = self.config.action_bounds[0][0]  # brake
            elif gx_body < 0 and grad_v < 0:
                safe_a = self.config.action_bounds[0][1]  # accelerate
            else:
                safe_a = 0.0  # neutral

            # --- Safe angular velocity (ω) ---
            if grad_theta > 0:
                safe_w = self.config.action_bounds[1][0]
            else:
                safe_w = self.config.action_bounds[1][1]

            action = np.array([safe_a, safe_w])
        
        elif self.config.system_name == "dubins2d":
            if np.cos(state[2]) * grad_x + np.sin(state[2]) * grad_y > 0:
                safe_v = self.config.action_bounds[0][1]
            else:
                safe_v = self.config.action_bounds[0][0]

            if np.sign(grad_theta) > 0:
                safe_w = self.config.action_bounds[1][1]
            else:
                safe_w = self.config.action_bounds[1][0]

            action = np.array([safe_v, safe_w])

        elif self.config.system_name == "dubins2d_fixed_velocity":
            gx_body = np.cos(state[2]) * grad_x + np.sin(state[2]) * grad_y
            if np.sign(grad_theta) > 0 or (abs(grad_theta) < 1e-3 and gx_body > 0):
                min_w = self.config.action_bounds[0][1]
                safe_w = min_w
            else:
                max_w = self.config.action_bounds[1][1]
                safe_w = max_w
            
            action = np.array([self.config.action_bounds[1][0], safe_w])
        else:
            raise ValueError(f"Unsupported system '{self.config.system_name}'. Expected 'dubins2d' or 'dubins2d_fixed_velocity' or 'unicycle2d'.")

        dotV = 5.0 * (grad_x*np.cos(state[2]) + grad_y*np.sin(state[2])) + grad_theta * safe_w

        return action, value, value_grad
        
    # ------------------ PLOTTING ------------------
    def plot_zero_level(self, grid_data, grid_map=None, title="HJ 0-Level Set"):
        x_res, y_res, _ = self.config.domain_cells
        x = np.linspace(self.config.domain[0][0], self.config.domain[1][0], x_res)
        y = np.linspace(self.config.domain[0][1], self.config.domain[1][1], y_res)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        if grid_map is not None:
            ax.imshow(grid_map, cmap="gray", origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
        cs = ax.contour(X, Y, grid_data[:, :, 0].T, levels=[0], colors="red")
        ax.clabel(cs, fmt="%2.1f", colors="black", fontsize=8)
        ax.set_title(title)
        plt.show()



if __name__ == "__main__":
    x_size, y_size = 40, 30

    smoke_blob_params = [
        SmokeBlobParams(x_pos=5, y_pos=20, intensity=1.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=15, y_pos=5, intensity=1.0, spread_rate=5.0),
        SmokeBlobParams(x_pos=35, y_pos=5, intensity=1.0, spread_rate=3.0),
    ]

    # Simple test First no estimation needed. GT is used directly and new sources appear instantly or step is called

    sensor_params = DownwardsSensorParams(world_x_size=x_size, world_y_size=y_size)
    smoke_params = DynamicSmokeParams(x_size=x_size, y_size=y_size, smoke_blob_params=smoke_blob_params, resolution=0.4, fov_sensor_params=sensor_params)
    smoke_simulator = DynamicSmoke(params=smoke_params)
    smoke_simulator.step(dt=0.1)

    hj_resolution = 1.0
    cell_y_size, cell_x_size = get_index_bounds(x_size, y_size, hj_resolution)
    domain_cells = np.array([cell_x_size, cell_y_size, 40, 20])
    domain = [[0, 0, 0, 0], [x_size, y_size, 2*np.pi, 10.0]]

    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="unicycle2d",
            domain_cells=domain_cells,
            domain=domain,
            mode="brt",
            accuracy="low",
            converged_values=None,
            until_convergent=True,
            print_progress=True,
        )
    )
    solver.problem_definition = solver.get_problem_definition()
    smoke_map = resize(smoke_simulator.get_smoke_map(), (cell_y_size, cell_x_size), anti_aliasing=True)
    failure_mask = (smoke_map < 0.6).astype(float)

    x_space = np.linspace(0, x_size, cell_x_size)
    y_space = np.linspace(0, y_size, cell_y_size)
    f, ax = plt.subplots(2, 2, figsize=(10, 5))
    levels_ = [solver.config.superlevel_set_epsilon]

    values = solver.compute_initial_values(failure_mask.T)

    z = values[:,:,0, -1].T
    contour = ax[0, 0].contour(x_space, y_space, z, levels=levels_, cmap='Spectral')
    ax[0, 0].imshow(failure_mask, cmap='gray', origin='lower', extent=[0, x_size, 0, y_size])
    ax[0, 0].clabel(contour, fmt="%2.1f", colors="black", fontsize=5)

    print("Solving...")
    values = solver.solve(failure_mask.T, target_time=-2.0, dt=0.1)
    z = values[:,:,0, -1].T
    contour = ax[0, 1].contour(x_space, y_space, z, levels=levels_, cmap='Spectral')
    ax[0, 1].imshow(failure_mask, cmap='gray', origin='lower', extent=[0, x_size, 0, y_size])
    ax[0, 1].clabel(contour, fmt="%2.1f", colors="black", fontsize=5)
    plt.show()

    exit()

    for i in range(2):
        smoke_simulator.step(dt=0.1)

    smoke_map = resize(smoke_simulator.get_smoke_map(), (cell_y_size, cell_x_size), anti_aliasing=True)
    failure_mask_after_update = (smoke_map < 0.6).astype(float)
    values = solver.solve(failure_mask_after_update.T, target_time=-2.0, dt=0.1)
    z = solver.initial_values[:,:,0].T
    contour = ax[1, 0].contour(x_space, y_space, z, levels=levels_, cmap='Spectral')
    im = ax[1, 0].imshow(z, cmap='turbo', origin='lower', extent=[0, x_size, 0, y_size])
    ax[1, 0].clabel(contour, fmt="%2.1f", colors="black", fontsize=5)


    contour = ax[1, 1].contour(x_space, y_space, solver.processed_updates[0][:,:,0].T, levels=levels_, colors='orange')
    ax[1, 1].contour(x_space, y_space, solver.changed_grid_map[:,:,0].T, levels=levels_, colors='blue')


    f.colorbar(im, ax=ax[1, 0])
    plt.show()


    # Second: Use learning approach instead of GT. Change solver to look at values to be predicted and update values based on new data
    exit()

    params = FailureMapParams(x_size=x_size, y_size=y_size, resolution=0.5, map_rule_type='threshold', map_rule_threshold=0.6)
    builder = FailureMapBuilder(params)

    gp = GaussianProcess()

    sample_size = 200

    for i in range(sample_size):
        X_sample = np.concatenate([np.random.uniform(0, x_size, 1), np.random.uniform(0, y_size, 1), np.array([0.0])])
        y_observe = smoke_simulator.get_smoke_density(np.array([X_sample[0], X_sample[1]]))[0]
        gp.track_data(X_sample, y_observe)
    gp.update()

    first_failure_map = builder.build_map(gp, 0.1)
    builder.plot_failure_map()
    # plt.show()

    # sample_size = 200

    # for i in range(sample_size):
    #     X_sample = np.concatenate([np.random.uniform(0, x_size, 1), np.random.uniform(0, y_size, 1)])
    #     y_observe = smoke_simulator.get_smoke_density(np.array([X_sample[0], X_sample[1]]))[0]
    #     gp.track_data(X_sample, y_observe)
    # gp.update()

    # second_failure_map = builder.build_map(gp)
    # builder.plot_failure_map()
    # plt.show()

    cell_y_size, cell_x_size = get_index_bounds(x_size, y_size, builder.params.resolution)
    domain_cells = np.array([cell_x_size, cell_y_size, 40])
    domain = [[0, 0, 0], [x_size, y_size, 2*np.pi]]

    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins3d",
            domain_cells=domain_cells,
            domain=domain,
            mode="brt",
            accuracy="low",
            converged_values=None,
            until_convergent=True,
            print_progress=True,
        )
    )

    first_values = solver.solve(first_failure_map.T, target_time=-2.0, dt=0.1, epsilon=0.01)

    state_ind = solver._state_to_grid(np.array([10, 20, 0.]))
    z = first_values[:,:,state_ind[2]].T

    x_space = np.linspace(0, x_size, cell_x_size)
    y_space = np.linspace(0, y_size, cell_y_size)

    contour = plt.contour(x_space, y_space, z, levels=20, cmap='Spectral')
    plt.show()

    # second_values = solver.solve(second_failure_map.T, target_time=-10.0, dt=0.1, epsilon=0.0001)

    # nominal_action = [0.5, 0.4]
    # safe_action = solver.compute_safe_control([-8, -6, 0.3], nominal_action)
    # print(safe_action)
