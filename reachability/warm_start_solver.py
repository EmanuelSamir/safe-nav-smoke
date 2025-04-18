import numpy as np
import skfmm
import skimage
import hj_reachability as hj
import jax.numpy as jnp
from hj_reachability import dynamics, sets

from matplotlib import pyplot as plt
from dataclasses import dataclass
from time import time as time_pkg


from learning.base_model import BaseModel
from learning.gaussian_process import GaussianProcess
from itertools import product
import matplotlib.pyplot as plt
from simulator.static_smoke import StaticSmoke, SmokeBlobParams
from src.failure_map_builder import FailureMapBuilder, FailureMapParams
from src.utils import *

@dataclass
class WarmStartSolverConfig:
    system_name: str # "dubins3d"
    domain_cells: np.ndarray # 1-dim: e.g [x_resolution, y_resolution, theta_resolution]
    domain: np.ndarray # 2-dim: e.g [[x_min, y_min, theta_min], [x_max, y_max, theta_max]]
    mode: str # "brs" or "brt"
    accuracy: str # "low", "medium", "high", "very_high"
    superlevel_set_epsilon: float = 0.05
    converged_values: np.ndarray | None = None
    until_convergent: bool = True
    print_progress: bool = True


speed = 5.0
class Dubins3D(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self,
                 max_turn_rate=4.,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.speed = speed
        if control_space is None:
            control_space = sets.Box(jnp.array([-max_turn_rate]), jnp.array([max_turn_rate]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0, 0]), jnp.array([0, 0]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, psi = state
        v = self.speed
        return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), 0.])

    def control_jacobian(self, state, time):
        x, y, _ = state
        return jnp.array([
            [0],
            [0],
            [1],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
            [0., 0.],
        ])

class WarmStartSolver:
    def __init__(
        self,
        config: WarmStartSolverConfig,
    ):
        self.config = config

        self.problem_definition = None
        self.initial_values = None  # used to check if robot reached a failure (not just unsafe) state
        self.last_values = self.config.converged_values if not None else None
        self.last_grid_map = None

    def get_problem_definition(self, system, domain, dims, mode, accuracy):
        dynamics = self.get_dynamics(system)
        solver_settings = self.get_solver_settings(accuracy=accuracy, mode=mode)
        grid = self.get_domain_grid(domain, dims)
        problem_definition = {
            "solver_settings": solver_settings,
            "dynamics": dynamics,
            "grid": grid,
        }
        return problem_definition

    def get_solver_settings(self, accuracy="low", mode="brt"):
        accuracies = ["low", "medium", "high", "very_high"]
        modes = ["brs", "brt"]

        if accuracy not in accuracies:
            print(f"'accuracy' must be one of {[accuracies]}")

        if mode not in modes:
            print(f"'mode' must be one of {[modes]}")

        if mode == "brs":
            return hj.SolverSettings.with_accuracy(accuracy)
        elif mode == "brt":
            return hj.SolverSettings.with_accuracy(
                accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
            )

    def get_dynamics(self, system_name: str, **kwargs):
        systems = ["dubins3d"]
        if system_name not in systems:
            print(f"'system' has to be one of {systems}")

        # TODO: we implemented this ourselves in the hj_reachability toolbox. we are still working out a nice way to realease this.
        # unfortunately, meanwhile you will have to implement this yourself in the toolbox too.
        if system_name == "dubins3d":
            return Dubins3D()

    def get_domain_grid(self, domain, domain_cells):
        """
        :param domain: [[-min values-], [-max values-]]
        :param domain_cells: [each domain dimension number of cells]
        """
        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                np.array(np.array(domain)[0, :]),
                np.array(np.array(domain)[1, :]),
            ),
            tuple(domain_cells),
            periodic_dims=2,
        )
        return grid

    def compute_warm_start_values(self, grid_map: np.ndarray) -> np.ndarray:
        """
        combines the previous solution V_last(x) to l(x) of the new map
        :param grid_map: 2D numpy array of occupancy values
        :return: system-dim-D numpy array of warm-started values. e.g. for dubins3d, this is a 3D array
        """
        # Erode the grid map
        # grid_map = skimage.morphology.erosion(grid_map, skimage.morphology.square(3))
        warm_values = self.last_values
        l_x = self.compute_initial_values(grid_map)
        changed = np.where(self.last_grid_map != grid_map)
        # TODO: Implement loading previous values from file
        try:
            warm_values = warm_values.at[changed].set(
                l_x[changed]
            )  # jax syntax for 'warm_values[changed] = l_x[changed]'
        except AttributeError as e:
            warm_values[changed] = l_x[changed]
        return warm_values

    def compute_initial_values(self, grid_map: np.ndarray, dx: float = 0.1) -> np.ndarray:
        """
        computes l(x) where grid_map is an occupancy map assuming 0 = occupied, 1 = free
        :param grid_map: 2D numpy array of occupancy values
        :param dx: float, grid spacing

        :return: system-dim-D numpy array of initial values. e.g. for dubins3d, this is a 3D array
        """
        if self.config.system_name == "dubins3d":
            initial_values = skfmm.distance(grid_map, dx=dx)
            initial_values = np.tile(initial_values[:, :, np.newaxis], (1, 1, self.config.domain_cells[2]))
            return initial_values
        else:
            raise NotImplementedError(f"System {self.config.system_name} not implemented")

    def step(self, problem_definition, initial_values, time, target_time):
        target_values = hj.step(
            **problem_definition,
            time=time,
            values=initial_values,
            target_time=target_time,
            progress_bar=False,
        )
        return target_values

    def solve(self, grid_map, time=0.0, target_time=-10.0, dt=0.01, epsilon=0.0001):
        if grid_map is None:
            print("Cannot solve because grid map was not provided yet")
            return None, None
        print("Grid map shape:", grid_map.shape) if self.config.print_progress else None
        print("Domain cells:", self.config.domain_cells) if self.config.print_progress else None

        initial_values = self.compute_initial_values(grid_map)
        # if self.last_values is None:
        #     print("Computing value function from scratch") if self.config.print_progress else None
        #     initial_values = self.compute_initial_values(grid_map)
        # else:
        #     print("Computing warm-started value function") if self.config.print_progress else None
        #     initial_values = self.compute_warm_start_values(grid_map)

        self.initial_values = initial_values

        self.last_grid_map = grid_map

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition(
                self.config.system_name, self.config.domain, self.config.domain_cells, self.config.mode, self.config.accuracy
            )

        times = np.linspace(time, target_time, int((-target_time - time) / dt))

        print("Starting BRT computation") if self.config.print_progress else None
        time_start = time_pkg()
        for i in range(1, len(times)):
            time = times[i - 1]
            target_time = times[i]
            values = self.step(
                self.problem_definition, initial_values, time, target_time
            )

            if self.config.until_convergent:
                diff = np.amax(np.abs(values - initial_values))
                print("diff: ", diff) if self.config.print_progress else None
                if diff < epsilon:
                    print("Converged fast, lucky you!")
                    break

            initial_values = values

            if self.config.print_progress:
                print(f"Current times step: {time} s to {target_time} s")
                print(f"Max absolute difference between V_prev and V_now = {diff}")
        time_end = time_pkg()
        print(f"Time taken: {time_end - time_start} seconds")# if self.config.print_progress else None

        self.last_values = values

        return np.array(values)

    def check_if_safe(self, state, values):
        """
        checks if `state` is safe given most recent Value function
        """
        # TODO: modify for other systems
        state_ind = self._state_to_grid(state)
        value = values[state_ind[0], state_ind[1], state_ind[2]]
        try:
            initial_value = self.initial_values[
                state_ind[0], state_ind[1], state_ind[2]
            ]
        except TypeError as e:
            initial_value = None

        return value > self.config.superlevel_set_epsilon, value, initial_value

    def compute_safe_control(
        self, state, nominal_action, action_bounds, values=None, values_grad=None
    ):
        # TODO: modify for other systems
        if values is None:
            values = self.last_values

        if values_grad is None:
            values_grad = np.gradient(values)

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition(
                self.config.system_name, self.config.domain, values.shape, self.config.mode, self.config.accuracy
            )

        state = np.array(state)
        is_safe, value, initial_value = self.check_if_safe(state, values)

        action = nominal_action

        if not is_safe:
            #print("\033[31m{}{}\033[0m".format("Safe controller intervening. Value is:", value))

            state_ind = self._state_to_grid(state)

            grad_x = values_grad[0][state_ind[0], state_ind[1], state_ind[2]]
            grad_y = values_grad[1][state_ind[0], state_ind[1], state_ind[2]]
            grad_theta = values_grad[2][state_ind[0], state_ind[1], state_ind[2]]

            # TODO: Replace temporal straight forward solution
            N_samples = 1000
            w_samples = np.linspace(action_bounds[1][0], action_bounds[1][1], N_samples)

            opt_problem = lambda w: 0.5 * np.cos(state[2]) * grad_x + 0.5 * np.sin(state[2]) * grad_y + grad_theta * w

            w_opt_ind = np.argmax(opt_problem(w_samples))
            w_opt = w_samples[w_opt_ind]
            safe_w = w_opt

            action = np.array([nominal_action[0], safe_w])
        else:
            #print("\033[32m{}\033[0m".format("Safe controller not intervening"))
            pass
        return action, value, initial_value

    def _state_to_grid(self, state):
        grid = self.problem_definition["grid"]
        state_ind = np.array(grid.nearest_index(state)) - 1
        return state_ind

    def plot_zero_level(
        self,
        grid_data,
        grid_map=None,
        fig=None,
        ax=None,
        title="Contour Map with 0-Level Set",
    ):
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if grid_map is not None:
            x = np.linspace(self.config.domain[0][0], self.config.domain[1][0], self.config.domain_cells[0])
            y = np.linspace(self.config.domain[0][1], self.config.domain[1][1], self.config.domain_cells[1])

            # Create the meshgrid
            x, y = np.meshgrid(x, y)

            # Plot the data
            plt.pcolormesh(x, y, grid_map, shading="auto", cmap="gray")

        x = np.linspace(self.config.domain[0][0], self.config.domain[1][0], self.config.domain_cells[0])
        y = np.linspace(self.config.domain[0][1], self.config.domain[1][1], self.config.domain_cells[1])

        x, y = np.meshgrid(x, y)
        z = grid_data

        # plot all contours
        contour_all = plt.contour(x, y, z, levels=20, cmap='viridis')
        plt.clabel(contour_all, inline=True, fontsize=8)

        # plot the 0-level set in red
        contour = plt.contour(x, y, z, levels=[0], colors="red")
        plt.clabel(contour, fmt="%2.1f", colors="black")

        plt.title(title)
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        # plt.gca().set_aspect('equal')

        plt.close(fig)




if __name__ == "__main__":
    x_size, y_size = 80, 50

    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=2.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=10, y_pos=20, intensity=1.5, spread_rate=5.0),
        SmokeBlobParams(x_pos=60, y_pos=20, intensity=2.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=50, y_pos=10, intensity=1.5, spread_rate=5.0),
    ]

    smoke_simulator = StaticSmoke(x_size=x_size, y_size=y_size, resolution=0.1, smoke_blob_params=smoke_blob_params)
    
    params = FailureMapParams(x_size=x_size, y_size=y_size, resolution=0.5, map_rule_type='threshold', map_rule_threshold=0.7)
    builder = FailureMapBuilder(params)

    gp = GaussianProcess()

    sample_size = 50

    for i in range(sample_size):
        X_sample = np.concatenate([np.random.uniform(0, y_size, 1), np.random.uniform(0, x_size, 1)])
        y_observe = smoke_simulator.get_smoke_density(X_sample[1], X_sample[0])
        gp.track_data(X_sample, y_observe)
    gp.update()

    first_failure_map = builder.build_map(gp)
    builder.plot_failure_map()
    plt.show()

    sample_size = 200

    for i in range(sample_size):
        X_sample = np.concatenate([np.random.uniform(0, y_size, 1), np.random.uniform(0, x_size, 1)])
        y_observe = smoke_simulator.get_smoke_density(X_sample[1], X_sample[0])
        gp.track_data(X_sample, y_observe)
    gp.update()

    second_failure_map = builder.build_map(gp)
    builder.plot_failure_map()
    plt.show()

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

    first_values = solver.solve(first_failure_map.T, target_time=-10.0, dt=0.1, epsilon=0.0001)

    second_values = solver.solve(second_failure_map.T, target_time=-10.0, dt=0.1, epsilon=0.0001)

    # nominal_action = [0.5, 0.4]
    # safe_action = solver.compute_safe_control([-8, -6, 0.3], nominal_action)
    # print(safe_action)
