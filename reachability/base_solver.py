import hj_reachability as hj
import numpy as np

class BaseSolver:
    def __init__(self, failure_map: np.ndarray):
        self.failure_map = failure_map

        if not (self.failure_map.ndim == 2 or not self.failure_map.dtype == bool):
            raise ValueError("Failure map must be 2D numpy array of boolean values")

        self.unsafe_set = np.zeros_like(self.failure_map, dtype=bool)
        self.unsafe_set[self.failure_map] = True

        self.domain = hj.Domain(self.failure_map.shape)

    def get_dynamics(self, system, **kwargs):
        systems = ["dubins3d"]
        if system not in systems:
            print(f"'system' has to be one of {systems}")

        if system == "dubins3d":
            return hj.systems.Dubins3d(**kwargs)

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

    def solve(self, start: np.ndarray, goal: np.ndarray):
        pass
