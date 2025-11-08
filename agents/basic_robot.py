import numpy as np
import matplotlib.pyplot as plt
import logging
from src.utils import *
from dataclasses import dataclass, field
import yaml

@dataclass
class RobotParams:
    action_dim: int = 2
    state_dim: int = 3

    action_max: list = field(default_factory=lambda: [3.0, 4.0])
    action_min: list = field(default_factory=lambda: [0.0, -4.0])
    state_max: list = field(default_factory=lambda: [80, 30, 2 * np.pi])
    state_min: list = field(default_factory=lambda: [0, 0, 0])

    robot_type: str = field(default="dubins2d") # "unicycle", "dubins2d", "dubins2d_fixed_speed"

    dt: float = 0.1

    def __post_init__(self):
        assert self.action_dim == len(self.action_max) == len(self.action_min), "Action dimension must match the length of action_max and action_min"
        assert self.state_dim == len(self.state_max) == len(self.state_min), "State dimension must match the length of state_max and state_min"
        self.action_min = np.array(self.action_min, dtype=np.float32)
        self.action_max = np.array(self.action_max, dtype=np.float32)
        self.state_min = np.array(self.state_min, dtype=np.float32)
        self.state_max = np.array(self.state_max, dtype=np.float32)

    @staticmethod
    def load_from_yaml(file_path: str) -> "RobotParams":
        """Load robot parameters from a YAML config file."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return RobotParams(**data)


class Robot:
    def __init__(self, robot_params: RobotParams, log_enabled: bool = False) -> None:
        self.robot_params = robot_params
        self.state = None
        self.log_enabled = log_enabled

    def reset(self, state: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def bound_state(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def get_state(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def open_loop_dynamics(self, state):
        raise NotImplementedError("Subclasses must implement this method")

    def control_jacobian(self, state):
        raise NotImplementedError("Subclasses must implement this method")

    def dynamic_step(self, action: np.ndarray) -> None:
        action = self.filter_action(action)

        def one_step_dynamics(state, action):
            next_state = self.open_loop_dynamics(state) + self.control_jacobian(state) @ action
            return next_state

        # k1
        k1 = one_step_dynamics(self.state, action)
        # k2
        mid_state_k2 = self.state + 0.5 * self.dt * k1
        k2 = one_step_dynamics(mid_state_k2, action)
        # k3
        mid_state_k3 = self.state + 0.5 * self.dt * k2
        k3 = one_step_dynamics(mid_state_k3, action)
        # k4
        end_state_k4 = self.state + self.dt * k3
        k4 = one_step_dynamics(end_state_k4, action)

        # Combine k1, k2, k3, k4 to compute the next state
        self.state = self.state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state = self.bound_state(self.state)

        return self.state
