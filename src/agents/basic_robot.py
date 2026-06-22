from dataclasses import dataclass

import torch


@dataclass
class RobotParams:
    name: str
    action_dim: int
    state_dim: int

    action_max: list[float]
    action_min: list[float]

    state_max: list[float]
    state_min: list[float]

    dt: float = 0.1

    device: str = "cpu"

    def __post_init__(self):
        """Post initialization for RobotParams. Validate dimensions."""
        if self.action_dim != len(self.action_max) or self.action_dim != len(self.action_min):
            raise ValueError("Action dimension must match the length of action_max and action_min")
        if self.state_dim != len(self.state_max) or self.state_dim != len(self.state_min):
            raise ValueError("State dimension must match the length of state_max and state_min")


class Robot:
    def __init__(self, params: RobotParams, log_enabled: bool = False) -> None:
        """Initializes a generic robot with the given parameters.

        Args:
            params: Robot configuration parameters
            log_enabled: Whether to enable logging
        """
        self.params = params
        self.state = None
        self.log_enabled = log_enabled

    def reset(self, state: torch.Tensor) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def bound_state(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def get_state(self) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def filter_action(self, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def dynamics(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Applies physical dynamics forecasts for a batch of states and actions.

        Subclasses must implement this method using PyTorch tensor operations.
        """
        raise NotImplementedError("Subclasses must implement vectorized dynamics")

    def dynamic_step(self, action: torch.Tensor) -> torch.Tensor:
        action = self.filter_action(action)

        next_state = self.dynamics(self.state, action)

        self.state = next_state.squeeze(0).detach()
        return self.state
