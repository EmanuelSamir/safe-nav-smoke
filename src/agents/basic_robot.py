import torch

from src.agents.schemas import RobotParams


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
