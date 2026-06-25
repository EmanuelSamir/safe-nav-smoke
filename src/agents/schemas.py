from typing import List, Literal

from pydantic import model_validator

from src.utils.config_utils import StrictBaseModel


class RobotConfig(StrictBaseModel):
    name: Literal["dubins2d", "base"]  # Add robot types here
    action_min: List[float]
    action_max: List[float]
    action_dim: int
    state_dim: int
    state_min: List[float]
    state_max: List[float]
    dt: float
    device: Literal["cpu", "cuda", "mps"]

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.action_dim != len(self.action_max) or self.action_dim != len(self.action_min):
            raise ValueError("Action dimension must match the length of action_max and action_min")
        if self.state_dim != len(self.state_max) or self.state_dim != len(self.state_min):
            raise ValueError("State dimension must match the length of state_max and state_min")
        return self


class DubinsConfig(RobotConfig):
    name: Literal["dubins2d"] = "dubins2d"
    action_min: List[float] = [0.0, -4.0]
    action_max: List[float] = [6.0, 4.0]
    action_dim: int = 2
    state_dim: int = 3
    state_min: List[float] = [0.0, 0.0, 0.0]
    state_max: List[float] = [35.0, 35.0, 6.28]
    dt: float = 0.1
    device: Literal["cpu", "cuda", "mps"] = "cpu"
