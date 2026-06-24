from typing import List, Literal

from pydantic import model_validator

from src.utils.config_utils import StrictBaseModel


class RobotParams(StrictBaseModel):
    name: Literal["dubins2d"] = "dubins2d"
    action_min: List[float] = [-1.0, -1.0]
    action_max: List[float] = [1.0, 1.0]
    action_dim: int = 2
    state_dim: int = 3
    state_min: List[float] = [0.0, 0.0, 0.0]
    state_max: List[float] = [50.0, 50.0, 6.28]
    dt: float = 0.1
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.action_dim != len(self.action_max) or self.action_dim != len(self.action_min):
            raise ValueError("Action dimension must match the length of action_max and action_min")
        if self.state_dim != len(self.state_max) or self.state_dim != len(self.state_min):
            raise ValueError("State dimension must match the length of state_max and state_min")
        return self
