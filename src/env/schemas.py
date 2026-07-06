from typing import List, Optional

from typing_extensions import Literal

from src.utils.config_utils import StrictBaseModel


# Env schema
class EnvConfig(StrictBaseModel):
    num_agents: int = 6
    world_x_size: float = 30.0
    world_y_size: float = 30.0
    clock: float = 0.1
    max_steps: int = 200
    collision_radius: float = 0.8
    goal_radius: float = 0.5
    initial_locations: Optional[List[List[float]]] = None
    goal_locations: Optional[List[List[float]]] = None
    render: Literal["none", "rgb_array", "human"] = "none"
    render_save_every: int = 2
    terminate_on_collision: bool = False
    collision_penalty: float = -10.0
    smoke_density_threshold: Optional[float] = None
    save_transitions: bool = False
    save_global_map_transitions: bool = True
    remove_dead_agents: bool = True

    # Extra parameters in smoke_env.yaml configuration
    test: Optional[bool] = False
    num_episodes: Optional[int] = 1
    save_transitions_path: Optional[str] = None
