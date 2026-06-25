from typing import Optional
from typing_extensions import Literal

from src.utils.config_utils import StrictBaseModel

class RenderConfig(StrictBaseModel):
    """Configuration for rendering the environment."""
    
    render_mode: Literal["human", "rgb_array", "none"] = "none"
    clock: float = 0.1
    world_x_size: float = 50.0
    world_y_size: float = 50.0
    collision_radius: float = 1.0
    
    show_metrics: bool = False
    show_predictions: bool = False
    export_path: Optional[str] = None
