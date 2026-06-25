from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.base_renderer import BaseRenderer
from src.visualization.schemas import RenderConfig
from src.visualization.plot_utils import (
    plot_smoke_background,
    plot_goals,
    plot_controller_rollouts,
    plot_robots_and_sensors,
)

class SimpleRenderer(BaseRenderer):
    """Simple single-plot renderer for Gymnasium environment visualization."""

    def __init__(self, cfg: Optional[Any] = None):
        if isinstance(cfg, dict):
            self.cfg = RenderConfig(**cfg)
        elif hasattr(cfg, "render_mode"):
            # Assume it's already a RenderConfig or similar object
            self.cfg = cfg
        else:
            self.cfg = RenderConfig()

        self.render_mode = self.cfg.render_mode
        self.clock = self.cfg.clock
        self.world_x_size = self.cfg.world_x_size
        self.world_y_size = self.cfg.world_y_size

        self.window = {"fig": None, "ax": None, "cax": None}
        self.frames = []

        # Headless Agg backend for rgb_array mode to prevent plotting GUI popup
        if self.render_mode == "rgb_array":
            plt.switch_backend("Agg")

    def _init_render_window(
        self, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None, env: Any = None
    ):
        """Initializes the plotting window/axes and draws static components."""
        if fig is not None and ax is not None:
            self.window["fig"] = fig
            self.window["ax"] = ax
        else:
            self.window["fig"], self.window["ax"] = plt.subplots(figsize=(8, 6))

        # Initial plot of the smoke background
        self.window["cax"] = plot_smoke_background(
            self.window["ax"], env.smoke_simulator
        )

        self.window["ax"].set_title("Simulation")
        self.window["ax"].set_xlim(0, self.world_x_size)
        self.window["ax"].set_ylim(0, self.world_y_size)
        self.window["ax"].set_xticks([])
        self.window["ax"].set_yticks([])

        # Draw static goal circles and texts
        goal_circles, goal_texts = plot_goals(self.window["ax"], env.smoke_agents)
        self.window["goal_circles"] = goal_circles
        self.window["goal_texts"] = goal_texts

    def render(self, info: Dict[str, Any]) -> Any:
        """Render a single frame based on the state/environment info."""
        if not self.render_mode or self.render_mode not in ["human", "rgb_array"]:
            return None

        env = info.get("env")
        controller = info.get("controller")

        if self.window["fig"] is None:
            self._init_render_window(env=env)

        # Update smoke background values
        plot_smoke_background(self.window["ax"], env.smoke_simulator)

        # Draw planning rollouts
        plot_controller_rollouts(self.window["ax"], controller)

        # Draw robot states and sensor field of view
        plot_robots_and_sensors(
            ax=self.window["ax"],
            env_cfg=env.env_cfg,
            agents=env.smoke_agents,
            sensor_cfg=getattr(env, "sensor_cfg", None),
            sensor=getattr(env, "sensor", None),
            smoke_simulator=env.smoke_simulator,
            goal_circles=self.window.get("goal_circles", []),
            controller=controller,
            get_smoke_density_sensor_fn=getattr(env, "get_smoke_density_sensor", None),
        )

        # Render display output
        self.window["fig"].canvas.draw()

        if self.render_mode == "human":
            self.window["fig"].canvas.flush_events()
            plt.pause(self.clock)
        elif self.render_mode == "rgb_array":
            width, height = self.window["fig"].canvas.get_width_height()
            try:
                rgba = np.asarray(self.window["fig"].canvas.buffer_rgba())
                img = rgba[..., :3].copy()
            except AttributeError:
                img = np.frombuffer(self.window["fig"].canvas.tostring_rgb(), dtype="uint8")
                img = img.reshape(height, width, 3).copy()
            return img
        return None

    def save_frame(self) -> None:
        """Save the current frame to buffer."""
        pass

    def close(self) -> None:
        """Clean up and close rendering resources."""
        if self.window["fig"] is not None:
            plt.close(self.window["fig"])
        self.window = {"fig": None, "ax": None, "cax": None}
