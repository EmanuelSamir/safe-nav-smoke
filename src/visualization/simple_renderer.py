import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow, Circle, FancyArrow, Polygon, Wedge
from typing import Dict, Any, Union, Optional
from enum import Enum
from utils import clip_world
from visualization.base_renderer import BaseRenderer


class SimpleRenderer(BaseRenderer):
    """
    Simple single-plot renderer for Gymnasium environment visualization.
    Extracts figure/window management and plotting commands from SmokeEnv.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        
        # Get render mode (resolving Enum if necessary)
        render_val = getattr(cfg, "render", "none")
        if isinstance(render_val, Enum):
            self.render_mode = render_val.value
        else:
            self.render_mode = str(render_val)
            
        self.clock = float(getattr(cfg, "clock", 0.1))
        self.world_x_size = float(getattr(cfg, "world_x_size", 100.0))
        self.world_y_size = float(getattr(cfg, "world_y_size", 100.0))
        self.collision_radius = float(getattr(cfg, "collision_radius", 0.8))

        self.window = {"fig": None, "ax": None, "cax": None}
        self.frames = []

        # Headless Agg backend for rgb_array mode to prevent plotting GUI popup
        if self.render_mode == "rgb_array":
            plt.switch_backend("Agg")

    def _init_render_window(self, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None, env: Any = None):
        """Initializes the plotting window/axes and draws static components."""
        if fig is not None and ax is not None:
            self.window["fig"] = fig
            self.window["ax"] = ax
        else:
            self.window["fig"], self.window["ax"] = plt.subplots(figsize=(8, 6))

        # Initial plot of the smoke background
        self.window["cax"] = self.plot_smoke_background(
            self.window["ax"], env.env_params, env.smoke_simulator
        )

        self.window["ax"].set_title("Simulation")
        self.window["ax"].set_xlim(0, self.world_x_size)
        self.window["ax"].set_ylim(0, self.world_y_size)
        self.window["ax"].set_xticks([])
        self.window["ax"].set_yticks([])

        # Draw static goal circles and texts
        goal_circles, goal_texts = self.plot_goals(
            self.window["ax"], env.env_params, env.agents
        )
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
        self.plot_smoke_background(
            self.window["ax"], env.env_params, env.smoke_simulator
        )

        # Draw planning rollouts
        self.plot_controller_rollouts(self.window["ax"], controller)

        # Draw robot states and sensor field of view
        self.plot_robots_and_sensors(
            self.window["ax"],
            env.env_params,
            env.agents,
            env.env_params.sensor_params,
            env.sensor,
            goal_circles=self.window.get("goal_circles", []),
            controller=controller,
            get_smoke_density_sensor_fn=env.get_smoke_density_sensor,
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
                img = rgba[..., :3]
            except AttributeError:
                img = np.frombuffer(self.window["fig"].canvas.tostring_rgb(), dtype="uint8")
                img = img.reshape(height, width, 3)
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

    # --- Shared Drawing Helpers ---

    @staticmethod
    def plot_smoke_background(ax: plt.Axes, env_params: Any, smoke_simulator: Any) -> Any:
        smoke_map = smoke_simulator.get_smoke_map()
        extent = smoke_simulator.get_smoke_extent()
        if ax.images:
            ax.images[0].set_array(smoke_map)
            return ax.images[0]
        else:
            cax = ax.imshow(
                smoke_map,
                cmap="gray",
                extent=extent,
                origin="lower",
                zorder=1,
            )
            cax.set_clim(vmin=0.0, vmax=1.0)
            return cax

    @staticmethod
    def plot_goals(ax: plt.Axes, env_params: Any, agents: list) -> tuple:
        goal_circles = []
        goal_texts = []
        for i, agent in enumerate(agents):
            if agent.goal_location is not None:
                circle = Circle(
                    (agent.goal_location[0], agent.goal_location[1]),
                    radius=agent.goal_radius,
                    color="g",
                    fill=True,
                    alpha=0.6,
                    zorder=5,
                )
                ax.add_patch(circle)
                goal_circles.append(circle)

                x0, y0 = circle.center
                r = circle.radius
                text = ax.text(
                    x0,
                    y0 - r - 1.5,
                    f"goal_{i}" if len(agents) > 1 else "goal",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="green",
                    zorder=10,
                )
                goal_texts.append(text)
        return goal_circles, goal_texts

    @staticmethod
    def plot_controller_rollouts(ax: plt.Axes, controller: Any) -> None:
        for line in list(ax.lines):
            line.remove()

        if controller is not None and hasattr(controller, "visualize_rollouts"):
            controller.visualize_rollouts(ax)
            for line in ax.lines:
                line.set_zorder(3)

    @staticmethod
    def plot_robots_and_sensors(
        ax: plt.Axes,
        env_params: Any,
        agents: list,
        sensor_params: Any,
        sensor: Any,
        goal_circles: list = None,
        controller: Any = None,
        get_smoke_density_sensor_fn: Any = None,
    ) -> None:
        if goal_circles is None:
            goal_circles = []

        for patch in list(ax.patches):
            if isinstance(patch, (FancyArrow, Arrow, Polygon, Wedge, Circle)):
                if patch not in goal_circles:
                    patch.remove()

        for i, agent in enumerate(agents):
            odom = agent.get_robot_odom()
            pos_x, pos_y = odom["location"]
            angle_rad = odom["angle"]
            angle_deg = np.rad2deg(angle_rad)

            # Draw sensor projection bound
            if sensor_params.sensor_type == "downwards":
                square = sensor.projection_bounds(pos_x, pos_y)
                bounded_square = np.array(
                    [
                        clip_world(
                            p[0], p[1], env_params.world_x_size, env_params.world_y_size
                        )
                        for p in square
                    ]
                )
                ax.add_patch(
                    Polygon(
                        bounded_square, facecolor="none", edgecolor="blue", linewidth=2, zorder=4
                    )
                )

            elif sensor_params.sensor_type == "camera_1d" and get_smoke_density_sensor_fn is not None:
                fov_deg = sensor_params.fov_size_degrees
                max_range = sensor_params.max_range
                num_rays = sensor_params.num_rays

                readings, _ = get_smoke_density_sensor_fn(
                    np.array([pos_x, pos_y, angle_rad]), return_location=True
                )

                d_theta = fov_deg / num_rays
                start_angle = angle_deg - fov_deg / 2

                for r_idx in range(num_rays):
                    val = float(readings[r_idx])
                    color = plt.cm.hot(val / sensor_params.opacity_threshold)

                    wedge = Wedge(
                        (pos_x, pos_y),
                        max_range,
                        start_angle + r_idx * d_theta,
                        start_angle + (r_idx + 1) * d_theta,
                        facecolor=color,
                        edgecolor="white",
                        alpha=0.8,
                        zorder=3,
                    )
                    ax.add_patch(wedge)

            # 1. Draw orientation directional arrow
            ax.arrow(
                pos_x,
                pos_y,
                0.2 * np.cos(angle_rad),
                0.2 * np.sin(angle_rad),
                head_width=0.3,
                head_length=0.3,
                fc="b",
                ec="b",
                zorder=5,
            )

            # 2. Render Physical Collision Boundary
            col_radius = env_params.collision_radius
            circ_phys = Circle(
                (pos_x, pos_y),
                radius=col_radius,
                facecolor="none",
                edgecolor="#FF3366",  # Bright Crimson Red
                linestyle="-",
                linewidth=1.6,
                alpha=0.9,
                zorder=4,
            )
            ax.add_patch(circ_phys)

            # 3. Render Control Safety Radius if provided by the experiment controller
            if controller is not None and hasattr(controller, "d_safe"):
                safe_rad = float(controller.d_safe) / 2.0
                circ_safe = Circle(
                    (pos_x, pos_y),
                    radius=safe_rad,
                    facecolor="none",
                    edgecolor="#00FFFF",  # Neon Cyan
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.9,
                    zorder=4,
                )
                ax.add_patch(circ_safe)
