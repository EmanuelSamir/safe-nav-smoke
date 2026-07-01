from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow, Circle, FancyArrow, Polygon, Wedge

from src.utils.geometry import clip_world


def plot_smoke_background(ax: plt.Axes, smoke_simulator: Any) -> Any:
    """Plot the smoke concentration map as a background image."""
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


def plot_goals(ax: plt.Axes, agents: list) -> tuple:
    """Plot the goals of the agents."""
    goal_circles = []
    goal_texts = []
    for i, agent in enumerate(agents):
        if getattr(agent, "goal_location", None) is not None:
            circle = Circle(
                (agent.goal_location[0], agent.goal_location[1]),
                radius=getattr(agent, "goal_radius", 1.0),
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
                f"goal_{agent.agent_id}" if len(agents) > 1 else "goal",
                ha="center",
                va="bottom",
                fontsize=10,
                color="green",
                zorder=10,
            )
            goal_texts.append(text)
    return goal_circles, goal_texts


def plot_controller_rollouts(ax: plt.Axes, controller: Any) -> None:
    """Visualize sampled trajectories or rollouts if the controller supports it."""
    for line in list(ax.lines):
        line.remove()

    if controller is not None and hasattr(controller, "visualize_rollouts"):
        controller.visualize_rollouts(ax, sample_stride=1)
        for line in ax.lines:
            line.set_zorder(3)


def plot_robots_and_sensors(
    ax: plt.Axes,
    env_cfg: Any,
    agents: list,
    sensor_cfg: Any,
    sensor: Any,
    smoke_simulator: Any,
    goal_circles: list = None,
    controller: Any = None,
    get_smoke_density_sensor_fn: Any = None,
) -> None:
    """Plot robots, collision boundaries, and sensor fields of view."""
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
        if getattr(sensor_cfg, "sensor_type", "") == "downwards":
            if hasattr(sensor, "projection_bounds"):
                square = sensor.projection_bounds(smoke_simulator, pos_x, pos_y)
                bounded_square = np.array(
                    [
                        clip_world(
                            p[0],
                            p[1],
                            getattr(env_cfg, "world_x_size", 100),
                            getattr(env_cfg, "world_y_size", 100),
                        )
                        for p in square
                    ]
                )
                ax.add_patch(
                    Polygon(
                        bounded_square, facecolor="none", edgecolor="blue", linewidth=2, zorder=4
                    )
                )

        elif (
            getattr(sensor_cfg, "sensor_type", "") == "camera_1d"
            and get_smoke_density_sensor_fn is not None
        ):
            fov_deg = getattr(sensor_cfg, "fov_size_degrees", 60.0)
            max_range = getattr(sensor_cfg, "max_range", 10.0)
            num_rays = getattr(sensor_cfg, "num_rays", 5)
            opacity_threshold = getattr(sensor_cfg, "opacity_threshold", 1.0)

            readings, _ = get_smoke_density_sensor_fn(
                np.array([pos_x, pos_y, angle_rad]), return_location=True
            )

            d_theta = fov_deg / max(1, num_rays)
            start_angle = angle_deg - fov_deg / 2

            for r_idx in range(num_rays):
                val = float(readings[r_idx]) if r_idx < len(readings) else 0.0
                color = plt.cm.hot(val / opacity_threshold)

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
        col_radius = getattr(env_cfg, "collision_radius", 1.0)
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


def set_ieee_plot_formatting():
    """Apply IEEE Paper Plot Formatting to Matplotlib RC Params."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
    })
