"""
Standard renderer for safe navigation experiments.
Consolidates Renderer classes from multiple experiment files.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import FancyArrow, Arrow, Circle, Polygon
from io import BytesIO
import imageio.v2 as imageio
from typing import Dict, Any, Optional
from src.utils import clip_world


class StandardRenderer:
    """
    Standard renderer for visualization of safe navigation experiments.
    
    Handles:
    - Environment visualization
    - Robot trajectory
    - Predicted risk maps  
    - Real-time metrics plots
    - GIF and MP4 export
    
    Consolidates duplicate Renderer classes from:
    - run/experiment_1.py
    - run/experiment_5_rssm.py
    - run/experiment_6_vaessm.py
    """
    
    def __init__(self, opts: Dict[str, Any]):
        """
        Initialize renderer.
        
        Args:
            opts: Configuration dictionary containing:
                - render: Whether to render
                - env_params: Environment parameters
                - seq_filepath: Path to save animation (optional)
        """
        self.opts = opts
        self.frames = []
        
        # Avoid opening an interactive GUI window if saving frames directly
        if self.opts.get('render') == 'rgb_array':
            plt.switch_backend('Agg')
        
        self.fig = None
        self.axes = None
        self.cmap = plt.colormaps["RdYlGn"]
        self.cbar_map = {}
        
        self.reset()
    
    def render(self, info: Dict[str, Any]):
        """
        Render a single frame.
        
        Args:
            info: Dictionary containing:
                - state: Current state
                - env: Environment instance
                - builder: Risk map builder
                - nom_controller: Nominal controller
                - logger_metrics: Metrics logger
                - predicted_maps: Predicted risk maps
                - path: Robot trajectory
        """
        if not self.opts.get('render'):
            return
        
        assert self.fig is not None and self.axes is not None, \
            "Figure and axes must be initialized"
        
        # Extract info
        state = info.get('state')
        location = state['location']
        angle = state['angle']
        predicted_maps = info.get('predicted_maps')
        env = info.get('env')
        builder = info.get('builder')
        
        # Clear decoratives from previous frame
        if 'pred' in self.axes:
            main_ax = self.axes['pred']
            self.flush_decoratives(main_ax)
        
        # Render environment
        env._render_frame(fig=self.fig, ax=self.axes['env'])
        
        # Render predicted risk map
        if predicted_maps and 'pred' in self.axes:
            if builder is not None:
                builder.plot_map(
                    risk_map=predicted_maps[-1][1],
                    x_robot_pos=location[0],
                    y_robot_pos=location[1],
                    fig=self.fig,
                )
            else:
                horizon = 10
                coords, map_data = predicted_maps[horizon]
                x_size, y_size = env.env_params.world_x_size, env.env_params.world_y_size
                ax = self.axes['pred']
                if len(coords) == len(map_data):
                    # Compute accurate grid shape from coordinates
                    W = len(np.unique(coords[:, 0]))
                    H = len(np.unique(coords[:, 1]))
                    
                    if H * W == len(map_data):
                        # Reshape the 1D map array to 2D
                        map_2d = map_data.reshape(H, W)
                        
                        ax.imshow(
                            map_2d, 
                            cmap='RdYlGn_r', 
                            vmin=0.0, vmax=1.0, 
                            origin='lower',
                            extent=[0, x_size, 0, y_size]
                        )
                    else:
                        print(f"Warning: Map data length ({len(map_data)}) doesn't match grid HxW ({H}x{W}).")
    
            pred_mappable = None
            if self.axes['pred'].images:
                pred_mappable = self.axes['pred'].images[0]
            elif hasattr(self.axes['pred'], '_mapped_scatter'):
                pred_mappable = self.axes['pred']._mapped_scatter
                
            if pred_mappable and self.cbar_map.get('pred') is None:
                self.cbar_map['pred'] = self.fig.colorbar(
                    pred_mappable, ax=self.axes['pred'],
                    orientation='horizontal',
                    location='bottom',
                    pad=0.08
                )
                self.cbar_map['pred'].set_label(f"Predicted Risk Map at horizon = {horizon}", fontsize=12)
            
        # Add colorbars (only once)
        if hasattr(self.axes['env'], 'images') and self.axes['env'].images:
            im = self.axes['env'].images[0]
            if self.cbar_map.get('env') is None:
                self.cbar_map['env'] = self.fig.colorbar(
                    im, ax=self.axes['env'],
                    orientation='horizontal',
                    location='bottom',
                    pad=0.08
                )
                self.cbar_map['env'].set_label("Smoke Density", fontsize=12)
        
        # Add decoratives (robot, trajectory, etc.)
        if 'pred' in self.axes:
            self.add_decoratives(self.axes['pred'], info)
        
        # Plot metrics
        self.plot_smoke(self.axes['risk'], info)
        self.plot_velocity(self.axes['velocity'], info)
        self.plot_angular_velocity(self.axes['angular_velocity'], info)
        
        # Update display
        self.fig.canvas.draw()
        
        if self.opts.get('render') == 'human':
            self.fig.canvas.flush_events()
            plt.pause(self.opts.get('env_params').clock)
    
    def add_decoratives(self, ax: plt.Axes, info: Dict[str, Any]):
        """Add robot pose, trajectory, and other visual elements."""
        location = info.get('state')['location']
        angle = info.get('state')['angle']
        path = info.get('path', [])
        env = info.get('env')
        nom_controller = info.get('nom_controller')
        
        # Draw robot pose as arrow
        ax.arrow(
            location[0], location[1],
            np.cos(angle) * 0.1, np.sin(angle) * 0.1,
            head_width=0.8, head_length=0.8,
            fc='blue', ec='blue', zorder=10
        )
        
        # Draw trajectory
        if len(path) > 0:
            path_array = np.array(path)
            ax.scatter(
                path_array[:, 0], path_array[:, 1],
                color='green', label='Nominal Path',
                marker='.', s=0.4, zorder=10
            )
        
        goal_loc = env.env_params.goal_location
        goal_radius = env.env_params.goal_radius
        
        circle = Circle(
            (goal_loc[0], goal_loc[1]),
            radius=goal_radius,
            color='black', fill=True, alpha=0.8
        )
        
        ax.add_patch(circle)
        ax.text(
            goal_loc[0], goal_loc[1] - goal_radius - 1.5,
            "goal", ha="center", va="bottom",
            fontsize=10, color="black", zorder=10
        )
        
        # Draw sensor field of view
        square = env.sensor.projection_bounds(location[0], location[1])
        bounded_square = np.array([
            clip_world(p[0], p[1],
                      env.env_params.world_x_size,
                      env.env_params.world_y_size)
            for p in square
        ])
        ax.add_patch(Polygon(
            bounded_square,
            facecolor='none',
            edgecolor='blue',
            linewidth=2
        ))
        
        # Draw sampled trajectories from MPPI
        if nom_controller:
            trajectories = nom_controller.get_sampled_trajectories()
            if trajectories is not None:
                omega = nom_controller.planner.omega.detach().cpu()
                weighted_traj = (omega[:, None, None] * trajectories).sum(dim=0).numpy()
                ax.plot(
                    weighted_traj[:, 0], weighted_traj[:, 1],
                    color="purple", linewidth=2, zorder=10
                )
    
    def plot_smoke(self, ax: plt.Axes, info: Dict[str, Any]):
        """Plot smoke exposure over time."""
        logger_metrics = info.get('logger_metrics')
        smoke_on_robot = logger_metrics.get_values('smoke_on_robot')
        steps = logger_metrics.get_values('steps')
        
        if not ax.lines:
            # Initialize plot
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.plot(steps, smoke_on_robot, color='red', label='smoke')
            ax.set_ylabel('Smoke at Robot Pos.')
            ax.grid(True)
        else:
            # Update plot
            ax.lines[0].set_data(steps, smoke_on_robot)
            ax.set_ylim(0, 1)
        
        self.fig.tight_layout()
    
    def plot_dist_error(self, ax: plt.Axes, info: Dict[str, Any]):
        """Plot distance to goal over time."""
        logger_metrics = info.get('logger_metrics')
        dist_error = logger_metrics.get_values('dist_to_goal')
        steps = logger_metrics.get_values('steps')
        
        if not ax.lines:
            ax.plot(steps, dist_error, color='green', label='dist error')
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.set_ylim(0, np.max(dist_error) if dist_error else 1)
            ax.set_xlabel('Time step')
            ax.set_ylabel('Distance error')
            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, dist_error)
        
        self.fig.tight_layout()
    
    def plot_velocity(self, ax: plt.Axes, info: Dict[str, Any]):
        """Plot linear velocity over time."""
        logger_metrics = info.get('logger_metrics')
        velocity = logger_metrics.get_values('action_0')
        steps = logger_metrics.get_values('steps')
        
        if not ax.lines:
            ax.plot(steps, velocity, color='orange', label='velocity')
            ax.set_ylabel('Linear velocity')
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, velocity)
            env = info.get('env')
            ax.set_ylim(
                env.robot_params.action_min[0],
                env.robot_params.action_max[0]
            )
        
        self.fig.tight_layout()
    
    def plot_angular_velocity(self, ax: plt.Axes, info: Dict[str, Any]):
        """Plot angular velocity over time."""
        logger_metrics = info.get('logger_metrics')
        angular_velocity = logger_metrics.get_values('action_1')
        steps = logger_metrics.get_values('steps')
        
        if not ax.lines:
            ax.plot(steps, angular_velocity, color='purple', label='angular velocity')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Angular velocity')
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, angular_velocity)
            env = info.get('env')
            ax.set_ylim(
                env.robot_params.action_min[1],
                env.robot_params.action_max[1]
            )
        
        self.fig.tight_layout()
    
    def plot_angle_err(self, ax: plt.Axes, info: Dict[str, Any]):
        """Plot angle error over time."""
        logger_metrics = info.get('logger_metrics')
        angle_err = logger_metrics.get_values('angle_err')
        steps = logger_metrics.get_values('steps')
        
        if not ax.lines:
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.set_ylim(0, np.pi)
            ax.plot(steps, angle_err, color='purple', label='angle err')
            ax.set_ylabel('Angle error')
            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, angle_err)
        
        self.fig.tight_layout()
    
    def flush_decoratives(self, ax: plt.Axes):
        """Remove all decorative elements from axes."""
        # Remove patches (arrows, polygons, circles)
        for patch in ax.patches[:]:  # Use slice to avoid modification during iteration
            if isinstance(patch, (FancyArrow, Arrow, Polygon)):
                patch.remove()
        
        # Remove collections and lines
        for item in (ax.collections + ax.lines)[:]:
            item.remove()
    
    def save_frame(self):
        """Save current frame to buffer."""
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            buf = BytesIO()
            self.fig.savefig(buf, format='png')
            buf.seek(0)
            image = imageio.imread(buf)
            self.frames.append(image)
    
    def reset(self):
        """Reset renderer and save animation if frames exist."""
        # Save animation if we have frames
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            if self.frames:
                self._save_animation()
            else:
                print(f"No frames to save for {self.opts.get('seq_filepath')}")
        
        # Create new figure
        self.fig = plt.figure(figsize=(10, 5))
        has_preds = self.opts.get('has_predictions', False)
        self.axes = {}

        if has_preds:
            gs = self.fig.add_gridspec(
                3, 3,
                height_ratios=[0.4, 0.4, 0.4],
                width_ratios=[1.2, 0.6, 1.2]
            )
            self.axes['env'] = self.fig.add_subplot(gs[0:, 0])
            self.axes['risk'] = self.fig.add_subplot(gs[0, 1])
            self.axes['velocity'] = self.fig.add_subplot(gs[1, 1])
            self.axes['angular_velocity'] = self.fig.add_subplot(gs[2, 1])
            self.axes['pred'] = self.fig.add_subplot(gs[0:, 2])
            spatial_axes = [self.axes['env'], self.axes['pred']]
        else:
            gs = self.fig.add_gridspec(
                3, 2,
                height_ratios=[0.33, 0.33, 0.33],
                width_ratios=[1.5, 0.7]
            )
            self.axes['env'] = self.fig.add_subplot(gs[0:, 0])
            self.axes['risk'] = self.fig.add_subplot(gs[0, 1])
            self.axes['velocity'] = self.fig.add_subplot(gs[1, 1])
            self.axes['angular_velocity'] = self.fig.add_subplot(gs[2, 1])
            spatial_axes = [self.axes['env']]
            self.axes['env'].set_title('Simulation', fontsize=12)
        
        # Configure spatial axes
        env_params = self.opts.get('env_params')
        for ax in spatial_axes:
            ax.set_xlim(0, env_params.world_x_size)
            ax.set_ylim(0, env_params.world_y_size)
            ax.autoscale(False)
        
        # Configure time-series axes
        for ax in [self.axes['risk'], self.axes['velocity'], self.axes['angular_velocity']]:
            ax.set_xlim(0, env_params.max_steps)
            ax.autoscale(tight=True)
        
        # Format all axes
        for ax in self.axes.values():
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        self.fig.tight_layout()
        
        # Reset frames
        self.frames = []
    
    def _save_animation(self):
        """Save frames as GIF and MP4."""
        import cv2
        
        seq_filepath = self.opts['seq_filepath']
        
        # Save GIF
        time_clock_vis = len(self.frames) * self.opts['env_params'].clock
        imageio.mimsave(seq_filepath, self.frames, duration=time_clock_vis, loop=0)
        
        # # Save MP4 (high quality, 1080p equivalent divisible by 16)
        # filepath_mp4 = seq_filepath.replace(".gif", ".mp4")
        # target_w, target_h = 1920, 1088
        
        # writer = imageio.get_writer(
        #     filepath_mp4,
        #     fps=10,
        #     codec='libx264',
        #     quality=10,
        #     pixelformat='yuv420p',
        #     macro_block_size=16
        # )
        
        # for frame in self.frames:
        #     h, w = frame.shape[:2]
            
        #     # Create white 1080p canvas
        #     canvas = np.ones((target_h, target_w, 4), dtype=np.uint8) * 255
            
        #     # Scale while maintaining aspect ratio
        #     scale = min(target_w / w, target_h / h)
        #     new_w = int(w * scale)
        #     new_h = int(h * scale)
            
        #     resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        #     # Center on canvas
        #     x0 = (target_w - new_w) // 2
        #     y0 = (target_h - new_h) // 2
            
        #     canvas[y0:y0+new_h, x0:x0+new_w] = resized
            
        #     writer.append_data(canvas)
        
        # writer.close()
    
    def close(self):
        """Close the renderer and clean up."""
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            if self.frames:
                self._save_animation()
            else:
                print(f"No frames to save for {self.opts.get('seq_filepath')}")

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None


# Backward compatibility alias
Renderer = StandardRenderer
