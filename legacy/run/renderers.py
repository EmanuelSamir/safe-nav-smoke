import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arrow, FancyArrow
from matplotlib.ticker import FormatStrFormatter
import imageio.v2 as imageio
from io import BytesIO
import numpy as np
import cv2
from src.utils import clip_world



class RendererDebugger:
    def __init__(self, opts: dict):
        self.opts = opts

        self.frames = []

        self.fig = None
        self.axes = None

        self.cmap = plt.colormaps["RdYlGn"]

        self.reset()

        self.cbar_map = {}

    def render(self, info: dict):
        if not self.opts.get('render'):
            return
        assert self.fig is not None and self.axes is not None, "Figure and axes must be initialized"

        state = info.get('state')
        location = info.get('state')['location']
        angle = info.get('state')['angle']
        smoke_density = info.get('state')['smoke_density']
        smoke_density_location = info.get('state')['smoke_density_location']
        predicted_maps = info.get('predicted_maps')
        predicted_maps_mean = info.get('predicted_maps_mean')
        predicted_maps_std = info.get('predicted_maps_std')
        env = info.get('env')
        builder = info.get('builder')

        self.flush_decoratives(self.axes['pred'])
        self.flush_decoratives(self.axes['pred_mean'])
        self.flush_decoratives(self.axes['pred_std'])

        env._render_frame(fig=self.fig, ax=self.axes['env'])
        builder.plot_map(risk_map=predicted_maps[-1][1], x_robot_pos=location[0], y_robot_pos=location[1], fig=self.fig, ax=self.axes['pred'], title="Predicted Risk (CVaR) t+H Map")

        builder.plot_map(risk_map=predicted_maps_mean[0][1], x_robot_pos=location[0], y_robot_pos=location[1], fig=self.fig, ax=self.axes['pred_mean'], title="Predicted Mean t+1 Map")
        builder.plot_map(risk_map=predicted_maps_mean[-1][1], x_robot_pos=location[0], y_robot_pos=location[1], fig=self.fig, ax=self.axes['pred_std'], title="Predicted Mean t+H Map")


        im = self.axes['env'].images[0]
        if self.cbar_map.get('env') is None:
            self.cbar_map['env'] = self.fig.colorbar(im, ax=self.axes['env'], orientation='horizontal', location='bottom', pad=0.05)
            self.cbar_map['env'].set_label("Smoke Density", fontsize=12)

        im = self.axes['pred'].images[0]
        if self.cbar_map.get('pred') is None:
            self.cbar_map['pred'] = self.fig.colorbar(im, ax=self.axes['pred'], orientation='horizontal', location='bottom', pad=0.05)
            self.cbar_map['pred'].set_label("Value", fontsize=12)
        
        im = self.axes['pred_mean'].images[0]
        if self.cbar_map.get('pred_mean') is None:
            self.cbar_map['pred_mean'] = self.fig.colorbar(im, ax=self.axes['pred_mean'], orientation='horizontal', location='bottom', pad=0.05)
            self.cbar_map['pred_mean'].set_label("Value", fontsize=12)

        im = self.axes['pred_std'].images[0]
        if self.cbar_map.get('pred_std') is None:
            self.cbar_map['pred_std'] = self.fig.colorbar(im, ax=self.axes['pred_std'], orientation='horizontal', location='bottom', pad=0.05)
            self.cbar_map['pred_std'].set_label("Value", fontsize=12)

        self.add_decoratives(self.axes['pred'], info)
        self.add_decoratives(self.axes['pred_mean'], info)
        self.add_decoratives(self.axes['pred_std'], info)

        if self.opts.get('render') == 'human':
            plt.pause(self.opts.get('env_params').clock)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_decoratives(self, ax, info: dict):
        location = info.get('state')['location']
        angle = info.get('state')['angle']
        path = info.get('path')
        env = info.get('env')
        nom_controller = info.get('nom_controller')

        # Draw the robot's pose as an arrow
        ax.arrow(location[0], location[1], np.cos(angle)*0.1, np.sin(angle)*0.1, 
                                head_width=0.8, head_length=0.8, fc='blue', ec='blue', zorder=10)

        if len(path) > 0:
            ax.scatter(np.array(path)[:, 0], np.array(path)[:, 1], color='green', label='Nominal Path', marker='.', s=0.4, zorder=10)

        # Plot the goal location
        if self.opts.get('inference') == 'global':
            circle = Circle((env.env_params.goal_location[0], env.env_params.goal_location[1]), radius=env.env_params.goal_radius, color='g', fill=True, alpha=0.8)
            (xc, yc), r = circle.center, circle.radius

            ax.add_patch(circle)
            ax.text(xc, yc - r - env.env_params.goal_radius - 0.5, "goal", ha="center", va="bottom",
                    fontsize=10, color="green", zorder=10)

        square = env.sensor.projection_bounds(location[0], location[1])

        bounded_square = np.array([clip_world(p[0], p[1], env.env_params.world_x_size, env.env_params.world_y_size) for p in square])
        ax.add_patch(Polygon(bounded_square, facecolor='none', edgecolor='blue', linewidth=2,))

        trajectories = nom_controller.get_sampled_trajectories()
        if trajectories is not None:
            omega = nom_controller.planner.omega.detach().cpu()
            weighted_traj = (omega[:, None, None] * trajectories).sum(dim=0).numpy()
            omega_norm = (omega - omega.min()) / (omega.max() - omega.min() + 1e-9)
            sorted_indices = np.argsort(omega_norm)
            # for i, traj in enumerate(trajectories[sorted_indices]):
            #     ax.plot(traj[:, 0], traj[:, 1], color='purple', alpha=0.4, linewidth=1.0)
            ax.plot(weighted_traj[:, 0], weighted_traj[:, 1], color="purple", linewidth=2)

    def flush_decoratives(self, ax: plt.Axes):
        for patch in ax.patches:
            if isinstance(patch, (FancyArrow, Arrow, Polygon)):  
                patch.remove()

        for patch in (ax.collections + ax.lines):
            patch.remove()

    def save_frame(self):
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            buf = BytesIO()
            self.fig.savefig(buf, format='png')
            buf.seek(0)
            image = imageio.imread(buf)
            self.frames.append(image)

    def reset(self):
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            if self.frames:
                time_clock_vis = 0.5
                imageio.mimsave(f'{self.opts["seq_filepath"]}', self.frames, duration=time_clock_vis, loop=0)
            
                fps = 1 / 0.1   # igual que tu duration=0.5 → 2 fps
                filepath = self.opts["seq_filepath"].replace(".gif", ".mp4")
                target_w, target_h = 1920, 1080   # 1080p exacto


                import cv2
                writer = imageio.get_writer(
                    filepath,
                    fps=10,                   # o tu fps = 1/duration
                    codec='libx264',
                    quality=10,
                    pixelformat='yuv420p',
                    macro_block_size=16      # requerido para compatibilidad total
                )

                for frame in self.frames:
                    h, w = frame.shape[:2]

                    # --- Fondo blanco 1080p ---
                    canvas = np.ones((target_h, target_w, 4), dtype=np.uint8) * 255

                    # --- Escalar manteniendo aspect ratio ---
                    scale = min(target_w / w, target_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)

                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # --- Centrar ---
                    x0 = (target_w - new_w) // 2
                    y0 = (target_h - new_h) // 2

                    canvas[y0:y0+new_h, x0:x0+new_w] = resized

                    writer.append_data(canvas)

                writer.close()
            else:
                print(f"No frames to save for {self.opts['seq_filepath']}")

        # Clean up and store
        self.fig = plt.figure(figsize=(10, 8))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[0.5, 0.5], width_ratios=[0.5, 0.5])

        self.axes = {}
        self.axes['env'] = self.fig.add_subplot(gs[0, 0])
        self.axes['pred'] = self.fig.add_subplot(gs[0, 1])

        self.axes['pred_mean'] = self.fig.add_subplot(gs[1, 0])
        self.axes['pred_std'] = self.fig.add_subplot(gs[1, 1])

        for ax in list(self.axes.values()):
            ax.set_xlim(0, self.opts.get('env_params').world_x_size)
            ax.set_ylim(0, self.opts.get('env_params').world_y_size)
            ax.autoscale(False)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        self.fig.tight_layout()



class SimpleRenderer:
    def __init__(self, opts: dict):
        self.opts = opts

        self.frames = []

        self.fig = None
        self.axes = None

        self.cmap = plt.colormaps["RdYlGn"]

        self.reset()

        self.cbar_map = {}

    def render(self, info: dict):
        if not self.opts.get('render'):
            return
        assert self.fig is not None and self.axes is not None, "Figure and axes must be initialized"

        state = info.get('state')
        location = info.get('state')['location']
        angle = info.get('state')['angle']
        smoke_density = info.get('state')['smoke_density']
        smoke_density_location = info.get('state')['smoke_density_location']
        env = info.get('env')

        env._render_frame(fig=self.fig, ax=self.axes['env'])

        im = self.axes['env'].images[0]
        if self.cbar_map.get('env') is None:
            self.cbar_map['env'] = self.fig.colorbar(im, ax=self.axes['env'], orientation='horizontal', location='bottom', pad=0.05)
            self.cbar_map['env'].set_label("Smoke Density", fontsize=12)

        self.plot_smoke(self.axes['risk'], info)
        self.plot_velocity(self.axes['velocity'], info)
        self.plot_angular_velocity(self.axes['angular_velocity'], info)

        if self.opts.get('render') == 'human':
            plt.pause(self.opts.get('env_params').clock)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_smoke(self, ax, info):
        smoke_on_robot = info.get('logger_metrics').get_values('smoke_on_robot')
        smoke_on_robot_acc = info.get('logger_metrics').get_values('smoke_on_robot_acc')
        steps = info.get('logger_metrics').get_values('steps')

        if not len(ax.lines):
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.plot(steps, smoke_on_robot, color='red', label='smoke')
            ax.set_ylabel('Smoke at Robot Pos.')
            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, smoke_on_robot)
            ax.set_ylim(0, 1)
        
        self.fig.tight_layout()

    def plot_velocity(self, ax, info):
        velocity = info.get('logger_metrics').get_values('action_0')
        steps = info.get('logger_metrics').get_values('steps')
        if not len(ax.lines):
            ax.plot(steps, velocity, color='orange', label='velocity')
            # ax.set_xlabel('Time step')
            ax.set_ylabel('Linear velocity')
            ax.set_xlim(0, self.opts.get('env_params').max_steps)

            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, velocity)
            ax.set_ylim(info.get('env').robot_params.action_min[0], info.get('env').robot_params.action_max[0])
        self.fig.tight_layout()

    def plot_angular_velocity(self, ax, info):
        angular_velocity = info.get('logger_metrics').get_values('action_1')
        steps = info.get('logger_metrics').get_values('steps')
        if not len(ax.lines):
            ax.plot(steps, angular_velocity, color='purple', label='angular velocity')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Angular velocity')
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, angular_velocity)
            ax.set_ylim(info.get('env').robot_params.action_min[1], info.get('env').robot_params.action_max[1])
        self.fig.tight_layout()

    def save_frame(self):
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            buf = BytesIO()
            self.fig.savefig(buf, format='png')
            buf.seek(0)
            image = imageio.imread(buf)
            self.frames.append(image)

    def reset(self):
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            if self.frames:
                time_clock_vis = 0.5
                imageio.mimsave(f'{self.opts["seq_filepath"]}', self.frames, duration=time_clock_vis, loop=0)
            
                fps = 1 / 0.1   # igual que tu duration=0.5 → 2 fps
                filepath = self.opts["seq_filepath"].replace(".gif", ".mp4")
                target_w, target_h = 1920, 1080   # 1080p exacto


                import cv2
                writer = imageio.get_writer(
                    filepath,
                    fps=10,                   # o tu fps = 1/duration
                    codec='libx264',
                    quality=10,
                    pixelformat='yuv420p',
                    macro_block_size=16      # requerido para compatibilidad total
                )

                for frame in self.frames:
                    h, w = frame.shape[:2]

                    # --- Fondo blanco 1080p ---
                    canvas = np.ones((target_h, target_w, 4), dtype=np.uint8) * 255

                    # --- Escalar manteniendo aspect ratio ---
                    scale = min(target_w / w, target_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)

                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # --- Centrar ---
                    x0 = (target_w - new_w) // 2
                    y0 = (target_h - new_h) // 2

                    canvas[y0:y0+new_h, x0:x0+new_w] = resized

                    writer.append_data(canvas)

                writer.close()
            else:
                print(f"No frames to save for {self.opts['seq_filepath']}")

        # Clean up and store
        self.fig = plt.figure(figsize=(8, 5))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[0.4, 0.4, 0.4], width_ratios=[1.2, 0.6])

        self.axes = {}
        self.axes['env'] = self.fig.add_subplot(gs[0:, 0])
        self.axes['risk'] = self.fig.add_subplot(gs[0, 1])
        self.axes['velocity'] = self.fig.add_subplot(gs[1, 1])
        self.axes['angular_velocity'] = self.fig.add_subplot(gs[2, 1])

        for ax in [self.axes['env']]:
            ax.set_xlim(0, self.opts.get('env_params').world_x_size)
            ax.set_ylim(0, self.opts.get('env_params').world_y_size)
            ax.autoscale(False)

        for ax in [self.axes['risk'], self.axes['velocity'], self.axes['angular_velocity']]:
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.autoscale(tight=True)

        for ax in self.axes.values():
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        self.fig.tight_layout()


class RendererWithRiskMap:
    def __init__(self, opts: dict):
        self.opts = opts

        self.frames = []

        self.fig = None
        self.axes = None

        self.cmap = plt.colormaps["RdYlGn"]

        self.reset()

        self.cbar_map = {}

    def render(self, info: dict):
        if not self.opts.get('render'):
            return
        assert self.fig is not None and self.axes is not None, "Figure and axes must be initialized"

        state = info.get('state')
        location = info.get('state')['location']
        angle = info.get('state')['angle']
        smoke_density = info.get('state')['smoke_density']
        smoke_density_location = info.get('state')['smoke_density_location']
        predicted_maps = info.get('predicted_maps')
        env = info.get('env')
        builder = info.get('builder')

        self.flush_decoratives(self.axes['pred'])

        env._render_frame(fig=self.fig, ax=self.axes['env'])
        builder.plot_map(risk_map=predicted_maps[-1][1], x_robot_pos=location[0], y_robot_pos=location[1], fig=self.fig, ax=self.axes['pred'])

        im = self.axes['env'].images[0]
        if self.cbar_map.get('env') is None:
            self.cbar_map['env'] = self.fig.colorbar(im, ax=self.axes['env'], orientation='horizontal', location='bottom', pad=0.05)
            self.cbar_map['env'].set_label("Smoke Density", fontsize=12)

        im = self.axes['pred'].images[0]
        if self.cbar_map.get('pred') is None:
            self.cbar_map['pred'] = self.fig.colorbar(im, ax=self.axes['pred'], orientation='horizontal', location='bottom', pad=0.05)
            self.cbar_map['pred'].set_label("Predicted Risk Map", fontsize=12)

        self.add_decoratives(self.axes['pred'], info)

        self.plot_smoke(self.axes['risk'], info)
        # self.plot_dist_error(self.axes['dist_to_goal'], info)
        # self.plot_angle_err(self.axes['angle_err'], info)
        self.plot_velocity(self.axes['velocity'], info)
        self.plot_angular_velocity(self.axes['angular_velocity'], info)

        if self.opts.get('render') == 'human':
            plt.pause(self.opts.get('env_params').clock)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_decoratives(self, ax, info: dict):
        location = info.get('state')['location']
        angle = info.get('state')['angle']
        path = info.get('path')
        env = info.get('env')
        nom_controller = info.get('nom_controller')

        # Draw the robot's pose as an arrow
        ax.arrow(location[0], location[1], np.cos(angle)*0.1, np.sin(angle)*0.1, 
                                head_width=0.8, head_length=0.8, fc='blue', ec='blue', zorder=10)

        if len(path) > 0:
            ax.scatter(np.array(path)[:, 0], np.array(path)[:, 1], color='green', label='Nominal Path', marker='.', s=0.4, zorder=10)

        # Plot the goal location
        if self.opts.get('inference') == 'global':
            circle = Circle((env.env_params.goal_location[0], env.env_params.goal_location[1]), radius=env.env_params.goal_radius, color='g', fill=True, alpha=0.8)
            (xc, yc), r = circle.center, circle.radius

            ax.add_patch(circle)
            ax.text(xc, yc - r - env.env_params.goal_radius - 0.5, "goal", ha="center", va="bottom",
                    fontsize=10, color="green", zorder=10)

        square = env.sensor.projection_bounds(location[0], location[1])

        bounded_square = np.array([clip_world(p[0], p[1], env.env_params.world_x_size, env.env_params.world_y_size) for p in square])
        ax.add_patch(Polygon(bounded_square, facecolor='none', edgecolor='blue', linewidth=2,))

        trajectories = nom_controller.get_sampled_trajectories()
        if trajectories is not None:
            omega = nom_controller.planner.omega.detach().cpu()
            weighted_traj = (omega[:, None, None] * trajectories).sum(dim=0).numpy()
            omega_norm = (omega - omega.min()) / (omega.max() - omega.min() + 1e-9)
            sorted_indices = np.argsort(omega_norm)
            # for i, traj in enumerate(trajectories[sorted_indices]):
            #     ax.plot(traj[:, 0], traj[:, 1], color='purple', alpha=0.4, linewidth=1.0)
            ax.plot(weighted_traj[:, 0], weighted_traj[:, 1], color="purple", linewidth=2)

    def plot_smoke(self, ax, info):
        smoke_on_robot = info.get('logger_metrics').get_values('smoke_on_robot')
        smoke_on_robot_acc = info.get('logger_metrics').get_values('smoke_on_robot_acc')
        steps = info.get('logger_metrics').get_values('steps')

        if not len(ax.lines):
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.plot(steps, smoke_on_robot, color='red', label='smoke')
            # ax.plot(steps, smoke_on_robot_acc, color='blue', label='acc smoke', alpha=0.5)
            # ax.set_title('Smoke on Robot')
            # ax.set_xlabel('Time step')
            ax.set_ylabel('Smoke at Robot Pos.')
            ax.grid(True)
            # ax.legend()

        else:
            ax.lines[0].set_data(steps, smoke_on_robot)
            # ax.lines[1].set_data(steps, smoke_on_robot_acc)
            # ax.set_ylim(np.min(smoke_on_robot), np.max(smoke_on_robot_acc))
            ax.set_ylim(0, 1)
        
        self.fig.tight_layout()

    def plot_dist_error(self, ax, info):
        dist_error = info.get('logger_metrics').get_values('dist_to_goal')
        steps = info.get('logger_metrics').get_values('steps')
        if not len(ax.lines):
            ax.plot(steps, dist_error, color='green', label='dist error')
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.set_ylim(0, np.max(dist_error))
            ax.set_xlabel('Time step')
            ax.set_ylabel('Distance error')
            ax.grid(True)
        
        else:
            ax.lines[0].set_data(steps, dist_error)
        self.fig.tight_layout()

    def plot_velocity(self, ax, info):
        velocity = info.get('logger_metrics').get_values('action_0')
        steps = info.get('logger_metrics').get_values('steps')
        if not len(ax.lines):
            ax.plot(steps, velocity, color='orange', label='velocity')
            # ax.set_xlabel('Time step')
            ax.set_ylabel('Linear velocity')
            ax.set_xlim(0, self.opts.get('env_params').max_steps)

            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, velocity)
            ax.set_ylim(info.get('env').robot_params.action_min[0], info.get('env').robot_params.action_max[0])
        self.fig.tight_layout()

    def plot_angular_velocity(self, ax, info):
        angular_velocity = info.get('logger_metrics').get_values('action_1')
        steps = info.get('logger_metrics').get_values('steps')
        if not len(ax.lines):
            ax.plot(steps, angular_velocity, color='purple', label='angular velocity')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Angular velocity')
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, angular_velocity)
            ax.set_ylim(info.get('env').robot_params.action_min[1], info.get('env').robot_params.action_max[1])
        self.fig.tight_layout()

    def plot_angle_err(self, ax, info):
        angle_err = info.get('logger_metrics').get_values('angle_err')
        steps = info.get('logger_metrics').get_values('steps')
        if not len(ax.lines):
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.set_ylim(0, np.pi)
            ax.plot(steps, angle_err, color='purple', label='angle err')
            # ax.set_xlabel('Time step')
            ax.set_ylabel('Angle error')
            ax.grid(True)
        else:
            ax.lines[0].set_data(steps, angle_err)

        self.fig.tight_layout()

    def flush_decoratives(self, ax: plt.Axes):
        for patch in ax.patches:
            if isinstance(patch, (FancyArrow, Arrow, Polygon)):  
                patch.remove()

        for patch in (ax.collections + ax.lines):
            patch.remove()

    def save_frame(self):
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            buf = BytesIO()
            self.fig.savefig(buf, format='png')
            buf.seek(0)
            image = imageio.imread(buf)
            self.frames.append(image)

    def reset(self):
        if self.fig is not None and self.opts.get('seq_filepath') and self.opts.get('render'):
            if self.frames:
                time_clock_vis = 0.5
                imageio.mimsave(f'{self.opts["seq_filepath"]}', self.frames, duration=time_clock_vis, loop=0)
            
                fps = 1 / 0.1   # igual que tu duration=0.5 → 2 fps
                filepath = self.opts["seq_filepath"].replace(".gif", ".mp4")
                target_w, target_h = 1920, 1080   # 1080p exacto


                import cv2
                writer = imageio.get_writer(
                    filepath,
                    fps=10,                   # o tu fps = 1/duration
                    codec='libx264',
                    quality=10,
                    pixelformat='yuv420p',
                    macro_block_size=16      # requerido para compatibilidad total
                )

                for frame in self.frames:
                    h, w = frame.shape[:2]

                    # --- Fondo blanco 1080p ---
                    canvas = np.ones((target_h, target_w, 4), dtype=np.uint8) * 255

                    # --- Escalar manteniendo aspect ratio ---
                    scale = min(target_w / w, target_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)

                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # --- Centrar ---
                    x0 = (target_w - new_w) // 2
                    y0 = (target_h - new_h) // 2

                    canvas[y0:y0+new_h, x0:x0+new_w] = resized

                    writer.append_data(canvas)

                writer.close()
            else:
                print(f"No frames to save for {self.opts['seq_filepath']}")

        # Clean up and store
        self.fig = plt.figure(figsize=(12, 5))
        gs = self.fig.add_gridspec(3, 3, height_ratios=[0.4, 0.4, 0.4], width_ratios=[1.2, 0.6, 1.2])

        self.axes = {}
        self.axes['env'] = self.fig.add_subplot(gs[0:, 0])
        self.axes['risk'] = self.fig.add_subplot(gs[0, 1])
        # self.axes['angle_err'] = self.fig.add_subplot(gs[1, 1])
        # self.axes['dist_to_goal'] = self.fig.add_subplot(gs[2, 1])

        self.axes['velocity'] = self.fig.add_subplot(gs[1, 1])
        self.axes['angular_velocity'] = self.fig.add_subplot(gs[2, 1])
        self.axes['pred'] = self.fig.add_subplot(gs[0:, 2])

        for ax in [self.axes['env'], self.axes['pred']]:
            ax.set_xlim(0, self.opts.get('env_params').world_x_size)
            ax.set_ylim(0, self.opts.get('env_params').world_y_size)
            ax.autoscale(False)

        for ax in [self.axes['risk'], self.axes['velocity'], self.axes['angular_velocity']]:
            ax.set_xlim(0, self.opts.get('env_params').max_steps)
            ax.autoscale(tight=True)

        for ax in self.axes.values():
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        self.fig.tight_layout()
