from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.patches import Arrow, Circle, FancyArrow, Polygon, Wedge
from omegaconf import DictConfig, OmegaConf

from agents.basic_robot import RobotParams
from agents.dubins_robot import DubinsRobot
from agents.dubins_robot_fixed_velocity import DubinsRobotFixedVelocity
from agents.unicycle_robot import UnicycleRobot
from envs.simulator.playback import Playback, PlaybackParams
from envs.simulator.sensor import (
    Camera1DSensor,
    Camera1DSensorParams,
    DownwardsSensor,
    DownwardsSensorParams,
    GlobalSensor,
    GlobalSensorParams,
    PointSensor,
    PointSensorParams,
)
from envs.simulator.smoke import BlobParams, Smoke, SmokeParams
from utils import clip_world


class RenderMode(str, Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    NONE = "none"


@dataclass
class EnvParams:
    world_x_size: float = field(default=100.0)
    world_y_size: float = field(default=100.0)
    max_steps: int = field(default=200)
    render: RenderMode = field(default=RenderMode.HUMAN)
    render_save_every: int = field(default=2)
    clock: float = field(default=0.1)

    goal_location: tuple[float, float] | None = field(default=None)
    goal_radius: float = field(default=1.0)

    smoke_density_threshold: float = None
    sensor_params: Any = field(default=None)
    playback_path: str | None = None


class SmokeEnv(gym.Env):
    def __init__(
        self,
        cfg: Union[str, DictConfig, dict] = "configs/env/smoke_env.yaml",
        robot_params: RobotParams | None = None,
    ) -> None:
        """Initializes the Smoke Environment from a Hydra/OmegaConf configuration."""
        super().__init__()

        # 1. Load/Parse the configuration
        if isinstance(cfg, str):
            cfg = OmegaConf.load(cfg)
        elif isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        # 2. Setup core environment parameters
        self._setup_env_params(cfg)

        # 3. Setup sensor parameters
        self._setup_sensor_params(cfg)

        # 4. Setup smoke and robot parameters
        self._setup_smoke_and_robot(cfg, robot_params)

        # 5. Initialize the simulator (Playback vs Dynamic)
        self._setup_simulator()

        # 6. Initialize spaces
        self._init_spaces()

        # Render window state
        self.window = {"fig": None, "ax": None, "cax": None}
        self.clock = self.env_params.clock
        self.current_step = 0

    def _setup_env_params(self, cfg: dict):
        """Initializes self.env_params from the YAML file using class defaults."""
        defaults = EnvParams()
        self.env_params = EnvParams()
        self.env_params.world_x_size = float(cfg.get("world_x_size", defaults.world_x_size))
        self.env_params.world_y_size = float(cfg.get("world_y_size", defaults.world_y_size))
        self.env_params.max_steps = int(cfg.get("max_steps", defaults.max_steps))
        self.env_params.render = RenderMode(cfg.get("render", defaults.render))
        self.env_params.render_save_every = int(
            cfg.get("render_save_every", defaults.render_save_every)
        )
        self.env_params.clock = float(cfg.get("clock", defaults.clock))
        self.env_params.playback_path = cfg.get("playback_path", defaults.playback_path)

        if cfg.get("goal_location"):
            self.env_params.goal_location = tuple(cfg["goal_location"])
        self.env_params.goal_radius = float(cfg.get("goal_radius", defaults.goal_radius))

    def _setup_sensor_params(self, cfg: dict):
        """Decodes the 'sensor' block from YAML using class defaults."""
        sensor_cfg = cfg.get("sensor", {})
        sensor_type = sensor_cfg.get("type", "global")

        common_args = {
            "world_x_size": self.env_params.world_x_size,
            "world_y_size": self.env_params.world_y_size,
        }

        if sensor_type == "camera_1d":
            d = Camera1DSensorParams(**common_args)
            self.env_params.sensor_params = Camera1DSensorParams(
                fov_size_degrees=float(sensor_cfg.get("fov_size_degrees", d.fov_size_degrees)),
                num_rays=int(sensor_cfg.get("num_rays", d.num_rays)),
                step_size=float(sensor_cfg.get("step_size", d.step_size)),
                opacity_threshold=float(sensor_cfg.get("opacity_threshold", d.opacity_threshold)),
                max_range=float(sensor_cfg.get("max_range", d.max_range)),
                **common_args,
            )
        elif sensor_type == "downwards":
            self.env_params.sensor_params = DownwardsSensorParams(**common_args)
        elif sensor_type == "point":
            self.env_params.sensor_params = PointSensorParams(**common_args)
        else:
            self.env_params.sensor_params = GlobalSensorParams(**common_args)

    def _setup_smoke_and_robot(self, cfg: dict, robot_params_override: RobotParams | None):
        """Logic to prepare SmokeParams and Robot objects."""
        # --- Smoke setup ---
        smoke_cfg = cfg.get("smoke", {})
        smoke_blobs = []
        if "blobs" in smoke_cfg:
            for b in smoke_cfg["blobs"]:
                smoke_blobs.append(
                    BlobParams(
                        x_pos=float(b["x"]),
                        y_pos=float(b["y"]),
                        intensity=float(b["intensity"]),
                        spread_rate=float(b["spread"]),
                    )
                )
        else:
            smoke_blobs = [BlobParams(x_pos=10, y_pos=40, intensity=1.0, spread_rate=1.0)]

        d_smoke = SmokeParams(x_size=10, y_size=10, smoke_blob_params=[])
        self.smoke_params = SmokeParams(
            x_size=self.env_params.world_x_size,
            y_size=self.env_params.world_y_size,
            smoke_blob_params=smoke_blobs,
            resolution=float(smoke_cfg.get("resolution", d_smoke.resolution)),
        )

        # --- Robot setup ---
        if robot_params_override is not None:
            self.robot_params = robot_params_override
        elif "robot" in cfg:
            rcfg = cfg["robot"]
            d_robot = RobotParams()
            self.robot_params = RobotParams(
                action_dim=int(rcfg.get("action_dim", d_robot.action_dim)),
                state_dim=int(rcfg.get("state_dim", d_robot.state_dim)),
                action_max=list(rcfg.get("action_max", d_robot.action_max)),
                action_min=list(rcfg.get("action_min", d_robot.action_min)),
                state_max=list(rcfg.get("state_max", d_robot.state_max)),
                state_min=list(rcfg.get("state_min", d_robot.state_min)),
                robot_type=rcfg.get("type", d_robot.robot_type),
                dt=float(rcfg.get("dt", d_robot.dt)),
            )
        else:
            self.robot_params = RobotParams()

        self.robot_params.state_max[0] = self.env_params.world_x_size
        self.robot_params.state_max[1] = self.env_params.world_y_size

    def _setup_simulator(self):
        """Choose between Playback or Dynamic simulation and create the robot."""
        if self.env_params.playback_path:
            p_params = PlaybackParams(data_path=self.env_params.playback_path)
            self.smoke_simulator = Playback(params=p_params)

            # Playback data overrides world sizes if provided
            if self.smoke_simulator.x_size:
                self.env_params.world_x_size = self.smoke_simulator.x_size
            if self.smoke_simulator.y_size:
                self.env_params.world_y_size = self.smoke_simulator.y_size
            if self.smoke_simulator.max_steps:
                self.env_params.max_steps = self.smoke_simulator.max_steps
        else:
            self.smoke_simulator = Smoke(params=self.smoke_params)

        # Re-sync sensor world sizes
        self.env_params.sensor_params.world_x_size = self.env_params.world_x_size
        self.env_params.sensor_params.world_y_size = self.env_params.world_y_size

        stype = self.env_params.sensor_params.sensor_type
        if stype == "camera_1d":
            self.sensor = Camera1DSensor(self.env_params.sensor_params)
        elif stype == "downwards":
            self.sensor = DownwardsSensor(self.env_params.sensor_params)
        elif stype == "point":
            self.sensor = PointSensor(self.env_params.sensor_params)
        else:
            self.sensor = GlobalSensor(self.env_params.sensor_params)

        # Robot Object creation
        rtype = self.robot_params.robot_type
        if rtype == "unicycle":
            self.robot = UnicycleRobot(self.robot_params)
        elif rtype == "dubins2d":
            self.robot = DubinsRobot(self.robot_params)
        elif rtype == "dubins2d_fixed_velocity":
            self.robot = DubinsRobotFixedVelocity(self.robot_params)
        else:
            raise NotImplementedError(f"Robot type {rtype} not implemented")

    def _init_spaces(self):
        """Defines Gym action and observation spaces."""
        self.action_space = spaces.Box(
            low=self.robot_params.action_min,
            high=self.robot_params.action_max,
            shape=(self.robot_params.action_dim,),
        )

        x_lim, y_lim = self.env_params.world_x_size, self.env_params.world_y_size

        self.observation_space = spaces.Dict(
            {
                "location": spaces.Box(
                    low=np.array([0, 0]), high=np.array([x_lim, y_lim]), shape=(2,)
                ),
                "angle": spaces.Box(low=0, high=2 * np.pi, shape=(1,)),
                "smoke_density": spaces.Sequence(spaces.Box(low=0, high=1, shape=(1,)), stack=True),
                "smoke_density_location": spaces.Sequence(
                    spaces.Box(low=np.array([0, 0]), high=np.array([x_lim, y_lim]), shape=(2,)),
                    stack=True,
                ),
            }
        )

        if self.robot_params.robot_type == "unicycle":
            self.observation_space["velocity"] = spaces.Box(
                low=self.robot_params.state_min[3], high=self.robot_params.state_max[3], shape=(1,)
            )

    def reset(self, initial_state=None, seed=None, options=None):
        super().reset(seed=seed)

        self.window = {"fig": None, "ax": None, "cax": None}
        self.current_step = 0
        if initial_state is None:
            obs = self.observation_space.sample()
        else:
            obs = initial_state

        if isinstance(self.smoke_simulator, Playback):
            # Support selecting specific episode via seed or options if desired,
            # otherwise Playback cycles automatically.
            episode_idx = None
            if options and "episode_idx" in options:
                episode_idx = options["episode_idx"]
            elif seed is not None:
                episode_idx = seed
            self.smoke_simulator.reset(episode_idx=episode_idx)
        else:
            self.smoke_simulator.reset()

        if self.robot_params.robot_type == "unicycle":
            # Safety checks for scalar vs array inputs
            loc_x = np.ravel(obs["location"])[0]
            loc_y = np.ravel(obs["location"])[1]
            angle = np.ravel(obs["angle"])[0]
            v = np.ravel(obs["velocity"])[0] if "velocity" in obs else 0.0
            
            self.robot.reset(np.array([loc_x, loc_y, angle, v]))
            
        elif self.robot_params.robot_type == "dubins2d":
            loc_x = np.ravel(obs["location"])[0]
            loc_y = np.ravel(obs["location"])[1]
            angle = np.ravel(obs["angle"])[0]
            
            self.robot.reset(np.array([loc_x, loc_y, angle]))
            
        elif self.robot_params.robot_type == "dubins2d_fixed_velocity":
            loc_x = np.ravel(obs["location"])[0]
            loc_y = np.ravel(obs["location"])[1]
            angle = np.ravel(obs["angle"])[0]
            
            self.robot.reset(np.array([loc_x, loc_y, angle]))
        else:
            raise NotImplementedError(f"Robot type {self.robot_params.robot_type} not implemented")

        return self._get_obs(), {}

    def get_robot_odom(self):
        if self.robot_params.robot_type == "unicycle":
            return {
                "location": self.robot.get_state()[:2],
                "angle": self.robot.get_state()[2],
                "velocity": self.robot.get_state()[3],
            }
        elif self.robot_params.robot_type == "dubins2d":
            return {"location": self.robot.get_state()[:2], "angle": self.robot.get_state()[2]}
        elif self.robot_params.robot_type == "dubins2d_fixed_velocity":
            return {"location": self.robot.get_state()[:2], "angle": self.robot.get_state()[2]}
        else:
            raise NotImplementedError(f"Robot type {self.robot_params.robot_type} not implemented")

    def get_smoke_density_sensor(self, pos: np.ndarray, return_location: bool = True):
        assert self.sensor is not None, "Sensor must have been initialized"

        sensor_output = self.sensor.read(self.smoke_simulator.get_smoke_density, curr_pos=pos)
        if return_location:
            return sensor_output.readings, sensor_output.positions
        return sensor_output.readings

    def get_smoke_density_in_robot(self) -> float:
        odom = self.get_robot_odom()
        pos_x, pos_y = odom["location"]
        smoke_density_in_robot = self.smoke_simulator.get_smoke_density(np.array([pos_x, pos_y]))
        return smoke_density_in_robot

    def get_future_smoke_density(self, pos: np.ndarray, relative_step: int):
        """Returns smoke density at a specific position relative_step in the future."""
        if hasattr(self.smoke_simulator, "get_future_smoke_density"):
            return self.smoke_simulator.get_future_smoke_density(pos, relative_step)
        # For non-playback/dynamic simulation, we might not have future sight
        # in a real scenario, but for distillation we assume it exists in playback.
        return self.smoke_simulator.get_smoke_density(pos)

    def get_local_smoke_map(self, size_meters: float = None):
        """Returns a high-resolution local crop of the smoke map centered at the robot."""
        odom = self.get_robot_odom()
        pos_x, pos_y = odom["location"]
        
        if size_meters is None:
            # Default to v_max * 1.5s horizon
            v_max = self.robot_params.action_max[0]
            # Hardcoded 1.5s for now as discussed, or pull from config if available
            size_meters = v_max * 1.5 * 2.0 # Full width is 2x radius

        res = self.smoke_simulator.resolution
        half_size_px = int((size_meters / 2) / res)
        
        # Grid indices
        center_x_idx = int(pos_x / res)
        center_y_idx = int(pos_y / res)
        
        full_map = self.smoke_simulator.get_smoke_map()
        H, W = full_map.shape
        
        # Calculate bounds with padding
        y_min = max(0, center_y_idx - half_size_px)
        y_max = min(H, center_y_idx + half_size_px)
        x_min = max(0, center_x_idx - half_size_px)
        x_max = min(W, center_x_idx + half_size_px)
        
        crop = full_map[y_min:y_max, x_min:x_max]
        
        # Ensure fixed output size by padding if near boundaries
        expected_dim = 2 * half_size_px
        if crop.shape[0] < expected_dim or crop.shape[1] < expected_dim:
            padded_crop = np.zeros((expected_dim, expected_dim), dtype=np.float32)
            # Offset in padded_crop
            start_y = max(0, half_size_px - center_y_idx)
            start_x = max(0, half_size_px - center_x_idx)
            
            # Clip crop dimensions if they exceed padded_crop (shouldn't happen but safe)
            h_c, w_c = crop.shape
            padded_crop[start_y:start_y+h_c, start_x:start_x+w_c] = crop
            return padded_crop
            
        return crop

    def _get_obs(self):
        odom = self.get_robot_odom()
        pos_x, pos_y = odom["location"]
        angle = odom["angle"]

        smoke_density, smoke_density_location = self.get_smoke_density_sensor(
            np.array([pos_x, pos_y, angle])
        )

        obs = {
            "location": np.array([pos_x, pos_y]),
            "angle": angle,
            "smoke_density": smoke_density,
            "smoke_density_location": smoke_density_location,
        }

        if self.robot_params.robot_type == "unicycle":
            obs["velocity"] = odom["velocity"]
        return obs

    def _get_info(self):
        return {}

    def _get_reward(self, obs, action):
        pos_x, pos_y = obs["location"]
        if self.env_params.goal_location is not None:
            if (
                np.linalg.norm(np.array([pos_x, pos_y]) - self.env_params.goal_location)
                < self.env_params.goal_radius
            ):
                return 1.0
        # TODO: Add reward if reached goal
        return 0.0

    def _get_terminated(self, obs):
        pos_x, pos_y = obs["location"]
        smoke_density_in_robot = self.smoke_simulator.get_smoke_density(np.array([pos_x, pos_y]))

        if (
            self.env_params.smoke_density_threshold is not None
            and smoke_density_in_robot > self.env_params.smoke_density_threshold
        ):
            return True

        if self.env_params.goal_location is not None:
            if (
                np.linalg.norm(np.array([pos_x, pos_y]) - self.env_params.goal_location)
                < self.env_params.goal_radius
            ):
                return True
        return False

    def _get_truncated(self, obs):
        if self.current_step >= self.env_params.max_steps:
            return True
        return False

    def step(self, action):
        self.current_step += 1
        self.robot.dynamic_step(action)

        self.smoke_simulator.step()

        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        terminated = self._get_terminated(obs)
        truncated = self._get_truncated(obs)
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _init_render_window(self, fig: plt.Figure, ax: plt.Axes):
        if fig is not None and ax is not None:
            self.window["fig"] = fig
            self.window["ax"] = ax
        else:
            self.window["fig"], self.window["ax"] = plt.subplots(figsize=(8, 6))

        self.window["cax"] = self.window["ax"].imshow(
            self.smoke_simulator.get_smoke_map(),
            cmap="gray",
            extent=self.smoke_simulator.get_smoke_extent(),
            origin="lower",
            zorder=1,
        )

        self.window["ax"].set_title("Simulation")
        self.window["cax"].set_clim(vmin=0.0, vmax=1.0)
        self.window["ax"].set_xlim(0, self.env_params.world_x_size)
        self.window["ax"].set_ylim(0, self.env_params.world_y_size)

        self.window["ax"].set_xticks([])
        self.window["ax"].set_yticks([])

        if self.env_params.goal_location is not None:
            self.window["goal_circle"] = Circle(
                (self.env_params.goal_location[0], self.env_params.goal_location[1]),
                radius=self.env_params.goal_radius,
                color="g",
                fill=True,
                alpha=0.6,
                zorder=5,
            )
            self.window["ax"].add_patch(self.window["goal_circle"])

            x0, y0 = self.window["goal_circle"].center
            r = self.window["goal_circle"].radius
            self.window["goal_text"] = self.window["ax"].text(
                x0,
                y0 - r - 1.5,
                "goal",
                ha="center",
                va="bottom",
                fontsize=10,
                color="green",
                zorder=10,
            )

    def _render_controller_rollouts(self, controller):
        # Clear old dynamic paths (MPPI lines)
        for line in self.window["ax"].lines:
            line.remove()

        # Draw MPPI paths if provided (zorder=3)
        if controller is not None and hasattr(controller, "visualize_rollouts"):
            controller.visualize_rollouts(self.window["ax"])
            # visualize_rollouts creates Line2D objects
            for line in self.window["ax"].lines:
                line.set_zorder(3)

    def _render_robot_and_sensors(self):
        # Clear and redraw robot/sensor patches (zorder=4)
        for patch in list(self.window["ax"].patches):
            if isinstance(patch, (FancyArrow, Arrow, Polygon, Wedge)):
                patch.remove()

        if self.env_params.sensor_params.sensor_type == "downwards":
            odom = self.get_robot_odom()
            pos_x, pos_y = odom["location"]
            square = self.sensor.projection_bounds(pos_x, pos_y)
            bounded_square = np.array(
                [
                    clip_world(
                        p[0], p[1], self.env_params.world_x_size, self.env_params.world_y_size
                    )
                    for p in square
                ]
            )
            self.window["ax"].add_patch(
                Polygon(bounded_square, facecolor="none", edgecolor="blue", linewidth=2, zorder=4)
            )

        elif self.env_params.sensor_params.sensor_type == "camera_1d":
            odom = self.get_robot_odom()
            pos_x, pos_y = odom["location"]
            angle_rad = odom["angle"]
            angle_deg = np.rad2deg(angle_rad)

            fov_deg = self.env_params.sensor_params.fov_size_degrees
            max_range = self.env_params.sensor_params.max_range
            num_rays = self.env_params.sensor_params.num_rays

            # Use current robot state to get real-time sensor readings for visualization
            readings, _ = self.get_smoke_density_sensor(
                np.array([pos_x, pos_y, angle_rad]), return_location=True
            )

            # angular step for each ray
            d_theta = fov_deg / num_rays
            start_angle = angle_deg - fov_deg / 2

            # Draw individual arcs for each ray/pixel
            for i in range(num_rays):
                val = float(readings[i])
                # Map value to color (e.g., using 'hot' or 'plasma' colormap)
                # High density -> Brighter/Hotter color
                color = plt.cm.hot(val / self.env_params.sensor_params.opacity_threshold)

                wedge = Wedge(
                    (pos_x, pos_y),
                    max_range,
                    start_angle + i * d_theta,
                    start_angle + (i + 1) * d_theta,
                    facecolor=color,
                    edgecolor="white",  # "none",
                    alpha=0.8,
                    zorder=3,
                )
                self.window["ax"].add_patch(wedge)

        odom = self.get_robot_odom()
        pos_x, pos_y = odom["location"]
        angle = odom["angle"]
        self.window["ax"].arrow(
            pos_x,
            pos_y,
            0.1 * np.cos(angle),
            0.1 * np.sin(angle),
            head_width=0.8,
            head_length=0.8,
            fc="b",
            ec="b",
            zorder=4,
        )

    def _get_render_output(self):
        self.window["fig"].canvas.draw()
        if self.env_params.render == RenderMode.HUMAN:
            self.window["fig"].canvas.flush_events()
            plt.pause(self.clock)
        elif self.env_params.render == RenderMode.RGB_ARRAY:
            width, height = self.window["fig"].canvas.get_width_height()
            try:
                rgba = np.asarray(self.window["fig"].canvas.buffer_rgba())
                img = rgba[..., :3]
            except AttributeError:
                img = np.frombuffer(self.window["fig"].canvas.tostring_rgb(), dtype="uint8")
                img = img.reshape(height, width, 3)
            return img
        return None

    def _render_frame(self, fig: plt.Figure = None, ax: plt.Axes = None, controller=None):
        if self.env_params.render and self.env_params.render not in [
            RenderMode.HUMAN,
            RenderMode.RGB_ARRAY,
        ]:
            return None

        if self.window["fig"] is None:
            self._init_render_window(fig, ax)

        # Update smoke background
        self.window["cax"].set_array(self.smoke_simulator.get_smoke_map())

        self._render_controller_rollouts(controller)
        self._render_robot_and_sensors()

        return self._get_render_output()

    def close(self):
        self.window = {"fig": None, "ax": None, "cax": None}


if __name__ == "__main__":
    # Example loading via OmegaConf (Hydra-style)
    config_path = "configs/env/smoke_env.yaml"
    cfg = OmegaConf.load(config_path)

    # Test initialization
    env = SmokeEnv(cfg=cfg)

    # Overwrite render to always show visually in testing script
    env.env_params.render = RenderMode.HUMAN

    initial_state = {
        "location": np.array([5, 5]),
        "angle": 1.0,
        "smoke_density": 0.0,
        "velocity": 1.0,
    }
    env.reset(initial_state=initial_state)

    fig, ax = plt.subplots(figsize=(6, 6))

    for _ in range(50):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(np.round(state["smoke_density"], 1))
        env._render_frame(fig=fig, ax=ax)

    env.close()
