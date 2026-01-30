import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator.dynamic_smoke import DynamicSmoke, SmokeBlobParams, DynamicSmokeParams
from simulator.playback_smoke import PlaybackSmoke, PlaybackSmokeParams
from simulator.sensor import GlobalSensorParams, PointSensorParams, DownwardsSensorParams, BaseSensorParams, DownwardsSensor, PointSensor, GlobalSensor
from agents.basic_robot import RobotParams
from matplotlib.collections import PathCollection

from agents.dubins_robot import DubinsRobot
from agents.dubins_robot_fixed_velocity import DubinsRobotFixedVelocity
from agents.unicycle_robot import UnicycleRobot
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.patches import FancyArrow, Arrow
from src.utils import *
import yaml
from io import BytesIO
import imageio
import time

@dataclass
class EnvParams:
    world_x_size: int = field(default=50)
    world_y_size: int = field(default=50)

    max_steps: int = field(default=1000)
    render: str = None
    clock: float = field(default=0.1)

    goal_location: tuple[int, int] | None = field(default=None)
    goal_radius: float = field(default=1.0)

    smoke_density_threshold: float = None
    sensor_params: BaseSensorParams | DownwardsSensorParams | PointSensorParams | GlobalSensorParams | None = field(default=None)
    playback_path: str = "/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/data/global_source_400_100_2nd.npz" # field(default=None)

    @staticmethod
    def load_from_yaml(file_path: str) -> "EnvParams":
        """Load environment parameters from a YAML config file."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return EnvParams(**data)
        
class DynamicSmokeEnv(gym.Env):
    def __init__(self, env_params: EnvParams, robot_params: RobotParams, smoke_params: DynamicSmokeParams) -> None:
        super().__init__()

        self.env_params = env_params
        self.robot_params = robot_params

        if self.env_params.playback_path:
            playback_params = PlaybackSmokeParams(data_path=self.env_params.playback_path)
            self.smoke_simulator = PlaybackSmoke(params=playback_params)
        else:
            self.smoke_simulator = DynamicSmoke(params=smoke_params)

        self.robot_params.state_max[0] = self.env_params.world_x_size
        self.robot_params.state_max[1] = self.env_params.world_y_size

        self.action_space = spaces.Box(low=self.robot_params.action_min,
                                    high=self.robot_params.action_max,
                                    shape=(self.robot_params.action_dim,))
        
        # 0: x, 1: y, 2: theta, 3: smoke_density
        # TODO: Check why world - 1?
        self.observation_space = spaces.Dict({
            "location": spaces.Box(low=np.array([0, 0]),
                                   high=np.array([self.env_params.world_x_size - 1, self.env_params.world_y_size - 1]),
                                   shape=(2,)),
            "angle": spaces.Box(low=0, high=2*np.pi, shape=(1,)),
            "smoke_density": spaces.Sequence(spaces.Box(low=0, high=1, shape=(1,)), stack=True),
            "smoke_density_location": spaces.Sequence(spaces.Box(low=np.array([0, 0]), high=np.array([self.env_params.world_x_size - 1, self.env_params.world_y_size - 1]), shape=(2,)), stack=True)
        })

        if self.robot_params.robot_type == "unicycle":
            self.observation_space["velocity"] = spaces.Box(low=self.robot_params.state_min[3], high=self.robot_params.state_max[3], shape=(1,))
        elif self.robot_params.robot_type == "dubins2d":
            pass
        elif self.robot_params.robot_type == "dubins2d_fixed_velocity":
            pass
        else:
            raise NotImplementedError(f"Robot type {self.robot_params.robot_type} not implemented")

        if self.env_params.sensor_params is not None:
            if self.env_params.sensor_params.sensor_type == "downwards":
                self.sensor = DownwardsSensor(self.env_params.sensor_params)
            elif self.env_params.sensor_params.sensor_type == "point":
                self.sensor = PointSensor(self.env_params.sensor_params)
            elif self.env_params.sensor_params.sensor_type == "global":
                self.sensor = GlobalSensor(self.env_params.sensor_params)
            else:
                raise ValueError(f"Sensor type {self.env_params.sensor_params.sensor_type} not supported")

        if robot_params.robot_type == "unicycle":
            self.robot = UnicycleRobot(robot_params)
        elif robot_params.robot_type == "dubins2d":
            self.robot = DubinsRobot(robot_params)
        elif robot_params.robot_type == "dubins2d_fixed_velocity":
            self.robot = DubinsRobotFixedVelocity(robot_params)
        else:
            raise NotImplementedError(f"Robot type {robot_params.robot_type} not implemented")
    
        self.window = {"fig": None, "ax": None, "cax": None}
        self.clock = self.env_params.clock
        self.current_step = 0

    def reset(self, initial_state=None, seed=None, options=None):
        super().reset(seed=seed)

        self.window = {"fig": None, "ax": None, "cax": None}
        if initial_state is None:
            obs = self.observation_space.sample()
        else:
            obs = initial_state

        if isinstance(self.smoke_simulator, PlaybackSmoke):
            # Support selecting specific episode via seed or options if desired, 
            # otherwise PlaybackSmoke cycles automatically.
            episode_idx = None
            if options and 'episode_idx' in options:
                episode_idx = options['episode_idx']
            elif seed is not None:
                episode_idx = seed
            self.smoke_simulator.reset(episode_idx=episode_idx)
        else:
            self.smoke_simulator.reset()

        if self.robot_params.robot_type == "unicycle":
            self.robot.reset(np.array([obs["location"][0], obs["location"][1], obs["angle"], obs["velocity"][0]]))
        elif self.robot_params.robot_type == "dubins2d":
            self.robot.reset(np.array([obs["location"][0], obs["location"][1], obs["angle"]]))
        elif self.robot_params.robot_type == "dubins2d_fixed_velocity":
            self.robot.reset(np.array([obs["location"][0], obs["location"][1], obs["angle"]]))
        else:
            raise NotImplementedError(f"Robot type {self.robot_params.robot_type} not implemented")

        return self._get_obs(), {}

    def get_robot_odom(self):
        if self.robot_params.robot_type == "unicycle":
            return {"location": self.robot.get_state()[:2],
                    "angle": self.robot.get_state()[2],
                    "velocity": self.robot.get_state()[3]}
        elif self.robot_params.robot_type == "dubins2d":
            return {"location": self.robot.get_state()[:2],
                    "angle": self.robot.get_state()[2]}
        elif self.robot_params.robot_type == "dubins2d_fixed_velocity":
            return {"location": self.robot.get_state()[:2],
                    "angle": self.robot.get_state()[2]}
        else:
            raise NotImplementedError(f"Robot type {self.robot_params.robot_type} not implemented")

    def get_smoke_density_sensor(self, pos: np.ndarray, return_location: bool = True):
        assert self.sensor is not None, "Sensor must have been initialized"

        sensor_output = self.sensor.read(self.smoke_simulator.get_smoke_density, curr_pos=pos)
        if return_location:
            return sensor_output["sensor_readings"], sensor_output["sensor_position_readings"]
        return sensor_output["sensor_readings"]

    def get_smoke_density_in_robot(self) -> float:
        odom = self.get_robot_odom()
        pos_x, pos_y = odom["location"]
        smoke_density_in_robot = self.smoke_simulator.get_smoke_density(np.array([pos_x, pos_y]))
        return smoke_density_in_robot
        
    def _get_obs(self):
        odom = self.get_robot_odom()
        pos_x, pos_y = odom["location"]
        angle = odom["angle"]

        smoke_density, smoke_density_location = self.get_smoke_density_sensor(np.array([pos_x, pos_y]))

        obs = {"location": np.array([pos_x, pos_y]),
                "angle": angle,
                "smoke_density": smoke_density,
                "smoke_density_location": smoke_density_location}
            
        if self.robot_params.robot_type == "unicycle":
            obs["velocity"] = odom["velocity"]
        return obs

    def _get_info(self):
        return {}

    def _get_reward(self, obs, action):
        pos_x, pos_y = obs["location"]
        if self.env_params.goal_location is not None:
            if np.linalg.norm(np.array([pos_x, pos_y]) - self.env_params.goal_location) < self.env_params.goal_radius:
                return 1.
        # TODO: Add reward if reached goal
        return 0.

    def _get_terminated(self, obs):
        pos_x, pos_y = obs["location"]
        smoke_density_in_robot = self.smoke_simulator.get_smoke_density(np.array([pos_x, pos_y]))

        if self.env_params.smoke_density_threshold is not None and smoke_density_in_robot > self.env_params.smoke_density_threshold:
            return True

        if self.env_params.goal_location is not None:
            if np.linalg.norm(np.array([pos_x, pos_y]) - self.env_params.goal_location) < self.env_params.goal_radius:
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
    
    def _render_frame(self, fig: plt.Figure = None, ax: plt.Axes = None):
        if self.env_params.render and self.env_params.render not in ["human", "rgb_array"]:
            return

        if self.window["fig"] is None:
            if fig is not None and ax is not None:
                self.window["fig"] = fig
                self.window["ax"] = ax
            else:
                self.window["fig"], self.window["ax"] = plt.subplots()

            self.window["cax"] = self.window["ax"].imshow(
                self.smoke_simulator.get_smoke_map(),
                cmap='gray',
                extent=self.smoke_simulator.get_smoke_extent(),
                origin='lower'
            )

            # self.window["ax"].set_axis_off()
            # self.window["fig"].colorbar(self.window["cax"], ax=self.window["ax"], label="Smoke Density", shrink=0.5)
            
            self.window["ax"].set_title("Simulation")
            self.window["cax"].set_clim(vmin=np.min(0.0),
                                        vmax=np.max(1.0))
            self.window["ax"].set_xlim(0, self.env_params.world_x_size)
            self.window["ax"].set_ylim(0, self.env_params.world_y_size)

            self.window["ax"].set_xticks([])
            self.window["ax"].set_yticks([])

            if self.env_params.goal_location is not None:
                circle = Circle((self.env_params.goal_location[0], self.env_params.goal_location[1]), 
                              radius=self.env_params.goal_radius, color='g', fill=True, alpha=0.8)
                self.window["ax"].add_patch(circle)
                x0, y0 = circle.center
                r = circle.radius
                self.window["ax"].text(x0, y0 - r - 1.5, "goal", ha="center", va="bottom",
                        fontsize=10, color="green", zorder=10)
        
        self.window["cax"].set_array(self.smoke_simulator.get_smoke_map())

        for arrow in self.window["ax"].patches:
            if isinstance(arrow, (FancyArrow, Arrow)):  
                arrow.remove()

        if self.env_params.sensor_params.sensor_type == "downwards":
            for patch in self.window["ax"].patches:
                if isinstance(patch, Polygon):
                    patch.remove()

            odom = self.get_robot_odom()
            pos_x, pos_y = odom["location"]

            square = self.sensor.projection_bounds(pos_x, pos_y)

            bounded_square = np.array([clip_world(p[0], p[1], self.env_params.world_x_size, self.env_params.world_y_size) for p in square])
            self.window["ax"].add_patch(
                Polygon(bounded_square, 
                facecolor='none',     # no fill
                edgecolor='blue',    # border color
                linewidth=2,), 
            )

        odom = self.get_robot_odom()
        pos_x, pos_y = odom["location"]
        angle = odom["angle"]
        self.window["ax"].arrow(pos_x, pos_y, 0.1*np.cos(angle), 0.1*np.sin(angle), 
                                head_width=0.8, head_length=0.8, fc='b', ec='b')
        
        if self.env_params.sensor_params.sensor_type == "downwards" or self.env_params.sensor_params.sensor_type == "global":
            for coll in list(self.window["ax"].collections):
                if isinstance(coll, PathCollection):
                    coll.remove()

            _, smoke_density_location = self.get_smoke_density_sensor(np.array([pos_x, pos_y]))
            self.window["ax"].scatter(smoke_density_location[:, 0], smoke_density_location[:, 1], color='red', s=0.1)

        self.window["fig"].canvas.draw()
        
        if self.env_params.render == "human":
            self.window["fig"].canvas.flush_events()
            plt.pause(self.clock)

    def close(self):
        self.window = {"fig": None, "ax": None, "cax": None}

if __name__ == "__main__":
    # env_params = EnvParams.load_from_yaml("config/env/smoke_env.yaml")
    env_params = EnvParams()
    robot_params = RobotParams()

    # robot_params = RobotParams.load_from_yaml("agents/dubins2d_fixed_velocity_cfg.yaml")
    world_x_size = 60
    world_y_size = 50
    env_params.world_x_size = world_x_size
    env_params.world_y_size = world_y_size
    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=1.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=20, y_pos=20, intensity=1.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=15, y_pos=45, intensity=1.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=40, y_pos=40, intensity=1.0, spread_rate=8.0)
    ]
    sensor_params = GlobalSensorParams(world_x_size=world_x_size, world_y_size=world_y_size)
    env_params.sensor_params = sensor_params
    smoke_params = DynamicSmokeParams(x_size=world_x_size, y_size=world_y_size, smoke_blob_params=smoke_blob_params, resolution=0.3)

    env = DynamicSmokeEnv(env_params, robot_params, smoke_params)
    initial_state = {"location": np.array([45, 15]), "angle": 0, "smoke_density": 0, "velocity": 4.0}
    env.reset(initial_state=initial_state)

    fig, ax = plt.subplots(figsize=(4, 4))

    frames = []
    for _ in range(100):
        action = env.action_space.sample()
        print(action)
        state, reward, terminated, truncated, info = env.step(action)
        print(20*"-")
        env._render_frame(fig=fig, ax=ax)
        plt.pause(0.1)

    env.close()
