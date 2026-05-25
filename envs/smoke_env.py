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
    SensorOutput,
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
    goal_locations: list[tuple[float, float]] | None = field(default=None)
    goal_radius: float = field(default=1.0)

    smoke_density_threshold: float = None
    sensor_params: Any = field(default=None)
    playback_path: str | None = None
    num_agents: int = field(default=1)
    collision_radius: float = field(default=0.8)
    terminate_on_collision: bool = field(default=True)
    collision_penalty: float = field(default=-10.0)


class SmokeAgent:
    """Defines an individual agent's robot, goal, and observation/action spaces."""

    def __init__(
        self,
        agent_id: int,
        robot_params: RobotParams,
        env_params: EnvParams,
        goal_location: tuple[float, float] | None,
    ):
        self.agent_id = agent_id
        self.robot_params = robot_params
        self.env_params = env_params
        self.goal_location = goal_location
        self.goal_radius = env_params.goal_radius

        # Robot Object creation
        rtype = robot_params.robot_type
        if rtype == "unicycle":
            self.robot = UnicycleRobot(robot_params)
        elif rtype == "dubins2d":
            self.robot = DubinsRobot(robot_params)
        elif rtype == "dubins2d_fixed_velocity":
            self.robot = DubinsRobotFixedVelocity(robot_params)
        else:
            raise NotImplementedError(f"Robot type {rtype} not implemented")

        self._init_spaces()

    def _init_spaces(self):
        """Defines Gym spaces for this individual agent."""
        self.action_space = spaces.Box(
            low=np.array(self.robot_params.action_min),
            high=np.array(self.robot_params.action_max),
            shape=(self.robot_params.action_dim,),
        )

        x_lim, y_lim = self.env_params.world_x_size, self.env_params.world_y_size

        self.observation_space = spaces.Dict(
            {
                "id": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
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

    def reset(self, initial_state=None):
        """Resets the individual agent's robot state."""
        if initial_state is None:
            obs = self.observation_space.sample()
        else:
            obs = initial_state

        if self.robot_params.robot_type == "unicycle":
            loc_x = np.ravel(obs["location"])[0]
            loc_y = np.ravel(obs["location"])[1]
            angle = np.ravel(obs["angle"])[0]
            v = np.ravel(obs["velocity"])[0] if "velocity" in obs else 0.0
            self.robot.reset(np.array([loc_x, loc_y, angle, v]))
        elif self.robot_params.robot_type in ["dubins2d", "dubins2d_fixed_velocity"]:
            loc_x = np.ravel(obs["location"])[0]
            loc_y = np.ravel(obs["location"])[1]
            angle = np.ravel(obs["angle"])[0]
            self.robot.reset(np.array([loc_x, loc_y, angle]))
        else:
            raise NotImplementedError(f"Robot type {self.robot_params.robot_type} not implemented")

    def get_robot_odom(self):
        """Retrieves robot odometry/state."""
        if self.robot_params.robot_type == "unicycle":
            return {
                "location": self.robot.get_state()[:2],
                "angle": self.robot.get_state()[2],
                "velocity": self.robot.get_state()[3],
            }
        elif self.robot_params.robot_type in ["dubins2d", "dubins2d_fixed_velocity"]:
            return {"location": self.robot.get_state()[:2], "angle": self.robot.get_state()[2]}
        else:
            raise NotImplementedError(f"Robot type {self.robot_params.robot_type} not implemented")


class SmokeEnv(gym.Env):
    def __init__(
        self,
        cfg: Union[str, DictConfig, dict] = "configs/env/smoke_env.yaml",
        robot_params: RobotParams | None = None,
    ) -> None:
        """Initializes the Smoke Environment supporting single or multiple agents."""
        super().__init__()

        # 1. Load/Parse the configuration
        if isinstance(cfg, str):
            cfg = OmegaConf.load(cfg)
        elif isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        # 2. Setup core environment parameters (including num_agents and goals assertions)
        self._setup_env_params(cfg)

        # 3. Setup sensor parameters
        self._setup_sensor_params(cfg)

        # 4. Setup smoke and robot parameters
        self._setup_smoke_and_robot(cfg, robot_params)

        # 5. Initialize the simulator and the agents
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
        self.env_params.num_agents = int(cfg.get("num_agents", defaults.num_agents))
        self.env_params.collision_radius = float(cfg.get("collision_radius", defaults.collision_radius))
        self.env_params.terminate_on_collision = bool(cfg.get("terminate_on_collision", defaults.terminate_on_collision))
        self.env_params.collision_penalty = float(cfg.get("collision_penalty", defaults.collision_penalty))

        # Setup goal locations (supporting both singular and plural formats)
        if cfg.get("goal_locations"):
            self.env_params.goal_locations = [tuple(loc) for loc in cfg["goal_locations"]]
        elif cfg.get("goal_location"):
            self.env_params.goal_locations = [tuple(cfg["goal_location"])]
            self.env_params.goal_location = tuple(cfg["goal_location"])
        else:
            self.env_params.goal_locations = None

        self.env_params.goal_radius = float(cfg.get("goal_radius", defaults.goal_radius))

        # Replicate a single goal if multiple agents but only one goal is specified
        if self.env_params.goal_locations is None:
            if self.env_params.goal_location is not None:
                self.env_params.goal_locations = [
                    self.env_params.goal_location
                ] * self.env_params.num_agents

        # Separation Assertion: assert goal numbers must equal agent numbers
        if self.env_params.goal_locations is not None:
            assert len(self.env_params.goal_locations) == self.env_params.num_agents, (
                f"Number of goal locations ({len(self.env_params.goal_locations)}) must match "
                f"the number of agents ({self.env_params.num_agents})."
            )

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
        """Choose between Playback or Dynamic simulation and create the agents."""
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

        # Create multi-agents
        self.agents = []
        for i in range(self.env_params.num_agents):
            goal = (
                self.env_params.goal_locations[i]
                if self.env_params.goal_locations is not None
                else None
            )
            agent = SmokeAgent(
                agent_id=i,
                robot_params=self.robot_params,
                env_params=self.env_params,
                goal_location=goal,
            )
            self.agents.append(agent)

        # Maintain direct reference to the first agent's robot for full single-agent backward compatibility
        self.robot = self.agents[0].robot

    def _init_spaces(self):
        """Defines Gym action and observation spaces conforming to composite gymnasium.spaces standards."""
        if self.env_params.num_agents == 1:
            self.action_space = self.agents[0].action_space
            self.observation_space = self.agents[0].observation_space
        else:
            self.action_space = spaces.Dict(
                {f"agent_{i}": agent.action_space for i, agent in enumerate(self.agents)}
            )
            self.observation_space = spaces.Dict(
                {f"agent_{i}": agent.observation_space for i, agent in enumerate(self.agents)}
            )

    def reset(self, initial_state=None, seed=None, options=None):
        super().reset(seed=seed)

        self.window = {"fig": None, "ax": None, "cax": None}
        self.current_step = 0

        if isinstance(self.smoke_simulator, Playback):
            episode_idx = None
            if options and "episode_idx" in options:
                episode_idx = options["episode_idx"]
            elif seed is not None:
                episode_idx = seed
            self.smoke_simulator.reset(episode_idx=episode_idx)
        else:
            self.smoke_simulator.reset()

        # Handle initial states for each agent (supports lists or single dictionaries)
        for i, agent in enumerate(self.agents):
            if initial_state is None:
                agent_init = None
            elif isinstance(initial_state, dict):
                agent_init = initial_state.get(
                    f"agent_{i}",
                    initial_state.get(
                        i, initial_state if self.env_params.num_agents == 1 else None
                    ),
                )
            elif isinstance(initial_state, list):
                agent_init = initial_state[i] if i < len(initial_state) else None
            else:
                agent_init = None

            agent.reset(initial_state=agent_init)

        return self._get_obs(), {}

    def get_robot_odom(self, agent_idx: int = 0):
        """Retrieves robot odometry for a specific agent index."""
        return self.agents[agent_idx].get_robot_odom()

    def get_smoke_density_sensor(self, pos: np.ndarray, return_location: bool = True):
        assert self.sensor is not None, "Sensor must have been initialized"

        sensor_output = self.sensor.read(self.smoke_simulator.get_smoke_density, curr_pos=pos)
        if return_location:
            return sensor_output.readings, sensor_output.positions
        return sensor_output.readings

    def get_smoke_density_in_robot(self, agent_idx: int = 0) -> float:
        odom = self.get_robot_odom(agent_idx)
        pos_x, pos_y = odom["location"]
        smoke_density_in_robot = self.smoke_simulator.get_smoke_density(np.array([pos_x, pos_y]))
        return smoke_density_in_robot

    def _get_obs_for_agent(
        self, agent: SmokeAgent, global_sensor_output: SensorOutput | None = None
    ):
        """Computes the observation dictionary for a single agent, reusing cached global readings if applicable."""
        odom = agent.get_robot_odom()
        pos_x, pos_y = odom["location"]
        angle = odom["angle"]

        if global_sensor_output is not None:
            # High efficiency: all agents point to the same global pre-computed arrays
            smoke_density = global_sensor_output.readings
            smoke_density_location = global_sensor_output.positions
        else:
            # Local FOV: each agent performs its own reading based on its specific position
            smoke_density, smoke_density_location = self.get_smoke_density_sensor(
                np.array([pos_x, pos_y, angle])
            )

        obs = {
            "id": np.array([agent.agent_id], dtype=np.int32),
            "location": np.array([pos_x, pos_y]),
            "angle": angle,
            "smoke_density": smoke_density,
            "smoke_density_location": smoke_density_location,
        }

        if agent.robot_params.robot_type == "unicycle":
            obs["velocity"] = odom["velocity"]
        return obs

    def _get_obs(self):
        """Constructs the centralized observation space."""
        global_sensor_output = None
        if self.env_params.sensor_params.sensor_type == "global":
            # High efficiency optimization: compute global readings only once per step for all agents
            dummy_pos = np.array([0.0, 0.0])
            global_sensor_output = self.sensor.read(
                self.smoke_simulator.get_smoke_density, curr_pos=dummy_pos
            )

        if self.env_params.num_agents == 1:
            return self._get_obs_for_agent(
                self.agents[0], global_sensor_output=global_sensor_output
            )

        return {
            f"agent_{i}": self._get_obs_for_agent(agent, global_sensor_output=global_sensor_output)
            for i, agent in enumerate(self.agents)
        }

    def _get_info(self):
        if self.env_params.num_agents == 1:
            return {}
        return {f"agent_{i}": {} for i in range(self.env_params.num_agents)}

    def _check_collision_for_agent(self, agent: SmokeAgent, pos: np.ndarray) -> bool:
        """Helper to check if a given agent position collides with any other agent."""
        if self.env_params.num_agents > 1:
            for other in self.agents:
                if other.agent_id != agent.agent_id:
                    other_pos = other.get_robot_odom()["location"]
                    dist = np.linalg.norm(pos - other_pos)
                    if dist < (self.env_params.collision_radius * 2):
                        return True
        return False

    def _get_reward_for_agent(self, agent: SmokeAgent, obs_agent):
        pos_x, pos_y = obs_agent["location"]
        reward = 0.0
        if agent.goal_location is not None:
            if np.linalg.norm(np.array([pos_x, pos_y]) - agent.goal_location) < agent.goal_radius:
                reward = 1.0

        # Check collision with other agents
        if self._check_collision_for_agent(agent, np.array([pos_x, pos_y])):
            reward += self.env_params.collision_penalty
        return reward

    def _get_reward(self, obs, action):
        if self.env_params.num_agents == 1:
            return self._get_reward_for_agent(self.agents[0], obs)
        else:
            return {
                f"agent_{i}": self._get_reward_for_agent(agent, obs[f"agent_{i}"])
                for i, agent in enumerate(self.agents)
            }

    def _get_terminated_for_agent(self, agent: SmokeAgent, obs_agent):
        pos_x, pos_y = obs_agent["location"]
        smoke_density_in_robot = self.smoke_simulator.get_smoke_density(np.array([pos_x, pos_y]))

        if (
            self.env_params.smoke_density_threshold is not None
            and smoke_density_in_robot > self.env_params.smoke_density_threshold
        ):
            return True

        if agent.goal_location is not None:
            if np.linalg.norm(np.array([pos_x, pos_y]) - agent.goal_location) < agent.goal_radius:
                return True

        # Optional collision termination
        if self.env_params.terminate_on_collision:
            if self._check_collision_for_agent(agent, np.array([pos_x, pos_y])):
                return True
        return False

    def _get_terminated(self, obs):
        if self.env_params.num_agents == 1:
            return self._get_terminated_for_agent(self.agents[0], obs)
        else:
            return {
                f"agent_{i}": self._get_terminated_for_agent(agent, obs[f"agent_{i}"])
                for i, agent in enumerate(self.agents)
            }

    def _get_truncated(self, obs):
        if self.current_step >= self.env_params.max_steps:
            if self.env_params.num_agents == 1:
                return True
            else:
                return {f"agent_{i}": True for i in range(self.env_params.num_agents)}
        if self.env_params.num_agents == 1:
            return False
        else:
            return {f"agent_{i}": False for i in range(self.env_params.num_agents)}

    def step(self, action):
        self.current_step += 1

        if self.env_params.num_agents == 1:
            # Support actions formatted as single action, list/tuple of size 1, or dictionaries
            if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 1:
                act = action[0]
            elif isinstance(action, dict) and 0 in action:
                act = action[0]
            elif isinstance(action, dict) and "agent_0" in action:
                act = action["agent_0"]
            else:
                act = action
            self.agents[0].robot.dynamic_step(act)
        else:
            for i, agent in enumerate(self.agents):
                if isinstance(action, dict):
                    act = action.get(f"agent_{i}", action.get(i))
                elif isinstance(action, (list, tuple, np.ndarray)):
                    act = action[i]
                else:
                    raise ValueError(
                        f"Action format not recognized for multi-agent stepping: {type(action)}"
                    )
                agent.robot.dynamic_step(act)

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

        self.window["goal_circles"] = []
        self.window["goal_texts"] = []
        for i, agent in enumerate(self.agents):
            if agent.goal_location is not None:
                circle = Circle(
                    (agent.goal_location[0], agent.goal_location[1]),
                    radius=agent.goal_radius,
                    color="g",
                    fill=True,
                    alpha=0.6,
                    zorder=5,
                )
                self.window["ax"].add_patch(circle)
                self.window["goal_circles"].append(circle)

                x0, y0 = circle.center
                r = circle.radius
                text = self.window["ax"].text(
                    x0,
                    y0 - r - 1.5,
                    f"goal_{i}" if self.env_params.num_agents > 1 else "goal",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="green",
                    zorder=10,
                )
                self.window["goal_texts"].append(text)

    def _render_controller_rollouts(self, controller):
        # Clear old dynamic paths (MPPI lines)
        for line in self.window["ax"].lines:
            line.remove()

        # Draw MPPI paths if provided (zorder=3)
        if controller is not None and hasattr(controller, "visualize_rollouts"):
            controller.visualize_rollouts(self.window["ax"])
            for line in self.window["ax"].lines:
                line.set_zorder(3)

    def _render_robot_and_sensors(self, controller=None):
        # Clear and redraw robot/sensor patches (zorder=4)
        # Protect the static goal circles from removal
        goal_circles = self.window.get("goal_circles", [])
        for patch in list(self.window["ax"].patches):
            if isinstance(patch, (FancyArrow, Arrow, Polygon, Wedge, Circle)):
                if patch not in goal_circles:
                    patch.remove()

        for i, agent in enumerate(self.agents):
            odom = agent.get_robot_odom()
            pos_x, pos_y = odom["location"]
            angle_rad = odom["angle"]
            angle_deg = np.rad2deg(angle_rad)

            if self.env_params.sensor_params.sensor_type == "downwards":
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
                    Polygon(
                        bounded_square, facecolor="none", edgecolor="blue", linewidth=2, zorder=4
                    )
                )

            elif self.env_params.sensor_params.sensor_type == "camera_1d":
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
                for r_idx in range(num_rays):
                    val = float(readings[r_idx])
                    color = plt.cm.hot(val / self.env_params.sensor_params.opacity_threshold)

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
                    self.window["ax"].add_patch(wedge)

            # 1. Draw cleaner orientation directional arrow
            self.window["ax"].arrow(
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

            # 2. Render Physical Collision Boundary (Solid Dark Red circle)
            col_radius = self.env_params.collision_radius
            circ_phys = Circle(
                (pos_x, pos_y), 
                radius=col_radius, 
                facecolor="none", 
                edgecolor="#8B0000", 
                linestyle="-", 
                linewidth=1.2, 
                alpha=0.7, 
                zorder=4
            )
            self.window["ax"].add_patch(circ_phys)

            # 3. Render Control Safety Radius if provided by the experiment controller (Dotted Blue circle)
            if controller is not None and hasattr(controller, "d_safe"):
                safe_rad = float(controller.d_safe) / 2.0  # Represents radius per robot
                circ_safe = Circle(
                    (pos_x, pos_y), 
                    radius=safe_rad, 
                    facecolor="none", 
                    edgecolor="#0000FF", 
                    linestyle="--", 
                    linewidth=1.0, 
                    alpha=0.5, 
                    zorder=4
                )
                self.window["ax"].add_patch(circ_safe)

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
        self._render_robot_and_sensors(controller)

        return self._get_render_output()

    def render(self):
        """Standard Gym/Gymnasium render method."""
        return self._render_frame()

    def close(self):
        self.window = {"fig": None, "ax": None, "cax": None}


if __name__ == "__main__":
    # Example loading via OmegaConf (Hydra-style)
    config_path = "configs/env/smoke_env.yaml"
    cfg = OmegaConf.load(config_path)

    # Test single agent initialization
    print("Testing Single Agent Environment Initialization...")
    env = SmokeEnv(cfg=cfg)
    env.env_params.render = RenderMode.HUMAN

    initial_state = {
        "location": np.array([5, 5]),
        "angle": 1.0,
        "smoke_density": 0.0,
        "velocity": 1.0,
    }
    obs, _ = env.reset(initial_state=initial_state)
    print(f"Single-agent Reset Obs ID: {obs['id']}")
    print(f"Single-agent Reset Obs Location: {obs['location']}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Single-agent Step Reward: {reward}")
    env.close()

    # Test multi-agent initialization with Dict composite spaces
    print(
        "\nTesting Multi-Agent Environment (num_agents = 3) Initialization with Composite Spaces..."
    )
    multi_cfg = OmegaConf.load(config_path)
    multi_cfg["num_agents"] = 3
    multi_cfg["goal_locations"] = [[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]]

    env_multi = SmokeEnv(cfg=multi_cfg)
    env_multi.env_params.render = RenderMode.HUMAN

    obs_multi, _ = env_multi.reset()
    print(f"Multi-agent Reset Obs Keys: {list(obs_multi.keys())}")
    for k, o in obs_multi.items():
        print(f"Agent {k} ID: {o['id']} | Location: {o['location']}")

    actions = env_multi.action_space.sample()
    obs_multi, rewards, terminateds, truncateds, infos = env_multi.step(actions)
    print(f"Multi-agent Step Rewards Keys: {list(rewards.keys())}")
    print(f"Multi-agent Step Rewards: {rewards}")

    for _ in range(100):
        actions = env_multi.action_space.sample()
        obs_multi, rewards, terminateds, truncateds, infos = env_multi.step(actions)
        env_multi.render()

    env_multi.close()
    print("\nAll tests executed successfully!")
