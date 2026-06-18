from enum import Enum
from typing import Any, Optional, Union

import gymnasium as gym
import hydra
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from agents.basic_robot import RobotParams
from agents.dubins_robot import DubinsRobot

# from agents.dubins_robot_fixed_velocity import DubinsRobotFixedVelocity
# from agents.unicycle_robot import UnicycleRobot
from env.simulator.playback import Playback, PlaybackParams
from env.simulator.sensor import (
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
from env.simulator.smoke import BlobParams, Smoke, SmokeParams
from visualization import BaseRenderer, SimpleRenderer


class RenderMode(str, Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    NONE = "none"


class SensorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "global"
    num_rays: Optional[int] = None
    fov_size_degrees: Optional[float] = None
    step_size: Optional[float] = None
    opacity_threshold: Optional[float] = None
    max_range: Optional[float] = None


class BlobConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    intensity: float
    spread: float


class SmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolution: float = 1.0
    blobs: list[BlobConfig] = Field(default_factory=list)


class BaseEnvConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    world_x_size: float = 100.0
    world_y_size: float = 100.0
    max_steps: int = 200
    clock: float = 0.1
    render: RenderMode = RenderMode.HUMAN
    render_save_every: int = 2
    playback_path: Optional[str] = None
    goal_radius: float = 1.0
    num_agents: int = 1
    collision_radius: float = 0.8
    terminate_on_collision: bool = True
    collision_penalty: float = -10.0
    smoke_density_threshold: Optional[float] = None


class FullEnvConfig(BaseEnvConfig):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = "DynamicSmokeEnv"
    test: bool = False
    num_episodes: int = 200

    goal_location: Optional[list[float]] = None
    goal_locations: Optional[list[list[float]]] = None
    initial_locations: Optional[list[list[float]]] = None

    sensor: SensorConfig = Field(default_factory=SensorConfig)
    robot: RobotParams = Field(default_factory=RobotParams)
    smoke: Optional[SmokeConfig] = None
    hydra: Optional[dict] = None


class EnvParams(BaseEnvConfig):
    goal_location: Optional[tuple[float, float]] = None
    goal_locations: Optional[list[tuple[float, float]]] = None
    sensor_params: Any = None


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
            raise NotImplementedError("UnicycleRobot is deprecated. Use dubins2d instead.")
        elif rtype == "dubins2d":
            self.robot = DubinsRobot(robot_params)
        elif rtype == "dubins2d_fixed_velocity":
            raise NotImplementedError(
                "DubinsRobotFixedVelocity is deprecated. Use dubins2d instead."
            )
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
        renderer: BaseRenderer | None = None,
    ) -> None:
        """Initializes the Smoke Environment supporting single or multiple agents."""
        super().__init__()

        # 1. Load/Parse/Validate the configuration
        if isinstance(cfg, str):
            cfg_loaded = OmegaConf.load(cfg)
        elif isinstance(cfg, dict):
            cfg_loaded = OmegaConf.create(cfg)
        elif isinstance(cfg, DictConfig):
            cfg_loaded = cfg
        else:
            raise TypeError(f"Invalid type for cfg: {type(cfg)}")

        # Convert to container without resolving to avoid immediate interpolation errors
        cfg_dict = OmegaConf.to_container(cfg_loaded, resolve=False)

        # Handle cases where cfg_dict is nested under a namespace (e.g., {"env": {...}})
        if "env" in cfg_dict and len(cfg_dict) == 1:
            cfg_dict = cfg_dict["env"]

        # Recursively remove the 'hydra' key from the raw python dictionary
        def remove_key_recursive(d, key_to_remove):
            if isinstance(d, dict):
                d.pop(key_to_remove, None)
                for k, v in list(d.items()):
                    remove_key_recursive(v, key_to_remove)
            elif isinstance(d, list):
                for item in d:
                    remove_key_recursive(item, key_to_remove)

        remove_key_recursive(cfg_dict, "hydra")

        # Re-resolve any interpolations now that hydra settings are stripped
        resolved_cfg = OmegaConf.create(cfg_dict)
        cfg_dict = OmegaConf.to_container(resolved_cfg, resolve=True)

        validated_cfg = FullEnvConfig(**cfg_dict)

        # 2. Setup core environment parameters (including num_agents and goals assertions)
        self._setup_env_params(validated_cfg)

        # 3. Setup sensor parameters
        self._setup_sensor_params(validated_cfg.sensor)

        # 4. Setup smoke and robot parameters
        self._setup_smoke_and_robot(validated_cfg, robot_params)

        # 5. Initialize the simulator and the agents
        self._setup_simulator()

        # 6. Initialize spaces
        self._init_spaces()

        # Setup renderer (Alternative B)
        if renderer is not None:
            self.renderer = renderer
        elif self.env_params.render and self.env_params.render != RenderMode.NONE:
            self.renderer = SimpleRenderer(validated_cfg)
        else:
            self.renderer = None

        self.current_step = 0

    def _setup_env_params(self, validated_cfg: FullEnvConfig):
        """Initializes self.env_params from the validated configuration."""
        self.env_params = EnvParams(
            world_x_size=validated_cfg.world_x_size,
            world_y_size=validated_cfg.world_y_size,
            max_steps=validated_cfg.max_steps,
            render=validated_cfg.render,
            render_save_every=validated_cfg.render_save_every,
            clock=validated_cfg.clock,
            goal_radius=validated_cfg.goal_radius,
            smoke_density_threshold=validated_cfg.smoke_density_threshold,
            playback_path=validated_cfg.playback_path,
            num_agents=validated_cfg.num_agents,
            collision_radius=validated_cfg.collision_radius,
            terminate_on_collision=validated_cfg.terminate_on_collision,
            collision_penalty=validated_cfg.collision_penalty,
        )

        # Setup goal locations (supporting both singular and plural formats)
        if validated_cfg.goal_locations:
            self.env_params.goal_locations = [tuple(loc) for loc in validated_cfg.goal_locations]
        elif validated_cfg.goal_location:
            self.env_params.goal_locations = [tuple(validated_cfg.goal_location)]
            self.env_params.goal_location = tuple(validated_cfg.goal_location)
        else:
            self.env_params.goal_locations = None

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

    def _setup_sensor_params(self, sensor_cfg: SensorConfig):
        """Decodes the 'sensor' block from validated configuration."""
        sensor_type = sensor_cfg.type

        common_args = {
            "world_x_size": self.env_params.world_x_size,
            "world_y_size": self.env_params.world_y_size,
        }

        if sensor_type == "camera_1d":
            d = Camera1DSensorParams(**common_args)
            self.env_params.sensor_params = Camera1DSensorParams(
                fov_size_degrees=float(
                    sensor_cfg.fov_size_degrees
                    if sensor_cfg.fov_size_degrees is not None
                    else d.fov_size_degrees
                ),
                num_rays=int(
                    sensor_cfg.num_rays if sensor_cfg.num_rays is not None else d.num_rays
                ),
                step_size=float(
                    sensor_cfg.step_size if sensor_cfg.step_size is not None else d.step_size
                ),
                opacity_threshold=float(
                    sensor_cfg.opacity_threshold
                    if sensor_cfg.opacity_threshold is not None
                    else d.opacity_threshold
                ),
                max_range=float(
                    sensor_cfg.max_range if sensor_cfg.max_range is not None else d.max_range
                ),
                **common_args,
            )
        elif sensor_type == "downwards":
            self.env_params.sensor_params = DownwardsSensorParams(**common_args)
        elif sensor_type == "point":
            self.env_params.sensor_params = PointSensorParams(**common_args)
        else:
            self.env_params.sensor_params = GlobalSensorParams(**common_args)

    def _setup_smoke_and_robot(
        self, validated_cfg: FullEnvConfig, robot_params_override: RobotParams | None
    ):
        """Logic to prepare SmokeParams and Robot objects."""
        # --- Smoke setup ---
        smoke_cfg = validated_cfg.smoke
        smoke_blobs = []
        if smoke_cfg is not None and smoke_cfg.blobs:
            for b in smoke_cfg.blobs:
                smoke_blobs.append(
                    BlobParams(
                        x_pos=float(b.x),
                        y_pos=float(b.y),
                        intensity=float(b.intensity),
                        spread_rate=float(b.spread),
                    )
                )
        else:
            smoke_blobs = [BlobParams(x_pos=10, y_pos=40, intensity=1.0, spread_rate=1.0)]

        d_smoke = SmokeParams(x_size=10, y_size=10, smoke_blob_params=[])
        self.smoke_params = SmokeParams(
            x_size=self.env_params.world_x_size,
            y_size=self.env_params.world_y_size,
            smoke_blob_params=smoke_blobs,
            resolution=float(
                smoke_cfg.resolution
                if (smoke_cfg is not None and smoke_cfg.resolution is not None)
                else d_smoke.resolution
            ),
        )

        # --- Robot setup ---
        if robot_params_override is not None:
            self.robot_params = robot_params_override
        else:
            self.robot_params = validated_cfg.robot

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

    def render(self, controller=None):
        """Standard Gym/Gymnasium render method."""
        if self.renderer is not None:
            return self.renderer.render({"env": self, "controller": controller})
        return None

    def _render_frame(self, controller=None):
        """Compatibility wrapper for standard/simple renderer integration."""
        return self.render(controller=controller)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def main(cfg) -> None:
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
    cfg_raw = OmegaConf.to_container(cfg, resolve=False)

    def remove_key_recursive(d, key_to_remove):
        if isinstance(d, dict):
            d.pop(key_to_remove, None)
            for k, v in list(d.items()):
                remove_key_recursive(v, key_to_remove)
        elif isinstance(d, list):
            for item in d:
                remove_key_recursive(item, key_to_remove)

    remove_key_recursive(cfg_raw, "hydra")

    multi_cfg = OmegaConf.to_container(OmegaConf.create(cfg_raw), resolve=True)
    if "env" in multi_cfg and len(multi_cfg) == 1:
        multi_cfg = multi_cfg["env"]

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


if __name__ == "__main__":
    from hydra import compose, initialize
    print("🧪 Running SmokeEnv tests with programmatic hydra.compose...")
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="env/smoke_env")
    main(cfg)
