from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf

from src.agents.basic_robot import RobotParams
from src.agents.dubins_robot import DubinsRobot
from src.env.simulator.playback import Playback, PlaybackParams
from src.env.simulator.playback_schema import SmokeDataSchema
from src.env.simulator.sensor import (
    Camera1DSensor,
    DownwardsSensor,
    GlobalSensor,
    SensorOutput,
)
from src.env.simulator.smoke import BlobParams, Smoke, SmokeParams
from src.utils.config_utils import get_device
from src.visualization import BaseRenderer, SimpleRenderer


class RenderMode(str, Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    NONE = "none"


@dataclass
class EnvConfig:
    world_x_size: float
    world_y_size: float
    max_steps: int
    clock: float
    render: Union[RenderMode, str]
    render_save_every: int
    goal_radius: float
    num_agents: int
    collision_radius: float
    terminate_on_collision: bool
    collision_penalty: float
    smoke_density_threshold: Optional[float]

    goal_locations: Optional[list[list[float]]] = None
    initial_locations: Optional[list[list[float]]] = None

    # Extra parameters in smoke_env.yaml configuration
    test: Optional[bool] = None
    num_episodes: Optional[int] = None
    save_transitions: bool = False

    def __post_init__(self):
        try:
            self.render = RenderMode(self.render)
        except ValueError:
            valid_values = [e.value for e in RenderMode]
            raise ValueError(f"render must be one of {valid_values}, got {self.render}")


class SmokeAgent:
    """Defines an individual agent's robot, goal, and observation/action spaces."""

    def __init__(
        self,
        agent_id: int,
        robot_params: RobotParams,
        env_params: EnvConfig,
        goal_location: tuple[float, float] | None,
    ):
        self.agent_id = agent_id
        self.robot_params = robot_params
        self.env_params = env_params
        self.goal_location = goal_location
        self.goal_radius = env_params.goal_radius

        if robot_params.name == "dubins2d":
            self.robot = DubinsRobot(robot_params)
        else:
            raise NotImplementedError(f"Robot type {robot_params.name} not implemented")

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

    def reset(self, initial_state=None):
        """Resets the individual agent's robot state."""
        if initial_state is None:
            obs = self.observation_space.sample()
        else:
            obs = initial_state

        loc_x = np.ravel(obs["location"])[0]
        loc_y = np.ravel(obs["location"])[1]
        angle = np.ravel(obs["angle"])[0]

        state_t = torch.tensor(
            [loc_x, loc_y, angle], dtype=torch.float32, device=self.robot_params.device
        )
        self.robot.reset(state_t)

    def get_robot_odom(self):
        """Retrieves robot odometry/state."""
        state = self.robot.get_state()
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        return {"location": state[:2], "angle": float(state[2])}


class SmokeEnv(gym.Env):
    def __init__(
        self,
        env_params: EnvConfig,
        robot_params: RobotParams,
        sensor_params: object,
        simulator_params: object,
        renderer: BaseRenderer | None = None,
    ) -> None:
        """Initializes the Smoke Environment supporting single or multiple agents."""
        super().__init__()

        self.env_params = env_params
        self.robot_params = robot_params

        if self.env_params.goal_locations is not None:
            self.env_params.goal_locations = [tuple(loc) for loc in self.env_params.goal_locations]
            assert len(self.env_params.goal_locations) == self.env_params.num_agents, (
                f"Number of goal locations ({len(self.env_params.goal_locations)}) must match "
                f"the number of agents ({self.env_params.num_agents})."
            )

        if self.env_params.initial_locations is not None:
            self.env_params.initial_locations = [
                tuple(loc) for loc in self.env_params.initial_locations
            ]
            assert len(self.env_params.initial_locations) == self.env_params.num_agents, (
                f"Number of initial locations ({len(self.env_params.initial_locations)}) must match "
                f"the number of agents ({self.env_params.num_agents})."
            )

        # 2. Setup sensor
        assert sensor_params is not None, "sensor_params must be provided to SmokeEnv"
        self.env_params_sensor_params = sensor_params

        sensor_type = getattr(self.env_params_sensor_params, "sensor_type", "global")
        if sensor_type in ["camera_1d", "camera1d"]:
            self.sensor = Camera1DSensor(self.env_params_sensor_params)
        elif sensor_type == "downwards":
            self.sensor = DownwardsSensor(self.env_params_sensor_params)
        else:
            self.sensor = GlobalSensor(self.env_params_sensor_params)

        # 3. Setup simulator
        assert simulator_params is not None, "simulator_params must be provided to SmokeEnv"
        if isinstance(simulator_params, PlaybackParams):
            self.smoke_simulator = Playback(params=simulator_params)
            self._apply_playback_overrides()
        elif isinstance(simulator_params, SmokeParams):
            self.smoke_params = simulator_params
            blobs_list = [BlobParams(x_pos=10, y_pos=40, intensity=1.0, spread_rate=1.0)]
            self.smoke_simulator = Smoke(params=self.smoke_params, blob_params_list=blobs_list)
        else:
            raise TypeError(f"Unsupported simulator params type: {type(simulator_params)}")

        self.env_params_sensor_params.world_x_size = self.env_params.world_x_size
        self.env_params_sensor_params.world_y_size = self.env_params.world_y_size

        # Setup transition buffer if saving is enabled
        self._transition_buffer = [] if self.env_params.save_transitions else None
        self._last_obs = None

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

        # Initialize spaces
        self._init_spaces()

        # Setup renderer
        if renderer is not None:
            self.renderer = renderer
        elif self.env_params.render and self.env_params.render != RenderMode.NONE:
            self.renderer = SimpleRenderer(self.env_params)
        else:
            self.renderer = None

        self.current_step = 0

    def _apply_playback_overrides(self) -> None:
        """Applies configuration overrides from the loaded playback data."""
        if isinstance(self.smoke_simulator, Playback):
            if self.smoke_simulator.x_size:
                self.env_params.world_x_size = self.smoke_simulator.x_size
            if self.smoke_simulator.y_size:
                self.env_params.world_y_size = self.smoke_simulator.y_size
            if self.smoke_simulator.max_steps:
                self.env_params.max_steps = self.smoke_simulator.max_steps

    def _init_spaces(self) -> None:
        """Defines Gym action and observation spaces conforming to composite gymnasium.spaces standards."""
        self.action_space = spaces.Dict(
            {f"agent_{i}": agent.action_space for i, agent in enumerate(self.agents)}
        )
        self.observation_space = spaces.Dict(
            {f"agent_{i}": agent.observation_space for i, agent in enumerate(self.agents)}
        )

    def reset(
        self,
        initial_state: list[dict] | None = None,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict], dict]:
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

        if initial_state is not None:
            assert isinstance(initial_state, list), "initial_state must be a list"
            assert len(initial_state) == self.env_params.num_agents, (
                f"initial_state length ({len(initial_state)}) must match num_agents ({self.env_params.num_agents})"
            )

        for i, agent in enumerate(self.agents):
            agent_init = initial_state[i] if initial_state is not None else None
            agent.reset(initial_state=agent_init)

        obs = self._get_obs()
        if self.env_params.save_transitions:
            self._last_obs = obs
        return obs, {}

    def get_robot_odom(self, agent_idx: int = 0):
        """Retrieves robot odometry for a specific agent index."""
        return self.agents[agent_idx].get_robot_odom()

    def get_smoke_density_sensor(
        self, pos: Union[np.ndarray, torch.Tensor], return_location: bool = True
    ):
        assert self.sensor is not None, "Sensor must have been initialized"
        if not isinstance(pos, torch.Tensor):
            pos = torch.as_tensor(pos, dtype=torch.float32, device=get_device())
        sensor_output = self.sensor.read(self.smoke_simulator, curr_pos=pos)
        if return_location:
            return sensor_output.readings, sensor_output.positions
        return sensor_output.readings

    def get_smoke_density_in_robot(self, agent_idx: int = 0) -> float:
        odom = self.get_robot_odom(agent_idx)
        pos_x, pos_y = odom["location"]
        smoke_density_in_robot = self.smoke_simulator.get_smoke_density(np.array([pos_x, pos_y]))
        return float(smoke_density_in_robot)

    def _get_obs_for_agent(
        self, agent: SmokeAgent, global_sensor_output: SensorOutput | None = None
    ):
        """Computes the observation dictionary for a single agent, reusing cached global readings if applicable."""
        odom = agent.get_robot_odom()
        pos_x, pos_y = odom["location"]
        angle = odom["angle"]

        if global_sensor_output is not None:
            smoke_density = global_sensor_output.readings
            smoke_density_location = global_sensor_output.positions
        else:
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
        return obs

    def _get_obs(self) -> dict[str, dict]:
        """Constructs the centralized observation space."""
        global_sensor_output = None
        if self.env_params_sensor_params.sensor_type == "global":
            dummy_pos = torch.tensor([0.0, 0.0], dtype=torch.float32, device=get_device())
            global_sensor_output = self.sensor.read(self.smoke_simulator, curr_pos=dummy_pos)

        return {
            f"agent_{i}": self._get_obs_for_agent(agent, global_sensor_output=global_sensor_output)
            for i, agent in enumerate(self.agents)
        }

    def _get_info(self) -> dict[str, dict]:
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

    def _get_reward_for_agent(self, agent: SmokeAgent, obs_agent: dict) -> float:
        pos_x, pos_y = obs_agent["location"]
        reward = 0.0
        if agent.goal_location is not None:
            if np.linalg.norm(np.array([pos_x, pos_y]) - agent.goal_location) < agent.goal_radius:
                reward = 1.0

        if self._check_collision_for_agent(agent, np.array([pos_x, pos_y])):
            reward += self.env_params.collision_penalty
        return reward

    def _get_terminated_for_agent(self, agent: SmokeAgent, obs_agent: dict) -> bool:
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

        if self.env_params.terminate_on_collision:
            if self._check_collision_for_agent(agent, np.array([pos_x, pos_y])):
                return True
        return False

    def _get_reward(self, obs: dict[str, dict], action: dict) -> dict[str, float]:
        return {
            f"agent_{i}": self._get_reward_for_agent(agent, obs[f"agent_{i}"])
            for i, agent in enumerate(self.agents)
        }

    def _get_terminated(self, obs: dict[str, dict]) -> dict[str, bool]:
        return {
            f"agent_{i}": self._get_terminated_for_agent(agent, obs[f"agent_{i}"])
            for i, agent in enumerate(self.agents)
        }

    def _get_truncated(self, obs: dict[str, dict]) -> dict[str, bool]:
        is_truncated = self.current_step >= self.env_params.max_steps
        return {f"agent_{i}": is_truncated for i in range(self.env_params.num_agents)}

    def step(
        self, action: dict
    ) -> tuple[
        dict[str, dict], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        self.current_step += 1

        assert isinstance(action, dict), "Action must be a dictionary mapping agent keys to actions"

        for i, agent in enumerate(self.agents):
            agent_key = f"agent_{i}"
            act = action.get(agent_key, action.get(i))
            assert act is not None, f"Action for agent {agent_key} not provided"

            robot = agent.robot
            if not isinstance(act, torch.Tensor):
                act = torch.as_tensor(act, dtype=torch.float32, device=robot.params.device)
            else:
                act = act.to(robot.params.device)
            robot.dynamic_step(act)

        self.smoke_simulator.step()

        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        terminated = self._get_terminated(obs)
        truncated = self._get_truncated(obs)
        info = self._get_info()

        self._record_transition(action, obs, reward, terminated, truncated)

        return obs, reward, terminated, truncated, info

    def render(self, controller=None):
        if self.renderer is not None:
            return self.renderer.render({"env": self, "controller": controller})
        return None

    def _render_frame(self, controller=None):
        return self.render(controller=controller)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
        self._save_transitions()

    def _record_transition(
        self,
        action: dict,
        obs: dict[str, dict],
        reward: dict[str, float],
        terminated: dict[str, bool],
        truncated: dict[str, bool],
    ) -> None:
        if not self.env_params.save_transitions or self._last_obs is None:
            return

        assert isinstance(action, dict), "Action must be a dictionary"

        for i in range(self.env_params.num_agents):
            agent_key = f"agent_{i}"
            obs_agent = self._last_obs[agent_key]
            next_obs_agent = obs[agent_key]

            act_val = action.get(agent_key, action.get(i))
            assert act_val is not None, f"Action for agent {agent_key} not provided"

            if isinstance(act_val, torch.Tensor):
                act_val = act_val.detach().cpu().numpy()
            elif isinstance(act_val, (list, tuple)):
                act_val = np.array(act_val)

            rew_val = reward[agent_key]
            term_val = terminated[agent_key]
            trunc_val = truncated[agent_key]

            def to_numpy(val):
                if isinstance(val, torch.Tensor):
                    return val.detach().cpu().numpy()
                return val

            obs_loc = to_numpy(obs_agent["location"])
            obs_angle = to_numpy(obs_agent["angle"])
            obs_readings = to_numpy(obs_agent["smoke_density"])
            next_obs_loc = to_numpy(next_obs_agent["location"])
            next_obs_readings = to_numpy(next_obs_agent["smoke_density"])

            transition = {
                SmokeDataSchema.OBS_LOCATION: [float(x) for x in np.ravel(obs_loc)],
                SmokeDataSchema.OBS_ANGLE: [float(obs_angle)]
                if isinstance(obs_angle, (int, float, np.number))
                else [float(x) for x in np.ravel(obs_angle)],
                SmokeDataSchema.OBS_READINGS: [float(x) for x in np.ravel(obs_readings)],
                SmokeDataSchema.ACTION: [float(x) for x in np.ravel(act_val)],
                SmokeDataSchema.REWARD: float(rew_val),
                SmokeDataSchema.NEXT_OBS_LOCATION: [float(x) for x in np.ravel(next_obs_loc)],
                SmokeDataSchema.NEXT_OBS_READINGS: [float(x) for x in np.ravel(next_obs_readings)],
                SmokeDataSchema.TERMINATED: bool(term_val),
                SmokeDataSchema.TRUNCATED: bool(trunc_val),
            }
            self._transition_buffer.append(transition)

        self._last_obs = obs

    def _save_transitions(self) -> None:
        if (
            getattr(self, "env_params", None) is not None
            and getattr(self.env_params, "save_transitions", False)
            and getattr(self, "_transition_buffer", None)
        ):
            import os

            try:
                from hydra.core.hydra_config import HydraConfig

                output_dir = HydraConfig.get().runtime.output_dir
            except (ValueError, ImportError, KeyError):
                output_dir = "outputs"

            import datasets
            from datasets import Dataset

            features = datasets.Features(
                {
                    SmokeDataSchema.OBS_LOCATION: datasets.Sequence(
                        datasets.Value("float32"), length=2
                    ),
                    SmokeDataSchema.OBS_ANGLE: datasets.Sequence(
                        datasets.Value("float32"), length=1
                    ),
                    SmokeDataSchema.OBS_READINGS: datasets.Sequence(datasets.Value("float32")),
                    SmokeDataSchema.ACTION: datasets.Sequence(datasets.Value("float32")),
                    SmokeDataSchema.REWARD: datasets.Value("float32"),
                    SmokeDataSchema.NEXT_OBS_LOCATION: datasets.Sequence(
                        datasets.Value("float32"), length=2
                    ),
                    SmokeDataSchema.NEXT_OBS_READINGS: datasets.Sequence(datasets.Value("float32")),
                    SmokeDataSchema.TERMINATED: datasets.Value("bool"),
                    SmokeDataSchema.TRUNCATED: datasets.Value("bool"),
                }
            )

            ds = Dataset.from_list(self._transition_buffer, features=features)
            
            save_dir = getattr(self, "save_transitions_path", None)
            if save_dir is None:
                save_dir = getattr(self.env_params, "save_transitions_path", None)
            if save_dir is None:
                save_dir = os.path.join(output_dir, "env_transitions")
                
            ds.save_to_disk(save_dir)
            print(f"[SmokeEnv] Successfully saved {len(ds)} transitions to {save_dir}")


def main(cfg) -> None:
    # 1. Resolve configuration in-place
    OmegaConf.resolve(cfg)

    env_params = OmegaConf.to_object(cfg.env)
    robot_params = OmegaConf.to_object(cfg.agent)
    smoke_params = OmegaConf.to_object(cfg.simulator)
    sensor_params = OmegaConf.to_object(cfg.sensor)

    # Force save_transitions to True for testing the dataset serialization code
    env_params.save_transitions = True

    # Test single agent initialization
    print("Testing Single Agent Environment Initialization...")
    env = SmokeEnv(
        env_params=env_params,
        robot_params=robot_params,
        sensor_params=sensor_params,
        simulator_params=smoke_params,
    )

    # Use config initial locations if specified, otherwise None
    initial_loc = env_params.initial_locations
    if initial_loc is not None and len(initial_loc) > 0:
        initial_state = [
            {
                "location": np.array(initial_loc[0]),
                "angle": 1.0,
            }
        ]
    else:
        initial_state = None

    obs, _ = env.reset(initial_state=initial_state)
    print(f"Single-agent Reset Obs ID: {obs['agent_0']['id']}")
    print(f"Single-agent Reset Obs Location: {obs['agent_0']['location']}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Single-agent Step Reward: {reward['agent_0']}")
    env.close()

    # Test multi-agent initialization
    num_agents = env_params.num_agents
    print(
        f"\nTesting Multi-Agent Environment (num_agents = {num_agents}) Initialization with Composite Spaces..."
    )
    env_multi = SmokeEnv(
        env_params=env_params,
        robot_params=robot_params,
        sensor_params=sensor_params,
        simulator_params=smoke_params,
    )

    # Use config initial locations if specified, otherwise None
    if env_params.initial_locations is not None and len(env_params.initial_locations) > 0:
        initial_state_multi = [
            {"location": np.array(loc), "angle": 1.0} for loc in env_params.initial_locations
        ]
    else:
        initial_state_multi = None

    obs_multi, _ = env_multi.reset(initial_state=initial_state_multi)
    if "id" in obs_multi:
        print(f"Agent 0 ID: {obs_multi['id']} | Location: {obs_multi['location']}")
    else:
        print(f"Multi-agent Reset Obs Keys: {list(obs_multi.keys())}")
        for k, o in obs_multi.items():
            print(f"Agent {k} ID: {o['id']} | Location: {o['location']}")

    actions = env_multi.action_space.sample()
    obs_multi, rewards, terminateds, truncateds, infos = env_multi.step(actions)
    if isinstance(rewards, dict):
        print(f"Multi-agent Step Rewards Keys: {list(rewards.keys())}")
        print(f"Multi-agent Step Rewards: {rewards}")
    else:
        print(f"Single-agent Step Reward: {rewards}")

    max_steps = env_params.max_steps
    steps_to_run = min(5, max_steps) if max_steps is not None else 5
    for _ in range(steps_to_run):
        actions = env_multi.action_space.sample()
        obs_multi, rewards, terminateds, truncateds, infos = env_multi.step(actions)
        env_multi.render()

    env_multi.close()
    print("\nAll tests executed successfully!")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def run_tests(cfg) -> None:
    main(cfg)


if __name__ == "__main__":
    run_tests()
