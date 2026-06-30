from enum import Enum
from typing import Union

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.agents.dubins_robot import DubinsRobot
from src.agents.schemas import RobotConfig
from src.env.schemas import EnvConfig
from src.env.simulator.playback import Playback
from src.env.simulator.schemas import (
    BaseSensorConfig,
    BaseSimConfig,
    PlaybackConfig,
    SmokeConfig,
)
from src.env.simulator.sensor import (
    Camera1DSensor,
    DownwardsSensor,
    GlobalSensor,
    SensorOutput,
)
from src.env.simulator.smoke import Smoke
from src.env.simulator.smoke_data_schema import SmokeDataSchema
from src.utils.config_utils import get_device
from src.visualization import BaseRenderer, RenderConfig, SimpleRenderer


class RenderMode(str, Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    NONE = "none"


class SmokeAgent:
    """Defines an individual agent's robot, goal, and observation/action spaces."""

    def __init__(
        self,
        agent_id: int,
        robot_cfg: RobotConfig,
        env_cfg: EnvConfig,
        goal_location: tuple[float, float] | None,
    ):
        """Initialize agent's class."""
        self.agent_id = agent_id
        self.robot_cfg = robot_cfg
        self.env_cfg = env_cfg
        self.goal_location = goal_location
        self.goal_radius = env_cfg.goal_radius

        if robot_cfg.name == "dubins2d":
            self.robot = DubinsRobot(robot_cfg)
        else:
            raise NotImplementedError(f"Robot type {robot_cfg.name} not implemented")

        self._init_spaces()

    def _init_spaces(self):
        """Defines Gym spaces for this individual agent."""
        self.action_space = spaces.Box(
            low=np.array(self.robot_cfg.action_min),
            high=np.array(self.robot_cfg.action_max),
            shape=(self.robot_cfg.action_dim,),
        )

        x_lim, y_lim = self.env_cfg.world_x_size, self.env_cfg.world_y_size

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

    def reset(self, initial_state: dict | None = None):
        """Resets the individual agent's robot state."""
        if initial_state is None:
            obs = self.observation_space.sample()
        else:
            obs = initial_state

        loc_x = np.ravel(obs["location"])[0]
        loc_y = np.ravel(obs["location"])[1]
        angle = np.ravel(obs["angle"])[0]

        state_t = torch.tensor(
            [loc_x, loc_y, angle], dtype=torch.float32, device=self.robot_cfg.device
        )
        self.robot.reset(state_t)

    def get_robot_odom(self):
        """Retrieves robot odometry/state."""
        state = self.robot.get_state()
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        return {"location": state[:2], "angle": float(state[2])}


class SmokeEnv(ParallelEnv):
    def __init__(
        self,
        env_cfg: EnvConfig,
        robot_cfg: RobotConfig,
        sensor_cfg: BaseSensorConfig,
        simulator_cfg: BaseSimConfig,
        renderer: BaseRenderer | None = None,
    ) -> None:
        """Initializes the Smoke Environment supporting single or multiple agents."""
        super().__init__()

        self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg

        if self.env_cfg.goal_locations is not None:
            self.env_cfg.goal_locations = [tuple(loc) for loc in self.env_cfg.goal_locations]
            assert len(self.env_cfg.goal_locations) == self.env_cfg.num_agents, (
                f"Number of goal locations ({len(self.env_cfg.goal_locations)}) must match "
                f"the number of agents ({self.env_cfg.num_agents})."
            )

        if self.env_cfg.initial_locations is not None:
            self.env_cfg.initial_locations = [tuple(loc) for loc in self.env_cfg.initial_locations]
            assert len(self.env_cfg.initial_locations) == self.env_cfg.num_agents, (
                f"Number of initial locations ({len(self.env_cfg.initial_locations)}) must match "
                f"the number of agents ({self.env_cfg.num_agents})."
            )

        # 2. Setup sensor
        assert sensor_cfg is not None, "sensor_cfg must be provided to SmokeEnv"
        self.sensor_cfg = sensor_cfg

        sensor_type = getattr(self.sensor_cfg, "sensor_type", "global")
        if sensor_type in ["camera_1d"]:
            self.sensor = Camera1DSensor(self.sensor_cfg)
        elif sensor_type == "downwards":
            self.sensor = DownwardsSensor(self.sensor_cfg)
        else:
            self.sensor = GlobalSensor(self.sensor_cfg)

        # 3. Setup simulator
        assert simulator_cfg is not None, "simulator_cfg must be provided to SmokeEnv"

        if isinstance(simulator_cfg, PlaybackConfig):
            self.smoke_simulator = Playback(cfg=simulator_cfg)
            self._apply_playback_overrides()
        elif isinstance(simulator_cfg, SmokeConfig):
            self.smoke_params = simulator_cfg
            self.smoke_simulator = Smoke(cfg=self.smoke_params)
        else:
            raise TypeError(f"Unsupported simulator params type: {type(simulator_cfg)}")

        # Setup transition buffer if saving is enabled
        self._transition_buffer = [] if self.env_cfg.save_transitions else None
        self._last_obs = None

        # Create multi-agents
        self.smoke_agents = []
        self.possible_agents = [f"agent_{i}" for i in range(self.env_cfg.num_agents)]
        self.agents = self.possible_agents[:]
        for i in range(self.env_cfg.num_agents):
            goal = (
                self.env_cfg.goal_locations[i] if self.env_cfg.goal_locations is not None else None
            )
            agent = SmokeAgent(
                agent_id=i,
                robot_cfg=self.robot_cfg,
                env_cfg=self.env_cfg,
                goal_location=goal,
            )
            self.smoke_agents.append(agent)

        # Initialize spaces
        self._init_spaces()

        # Setup renderer
        if renderer is not None:
            self.renderer = renderer
        elif self.env_cfg.render and self.env_cfg.render != "none":
            render_cfg = RenderConfig(
                render_mode=self.env_cfg.render,
                clock=self.env_cfg.clock,
                world_x_size=self.env_cfg.world_x_size,
                world_y_size=self.env_cfg.world_y_size,
                collision_radius=self.env_cfg.collision_radius,
            )
            self.renderer = SimpleRenderer(render_cfg)
        else:
            self.renderer = None

        self.current_step = 0

    def _apply_playback_overrides(self) -> None:
        """Applies configuration overrides from the loaded playback data."""
        if isinstance(self.smoke_simulator, Playback):
            print(
                "Overriding sim params from playback:\n",
                f"x_size: {self.smoke_simulator.cfg.x_size}",
                f"y_size: {self.smoke_simulator.cfg.y_size}",
                f"max_steps: {self.smoke_simulator.max_steps}",
            )
            self.env_cfg.world_x_size = self.smoke_simulator.cfg.x_size
            self.env_cfg.world_y_size = self.smoke_simulator.cfg.y_size
            self.env_cfg.max_steps = self.smoke_simulator.max_steps

    def observation_space(self, agent: str):
        agent_idx = int(agent.split("_")[1])
        return self.smoke_agents[agent_idx].observation_space

    def action_space(self, agent: str):
        agent_idx = int(agent.split("_")[1])
        return self.smoke_agents[agent_idx].action_space

    def _init_spaces(self) -> None:
        """Defines Gym action and obs spaces conforming to composite gymnasium.spaces standards."""
        self.action_space = spaces.Dict(
            {f"agent_{i}": agent.action_space for i, agent in enumerate(self.smoke_agents)}
        )
        self.observation_space = spaces.Dict(
            {f"agent_{i}": agent.observation_space for i, agent in enumerate(self.smoke_agents)}
        )

    def reset(
        self,
        initial_state: list[dict] | None = None,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict], dict]:
        self.window = {"fig": None, "ax": None, "cax": None}
        self.current_step = 0
        self.agents = self.possible_agents[:]

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
            assert len(initial_state) == self.env_cfg.num_agents, (
                f"initial_state length ({len(initial_state)}) must match num_agents ({self.env_cfg.num_agents})"
            )

        for i, agent in enumerate(self.smoke_agents):
            agent_init = initial_state[i] if initial_state is not None else None
            agent.reset(initial_state=agent_init)

        obs = self._get_obs()
        if self.env_cfg.save_transitions:
            self._last_obs = obs
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def get_robot_odom(self, agent_idx: int = 0):
        """Retrieves robot odometry for a specific agent index."""
        return self.smoke_agents[agent_idx].get_robot_odom()

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
        if self.sensor_cfg.sensor_type == "global":
            dummy_pos = torch.tensor([0.0, 0.0], dtype=torch.float32, device=get_device())
            global_sensor_output = self.sensor.read(self.smoke_simulator, curr_pos=dummy_pos)

        return {
            agent_key: self._get_obs_for_agent(
                self.smoke_agents[int(agent_key.split("_")[1])],
                global_sensor_output=global_sensor_output,
            )
            for agent_key in self.agents
        }

    def _get_info(self) -> dict[str, dict]:
        infos = {}
        for agent_key in self.agents:
            i = int(agent_key.split("_")[1])
            agent = self.smoke_agents[i]
            pos_x, pos_y = agent.get_robot_odom()["location"]
            pos = np.array([pos_x, pos_y])

            reached_goal = False
            if agent.goal_location is not None:
                if np.linalg.norm(pos - agent.goal_location) < agent.goal_radius:
                    reached_goal = True

            collided = self._check_collision_for_agent(agent, pos)

            smoke_density = self.smoke_simulator.get_smoke_density(pos)
            smoke_death = False
            if (
                self.env_cfg.smoke_density_threshold is not None
                and smoke_density > self.env_cfg.smoke_density_threshold
            ):
                smoke_death = True

            infos[agent_key] = {
                "is_success": reached_goal,
                "collision": collided,
                "smoke_death": smoke_death,
            }
        return infos

    def _check_collision_for_agent(self, agent: SmokeAgent, pos: np.ndarray) -> bool:
        """Helper to check if a given agent position collides with any other agent."""
        if self.env_cfg.num_agents > 1:
            for other in self.smoke_agents:
                if other.agent_id != agent.agent_id:
                    other_pos = other.get_robot_odom()["location"]
                    dist = np.linalg.norm(pos - other_pos)
                    if dist < (self.env_cfg.collision_radius * 2):
                        return True
        return False

    def _get_reward_for_agent(self, agent: SmokeAgent, obs_agent: dict) -> float:
        pos_x, pos_y = obs_agent["location"]
        reward = 0.0
        if agent.goal_location is not None:
            if np.linalg.norm(np.array([pos_x, pos_y]) - agent.goal_location) < agent.goal_radius:
                reward = 1.0

        if self._check_collision_for_agent(agent, np.array([pos_x, pos_y])):
            reward += self.env_cfg.collision_penalty
        return reward

    def _get_terminated_for_agent(self, agent: SmokeAgent, obs_agent: dict) -> bool:
        pos_x, pos_y = obs_agent["location"]
        smoke_density_in_robot = self.smoke_simulator.get_smoke_density(np.array([pos_x, pos_y]))

        if (
            self.env_cfg.smoke_density_threshold is not None
            and smoke_density_in_robot > self.env_cfg.smoke_density_threshold
        ):
            return True

        if agent.goal_location is not None:
            if np.linalg.norm(np.array([pos_x, pos_y]) - agent.goal_location) < agent.goal_radius:
                return True

        if self.env_cfg.terminate_on_collision:
            if self._check_collision_for_agent(agent, np.array([pos_x, pos_y])):
                return True
        return False

    def _get_reward(self, obs: dict[str, dict], action: dict) -> dict[str, float]:
        return {
            agent_key: self._get_reward_for_agent(
                self.smoke_agents[int(agent_key.split("_")[1])], obs[agent_key]
            )
            for agent_key in self.agents
        }

    def _get_terminations(self, obs: dict[str, dict]) -> dict[str, bool]:
        return {
            agent_key: self._get_terminated_for_agent(
                self.smoke_agents[int(agent_key.split("_")[1])], obs[agent_key]
            )
            for agent_key in self.agents
        }

    def _get_truncations(self, obs: dict[str, dict]) -> dict[str, bool]:
        is_truncated = self.current_step >= self.env_cfg.max_steps
        return dict.fromkeys(self.agents, is_truncated)

    def step(
        self, action: dict
    ) -> tuple[
        dict[str, dict], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        self.current_step += 1

        assert isinstance(action, dict), "Action must be a dictionary mapping agent keys to actions"

        for agent_key in self.agents:
            i = int(agent_key.split("_")[1])
            agent = self.smoke_agents[i]
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
        terminations = self._get_terminations(obs)
        truncations = self._get_truncations(obs)
        info = self._get_info()

        self._record_transition(action, obs, reward, terminations, truncations)

        if getattr(self.env_cfg, "remove_dead_agents", True):
            agents_to_remove = [
                agent_key
                for agent_key in self.agents
                if terminations.get(agent_key, False) or truncations.get(agent_key, False)
            ]
            for agent_key in agents_to_remove:
                self.agents.remove(agent_key)

        return obs, reward, terminations, truncations, info

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
        terminations: dict[str, bool],
        truncations: dict[str, bool],
    ) -> None:
        if not self.env_cfg.save_transitions or self._last_obs is None:
            return

        assert isinstance(action, dict), "Action must be a dictionary"

        for agent_key in self.agents:
            i = int(agent_key.split("_")[1])
            obs_agent = self._last_obs[agent_key]
            next_obs_agent = obs[agent_key]

            act_val = action.get(agent_key, action.get(i))
            assert act_val is not None, f"Action for agent {agent_key} not provided"

            if isinstance(act_val, torch.Tensor):
                act_val = act_val.detach().cpu().numpy()
            elif isinstance(act_val, (list, tuple)):
                act_val = np.array(act_val)

            rew_val = reward[agent_key]
            term_val = terminations[agent_key]
            trunc_val = truncations[agent_key]

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
                SmokeDataSchema.TERMINATIONS: bool(term_val),
                SmokeDataSchema.TRUNCATIONS: bool(trunc_val),
            }
            self._transition_buffer.append(transition)

        self._last_obs = obs

    def _save_transitions(self) -> None:
        if (
            getattr(self, "env_cfg", None) is not None
            and getattr(self.env_cfg, "save_transitions", False)
            and getattr(self, "_transition_buffer", None)
        ):
            import os

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
                    SmokeDataSchema.TERMINATIONS: datasets.Value("bool"),
                    SmokeDataSchema.TRUNCATIONS: datasets.Value("bool"),
                }
            )

            ds = Dataset.from_list(self._transition_buffer, features=features)

            save_dir = getattr(self, "save_transitions_path", None)
            if save_dir is None:
                save_dir = getattr(self.env_cfg, "save_transitions_path", None)
            if save_dir is None:
                save_dir = os.path.join(output_dir, "env_transitions")

            ds.save_to_disk(save_dir)
            print(f"[SmokeEnv] Successfully saved {len(ds)} transitions to {save_dir}")


def run_tests() -> None:
    """Run tests for SmokeEnv."""
    from src.agents.schemas import DubinsConfig
    from src.env.schemas import EnvConfig
    from src.env.simulator.schemas import GlobalSensorConfig, SmokeConfig

    env_cfg = EnvConfig()

    robot_cfg = DubinsConfig()

    smoke_params = SmokeConfig()

    sensor_cfg = GlobalSensorConfig()

    # Test single agent initialization
    print("Testing Single Agent Environment Initialization...")
    env = SmokeEnv(
        env_cfg=env_cfg,
        robot_cfg=robot_cfg,
        sensor_cfg=sensor_cfg,
        simulator_cfg=smoke_params,
    )

    # Use config initial locations if specified, otherwise None
    initial_loc = env_cfg.initial_locations
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
    obs, reward, terminations, truncations, info = env.step(action)
    print(f"Single-agent Step Reward: {reward['agent_0']}")
    env.close()

    # Test multi-agent initialization
    env_cfg.num_agents = 2
    env_cfg.initial_locations = [[10.0, 10.0], [20.0, 20.0]]
    env_cfg.goal_locations = [[40.0, 40.0], [30.0, 30.0]]

    num_agents = env_cfg.num_agents
    print(
        f"\nTesting Multi-Agent Environment (num_agents = {num_agents}) Initialization with Composite Spaces..."
    )
    env_multi = SmokeEnv(
        env_cfg=env_cfg,
        robot_cfg=robot_cfg,
        sensor_cfg=sensor_cfg,
        simulator_cfg=smoke_params,
    )

    # Use config initial locations if specified, otherwise None
    if env_cfg.initial_locations is not None and len(env_cfg.initial_locations) > 0:
        initial_state_multi = [
            {"location": np.array(loc), "angle": 1.0} for loc in env_cfg.initial_locations
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
    obs_multi, rewards, terminations, truncations, infos = env_multi.step(actions)
    if isinstance(rewards, dict):
        print(f"Multi-agent Step Rewards Keys: {list(rewards.keys())}")
        print(f"Multi-agent Step Rewards: {rewards}")
    else:
        print(f"Single-agent Step Reward: {rewards}")

    max_steps = env_cfg.max_steps
    steps_to_run = min(5, max_steps) if max_steps is not None else 5
    for _ in range(steps_to_run):
        actions = env_multi.action_space.sample()
        obs_multi, rewards, terminations, truncations, infos = env_multi.step(actions)
        env_multi.render()

    env_multi.close()

    print("\nTesting PettingZoo Agent Removal on Collision...")
    env_cfg_col = EnvConfig(
        world_x_size=50.0,
        world_y_size=50.0,
        num_agents=2,
        collision_radius=1.0,
        goal_radius=1.0,
        terminate_on_collision=True,
        remove_dead_agents=True,
    )
    env_col = SmokeEnv(
        env_cfg=env_cfg_col,
        robot_cfg=robot_cfg,
        sensor_cfg=sensor_cfg,
        simulator_cfg=smoke_params,
    )
    initial_state_col = [
        {"location": np.array([10.0, 10.0]), "angle": 0.0},
        {"location": np.array([10.1, 10.0]), "angle": 0.0},
    ]
    env_col.reset(initial_state=initial_state_col)

    # Send dummy actions
    actions = {agent: np.array([0.0, 0.0]) for agent in env_col.agents}
    obs_col, rewards_col, term_col, trunc_col, infos_col = env_col.step(actions)

    print(f"Step 1 Terminations: {term_col}")
    print(f"Step 1 Infos: {infos_col}")
    print(f"Active Agents after Step 1: {env_col.agents}")

    if len(env_col.agents) == 0:
        print("Agent removal test passed! Both agents collided and were removed from self.agents.")
    else:
        print("Agent removal test failed! Agents were not removed.")

    print("\nAll tests executed successfully!")


if __name__ == "__main__":
    run_tests()
