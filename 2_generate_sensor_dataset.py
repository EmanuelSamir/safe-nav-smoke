import os
import sys

import hydra
import numpy as np
from datasets import Dataset
from omegaconf import DictConfig
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from envs.simulator.playback_schema import SmokeDataSchema
from envs.smoke_env import SmokeEnv


class RandomizedLQRController:
    """A controller that uses LQR-like feedback for smoothness and varying velocity.

    Includes 20% randomization to increase dataset diversity.
    """

    def __init__(self, world_size_x, world_size_y, robot_type="dubins2d", action_space=None):
        """Initializes the controller with the world size, robot type, and action space."""
        self.world_size = np.array([world_size_x, world_size_y])
        self.target = None
        self.robot_type = robot_type
        self.action_space = action_space

    def get_action(self, state):
        pos = state["location"]
        angle = state["angle"]

        # 1. Update target if reached
        if self.target is None or np.linalg.norm(pos - self.target) < 2.0:
            # Keep targets away from the very edge
            self.target = np.random.uniform(5.0, self.world_size - 5.0)

        diff = self.target - pos
        dist = np.linalg.norm(diff)
        target_angle = np.arctan2(diff[1], diff[0])

        angle_err = target_angle - angle
        angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi

        # 2. Determine Nominal Action (LQR-like behavior)
        if self.robot_type == "unicycle":
            # Action: [acceleration, angular_velocity]
            curr_v = state.get("velocity", 0.0)
            # Desired velocity decreases as we approach target, and is higher if pointing towards it
            v_ref = np.clip(dist * 0.5, 0.5, 3.0) * np.cos(angle_err)
            v_ref = max(0.2, v_ref)  # Always move forward a bit

            a = 2.0 * (v_ref - curr_v)  # Proportional acceleration
            w = 4.0 * angle_err  # Steer towards target
            action = np.array([a, w])
        else:
            # Dubins Action: [velocity, angular_velocity]
            v = np.clip(dist * 0.4, 0.5, 4.0) * np.cos(angle_err)
            v = max(0.5, v)
            w = 5.0 * angle_err
            action = np.array([v, w])

        # 3. Apply 20% randomization to the linear component (velocity or acceleration)
        if np.random.rand() < 0.2:
            action[0] = np.random.uniform(self.action_space.low[0], self.action_space.high[0])

        return np.clip(action, self.action_space.low, self.action_space.high)


@hydra.main(version_base=None, config_path="configs/env", config_name="smoke_env")
def main(cfg: DictConfig):
    # 1. Setup paths and parameters
    test_mode = cfg.get("test", False)  # Enable via 'python script.py test=true'
    playback_path = "data/smoke_playback_hf"
    output_dir = "data/sensor_dataset_hf"

    if not test_mode:
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Override cfg to use playback
    cfg.playback_path = playback_path
    cfg.render = "human" if test_mode else "none"

    print(f"Loading playback from {playback_path} (Test Mode: {test_mode})...")
    env = SmokeEnv(cfg=cfg)

    # 2. Get playback info to know how many episodes to run
    if test_mode:
        num_episodes = 1
        steps_per_episode = 50
    else:
        num_episodes = cfg.get("num_episodes", env.smoke_simulator.num_episodes)
        steps_per_episode = env.smoke_simulator.max_steps

    print(f"Starting data collection: {num_episodes} episodes, {steps_per_episode} steps each.")

    controller = RandomizedLQRController(
        env.env_params.world_x_size,
        env.env_params.world_y_size,
        robot_type=env.robot_params.robot_type,
        action_space=env.action_space,
    )
    print(env.action_space)

    def generate_rows():
        """Generator to yield transition rows for data efficiency."""
        # Note: we use values defined in main()'s scope
        for ep in tqdm(range(num_episodes), desc="Episodes"):
            obs, _ = env.reset()

            for step in range(steps_per_episode - 1):
                # 1. Get current full state from simulator
                full_map = env.smoke_simulator.get_smoke_map().copy()

                # 2. Get action from randomized controller
                action = controller.get_action(obs)

                # 3. Record current step (Standard Replay Buffer: s, a)
                current_row = {
                    SmokeDataSchema.OBS_LOCATION: obs["location"].tolist(),
                    SmokeDataSchema.OBS_ANGLE: [float(obs["angle"])],
                    SmokeDataSchema.OBS_READINGS: obs["smoke_density"].tolist(),
                    SmokeDataSchema.OBS_FULL_MAP: full_map.tolist(),
                    SmokeDataSchema.ACTION: action.tolist(),
                }

                # 4. Step environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_full_map = env.smoke_simulator.get_smoke_map().copy()

                if test_mode:
                    env._render_frame(controller=controller)
                    print(
                        f"Recorded Step Data - Location: {np.round(obs['location'], 2)}, Action: {np.round(action, 2)}"
                    )
                    print(f"Observation: {np.round(obs['smoke_density'], 1)}")

                # 5. Add next state and RL signals (s', r, d)
                current_row.update(
                    {
                        SmokeDataSchema.NEXT_OBS_LOCATION: next_obs["location"].tolist(),
                        SmokeDataSchema.NEXT_OBS_READINGS: next_obs["smoke_density"].tolist(),
                        SmokeDataSchema.NEXT_OBS_FULL_MAP: next_full_map.tolist(),
                        SmokeDataSchema.REWARD: float(reward),
                        SmokeDataSchema.TERMINATED: bool(terminated),
                        SmokeDataSchema.TRUNCATED: bool(truncated),
                    }
                )

                yield current_row
                obs = next_obs

                if terminated or truncated:
                    break

    # 3. Create or consume the generator based on test mode
    if test_mode:
        # Just consume the generator once for visibility
        print("Starting Test Mode simulation...")
        for _ in generate_rows():
            pass
        print("Test Mode complete. No data saved.")
        return

    print(f"Starting incremental data collection of {num_episodes} episodes...")
    dataset = Dataset.from_generator(generate_rows)

    # Save to disk
    dataset.save_to_disk(output_dir)
    print(f"Dataset successfully saved to {output_dir}")
    print(f"Features: {dataset.features}")


if __name__ == "__main__":
    main()
