import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv, DynamicSmokeParams
from agents.basic_robot import RobotParams
from simulator.sensor import DownwardsSensorParams
from simulator.static_smoke import SmokeBlobParams
from src.cbf_controller import CBFController
from envs.replay_buffer import GenericReplayBuffer

@hydra.main(version_base=None, config_path="config", config_name="data_collection/default")
def main(cfg: DictConfig):
    print(f"Starting Data Collection: {cfg.data_collection.num_samples} samples")
    
    # 1. Setup Params
    env_params = EnvParams()
    env_params.world_x_size = cfg.env.world_x_size
    env_params.world_y_size = cfg.env.world_y_size
    env_params.max_steps = cfg.env.max_steps
    env_params.clock = cfg.env.clock
    env_params.render = cfg.env.render
    
    # Sensor
    sensor_params = DownwardsSensorParams(
        world_x_size=env_params.world_x_size,
        world_y_size=env_params.world_y_size,
        points_in_range=cfg.data_collection.sensor.points_in_range,
        fov_size_degrees=cfg.data_collection.sensor.fov_size_degrees
    )
    env_params.sensor_params = sensor_params

    # Robot
    robot_params = RobotParams()
    robot_params.world_x_size = env_params.world_x_size
    robot_params.world_y_size = env_params.world_y_size
    robot_params.action_min = np.array(cfg.agent.action_min)
    robot_params.action_max = np.array(cfg.agent.action_max)
    robot_params.dt = cfg.agent.dt

    # Replay Buffer
    replay_buffer = GenericReplayBuffer(
        buffer_size=cfg.data_collection.num_samples, 
        data_keys=['actions', 'state', 'smoke_values', 'smoke_value_positions', 'done']
    )

    datasize = 0
    pbar = tqdm(total=cfg.data_collection.num_samples)

    while datasize < cfg.data_collection.num_samples:
        # Randomize Smoke
        smoke_blobs = [
            SmokeBlobParams(
                x_pos=env_params.world_x_size * np.random.rand(),
                y_pos=env_params.world_y_size * np.random.rand(),
                intensity=cfg.data_collection.smoke.intensity,
                spread_rate=cfg.data_collection.smoke.spread_rate
            ) for _ in range(cfg.data_collection.num_smoke_blobs)
        ]
        
        smoke_params = DynamicSmokeParams(
            x_size=env_params.world_x_size, 
            y_size=env_params.world_y_size, 
            smoke_blob_params=smoke_blobs, 
            resolution=cfg.data_collection.smoke.resolution
        )

        # Init Env
        env = DynamicSmokeEnv(env_params, robot_params, smoke_params)
        
        # Random Start
        initial_loc = np.array([
            env_params.world_x_size * np.random.rand(), 
            env_params.world_y_size * np.random.rand()
        ])
        
        state, _ = env.reset(initial_state={
            "location": initial_loc, 
            "angle": 0.0, 
            "smoke_density": 0.0
        })

        # Controller (Waypoint Follower)
        waypoint = np.array([
            env_params.world_x_size * np.random.rand(), 
            env_params.world_y_size * np.random.rand()
        ])
        # We use CBFController just for nominal_control logic
        cbf = CBFController(env_params, robot_params, waypoint)

        finished = False
        while not finished:
            # Check waypoint reached
            if np.linalg.norm(waypoint - state["location"]) < 2.0: # 2.0 radius
                waypoint = np.array([
                    env_params.world_x_size * np.random.rand(), 
                    env_params.world_y_size * np.random.rand()
                ])
                cbf.goal = waypoint

            # Get Action
            pose = np.array([state["location"][0], state["location"][1], state["angle"]])
            action_input = cbf.nominal_control(pose)

            # Step
            next_state, reward, terminated, truncated, _ = env.step(np.array(action_input))
            
            # Store
            replay_buffer.add(
                state=pose,
                actions=action_input,
                smoke_values=state['smoke_density'],
                smoke_value_positions=state['smoke_density_location'],
                done=truncated # Using truncated as done signal for data collection chunks
            )

            state = next_state
            datasize += 1
            pbar.update(1)

            if truncated or datasize >= cfg.data_collection.num_samples:
                finished = True
        
        env.close()

    pbar.close()
    
    # Save
    os.makedirs(cfg.data_collection.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.data_collection.output_dir, "replay_buffer.npz")
    replay_buffer.save_to_file(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()
