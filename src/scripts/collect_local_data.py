import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv, DynamicSmokeParams
from agents.basic_robot import RobotParams
from simulator.sensor import DownwardsSensorParams
from simulator.sensor import DownwardsSensorParams
from simulator.dynamic_smoke import SmokeBlobParams
from src.cbf_controller import CBFController
from envs.replay_buffer import GenericReplayBuffer

@hydra.main(version_base=None, config_path="config", config_name="data_collection/default")
def main(cfg: DictConfig):
    num_robots = cfg.data_collection.get("num_robots", 1)
    print(f"Starting Data Collection: {cfg.data_collection.num_samples} samples using {num_robots} virtual robots")
    
    # 1. Setup Params
    env_params = EnvParams()
    env_params.world_x_size = cfg.env.world_x_size
    env_params.world_y_size = cfg.env.world_y_size
    env_params.max_steps = cfg.env.max_steps
    
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

    # --- VECTORIZED STATE ---
    # Robots: [x, y, theta] -> (N, 3)
    # Waypoints: [x, y] -> (N, 2)
    
    def reset_robots(indices):
        """Resets specified robots to random positions"""
        n = len(indices)
        if n == 0: return
        
        # Random Pos
        robots_state[indices, 0] = np.random.rand(n) * env_params.world_x_size
        robots_state[indices, 1] = np.random.rand(n) * env_params.world_y_size
        robots_state[indices, 2] = np.random.rand(n) * 2 * np.pi - np.pi # Theta
        
        # Random Waypoints
        waypoints[indices, 0] = np.random.rand(n) * env_params.world_x_size
        waypoints[indices, 1] = np.random.rand(n) * env_params.world_y_size

    # Global Initialization
    robots_state = np.zeros((num_robots, 3))
    waypoints = np.zeros((num_robots, 2))
    reset_robots(np.arange(num_robots))
    
    # CBF Controllers (One per robot is expensive if it has state, 
    # but CBFController seems lightweight. We instantiate it once and use its vectorized nominal method if possible,
    # or loop. Nominal control is simple math).
    # To simplify, we'll use a simple vectorized P controller here to avoid slow loops.
    
    def vectorized_nominal_control(states, targets):
        """Simple P controller towards vectorized waypoint"""
        # states: (N, 3) [x, y, theta]
        # targets: (N, 2) [x, y]
        
        dx = targets[:, 0] - states[:, 0]
        dy = targets[:, 1] - states[:, 1]
        
        target_angle = np.arctan2(dy, dx)
        current_angle = states[:, 2]
        
        # Angle diff normalization
        angle_diff = (target_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Simple P controller
        w = 2.0 * angle_diff
        v = 2.0 * np.ones_like(w) # Constant velocity or proportional to distance
        
        dist = np.sqrt(dx**2 + dy**2)
        v = np.minimum(v, dist) # Decelerate on arrival
        
        # Clip
        actions = np.stack([v, w], axis=1)
        actions = np.clip(actions, robot_params.action_min, robot_params.action_max)
        return actions

    # Determine Mode
    playback_path = cfg.data_collection.get("playback_path", None)
    playback_mode = playback_path is not None
    
    env = None
    if playback_mode:
        playback_path = hydra.utils.to_absolute_path(playback_path)
        print(f"Mode: PLAYBACK from {playback_path}")
        env_params.playback_path = playback_path
        # Placeholder smoke params for init (ignored by PlaybackSmoke)
        dummy_blobs = [SmokeBlobParams(0,0,0,0)]
        smoke_params = DynamicSmokeParams(
             x_size=env_params.world_x_size, 
             y_size=env_params.world_y_size, 
             smoke_blob_params=dummy_blobs, 
             resolution=cfg.data_collection.smoke.resolution
        )
        env = DynamicSmokeEnv(env_params, robot_params, smoke_params)
        
        # Check for data repetition
        available_eps = env.smoke_simulator.num_episodes
        # Est. episodes needed = (Total Samples / Robots) / StepsPerEpisode
        # This is approximate because robots might reset or finish early, but gives order of magnitude.
        samples_per_ep = env_params.max_steps * num_robots
        needed_eps = np.ceil(cfg.data_collection.num_samples / samples_per_ep)
        
        if needed_eps > available_eps:
             print(f"\n{'='*60}")
             print(f"WARNING: DATA REPETITION DETECTED")
             print(f"Requested ~{int(needed_eps)} episodes for {cfg.data_collection.num_samples} samples.")
             print(f"Playback file only contains {available_eps} episodes.")
             print(f"Some smoke data will be repeated!")
             print(f"{'='*60}\n")
    else:
        print("Mode: SIMULATION (Procedural)")

    while datasize < cfg.data_collection.num_samples:
        
        if not playback_mode:
            # 1. Randomize Smoke Env (Procedural)
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

            # Re-create Environment for new params
            if env is not None: env.close()
            env = DynamicSmokeEnv(env_params, robot_params, smoke_params)
            env.reset()
        else:
            # Playback Mode: Just reset to cycle episodes
            env.reset()
        
        # Reset robots at the start of each "smoke episode"
        reset_robots(np.arange(num_robots))
        
        steps = 0
        while steps < env_params.max_steps and datasize < cfg.data_collection.num_samples:
            # A. Compute Actions
            # Use actual action_max from config
            v_max = robot_params.action_max[0]
            w_max = robot_params.action_max[1]
            
            # Vectorized Nominal Controller
            dx = waypoints[:, 0] - robots_state[:, 0]
            dy = waypoints[:, 1] - robots_state[:, 1]
            target_angle = np.arctan2(dy, dx)
            current_angle = robots_state[:, 2]
            angle_diff = (target_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
            
            # P Controller
            w_cmd = 2.0 * angle_diff
            
            # Variable Nominal Velocity:
            # Instead of always going at v_max, vary base target velocity
            # to cover full velocity range, not just 0 and max.
            # Assign a random \"cruise speed\" to each robot when changing waypoint?
            # Or simply vary a bit each step.
            # Better: v_cmd proportional to distance but clipped to random v_target.
            
            # Generate random v_target between 0.5*v_max and v_max for variety
            v_targets = np.random.uniform(0.2 * v_max, v_max, size=num_robots)
            
            # Brake on sharp turns (natural behavior)
            v_targets *= np.clip(1.0 - np.abs(angle_diff) / (np.pi/2), 0.1, 1.0)
            
            dist = np.sqrt(dx**2 + dy**2)
            v_cmd = np.minimum(v_targets, dist) # Decelerate on arrival
            
            nominal_actions = np.stack([v_cmd, w_cmd], axis=1)
            nominal_actions = np.clip(nominal_actions, robot_params.action_min, robot_params.action_max)
            
            # B. Inject Noise
            rand_vals = np.random.rand(num_robots)
            
            # Distribution:
            # 0.00 - 0.30: Scaled Nominal (30%) - Covers intermediate range
            # 0.30 - 0.35: Stop (5%) - Ceros
            # 0.35 - 1.00: Nominal (65%)
            
            mask_scaled = rand_vals < 0.30
            mask_stop = (rand_vals >= 0.30) & (rand_vals < 0.35)
            
            final_actions = nominal_actions.copy()
            
            if np.any(mask_scaled):
                n_scaled = np.sum(mask_scaled)
                scales = np.random.rand(n_scaled, 1) # (N, 1)
                final_actions[mask_scaled] *= scales
                
            if np.any(mask_stop):
                final_actions[mask_stop] = 0.0
                
            # C. Kinematics
            dt = robot_params.dt
            v = final_actions[:, 0]
            w = final_actions[:, 1]
            theta = robots_state[:, 2]
            
            next_robots_state = robots_state.copy()
            next_robots_state[:, 0] += v * np.cos(theta) * dt
            next_robots_state[:, 1] += v * np.sin(theta) * dt
            next_robots_state[:, 2] += w * dt
            
            # D. Simulate Smoke
            env.smoke_simulator.step(dt=dt)
            
            # E. Query Sensors (REAL)
            # Instantiate sensor once outside loop if possible, or use env's.
            # env has self.sensor initialized with sensor_params.
            # DownwardsSensor.read(smoke_func, curr_pos)
            # smoke_func is env.smoke_simulator.get_smoke_density
            
            # Iterate robots to get accurate readings
            # This is slightly slower than pure vectorized, but ensures geometric correctness.
            # With 20 robots it is still very fast.
            
            sensor_readings_batch = []
            sensor_positions_batch = []
            
            # Smoke wrapper function for sensor (expects x, y)
            def get_smoke_wrapper(pos):
                return env.smoke_simulator.get_smoke_density(pos)

            for i in range(num_robots):
                # Robot current position
                r_pos = robots_state[i, :2]
                # Read sensor using real DownwardsSensor logic
                # env.sensor is already configured
                readings = env.sensor.read(get_smoke_wrapper, curr_pos=r_pos)
                
                sensor_readings_batch.append(readings["sensor_readings"])
                sensor_positions_batch.append(readings["sensor_position_readings"])
                
            # F. Save to Temporary Buffers
            
            # Check bounds and waypoints
            dones = np.zeros(num_robots, dtype=bool)
            
            # Check bounds
            out_of_bounds = (next_robots_state[:, 0] < 0) | (next_robots_state[:, 0] > env_params.world_x_size) | \
                            (next_robots_state[:, 1] < 0) | (next_robots_state[:, 1] > env_params.world_y_size)
            
            dones = dones | out_of_bounds
            
            # Check waypoints reached
            dists = np.linalg.norm(robots_state[:, :2] - waypoints, axis=1)
            reached = dists < 2.0
            
            # Update waypoints for reached
            if np.any(reached):
                n_reached = np.sum(reached)
                waypoints[reached, 0] = np.random.rand(n_reached) * env_params.world_x_size
                waypoints[reached, 1] = np.random.rand(n_reached) * env_params.world_y_size
            
            # Initialize temp buffers if not exist
            if 'temp_buffers' not in locals():
                temp_buffers = [[] for _ in range(num_robots)]

            for i in range(num_robots):
                # Save to temporary buffer for robot i
                points_i = sensor_positions_batch[i]
                vals_i = sensor_readings_batch[i]
                
                sample = {
                    'state': robots_state[i].copy(),
                    'actions': final_actions[i].copy(),
                    'smoke_values': vals_i,
                    'smoke_value_positions': points_i,
                    'done': dones[i]
                }
                temp_buffers[i].append(sample)
                
                # If the robot finishes (out of bounds), flush its buffer
                if dones[i]:
                    for s in temp_buffers[i]:
                        if datasize >= cfg.data_collection.num_samples: break
                        replay_buffer.add(**s)
                        datasize += 1
                        pbar.update(1)
                    temp_buffers[i] = [] # Clear buffer
            
            # Update States
            robots_state = next_robots_state
            
            # Reset robots out of bounds
            if np.any(out_of_bounds):
                reset_robots(np.where(out_of_bounds)[0])
                
            steps += 1
        
        # At the end of the smoke episode (max_steps reached), flush all remaining buffers
        if 'temp_buffers' in locals():
            for i in range(num_robots):
                for s in temp_buffers[i]:
                    if datasize >= cfg.data_collection.num_samples: break
                    replay_buffer.add(**s)
                    datasize += 1
                    pbar.update(1)
                temp_buffers[i] = []
        
        if not playback_mode:
            env.close()
    
    pbar.close()
    if playback_mode and env is not None:
        env.close()
    
    # Save
    os.makedirs(cfg.data_collection.output_dir, exist_ok=True)
    filename = cfg.data_collection.get("output_filename", "replay_buffer.npz")
    output_path = os.path.join(cfg.data_collection.output_dir, filename)
    replay_buffer.save_to_file(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()
