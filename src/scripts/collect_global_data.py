import hydra
from omegaconf import DictConfig
import numpy as np
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from simulator.dynamic_smoke import DynamicSmoke, DynamicSmokeParams, SmokeBlobParams
from simulator.playback_smoke import PlaybackSmoke, PlaybackSmokeParams
from envs.replay_buffer import GenericReplayBuffer

@hydra.main(version_base=None, config_path="../../config", config_name="data_collection/default")
def main(cfg: DictConfig):
    print(f"Starting GLOBAL Data Collection")
    
    # Parameters
    max_steps = cfg.env.max_steps
    num_samples = cfg.data_collection.num_samples
    num_episodes = int(np.ceil(num_samples / max_steps))
    
    output_dir = cfg.data_collection.output_dir
    os.makedirs(output_dir, exist_ok=True)
    filename = cfg.data_collection.get("output_filename", "global_data.npz")
    output_path = os.path.join(output_dir, filename)
    
    playback_path = cfg.data_collection.get("playback_path", None)
    playback_mode = playback_path is not None
    
    x_size = cfg.env.world_x_size
    y_size = cfg.env.world_y_size
    res = cfg.data_collection.smoke.resolution
    
    # Initialize Replay Buffer
    # Matching structure of collect_local_data.py
    # actions and state will be dummy (zeros)
    replay_buffer = GenericReplayBuffer(
        buffer_size=num_samples, 
        data_keys=['actions', 'state', 'smoke_values', 'done']
    )
    
    if playback_mode:
        playback_path = hydra.utils.to_absolute_path(playback_path)
        print(f"Mode: PLAYBACK from {playback_path}")
        sim = PlaybackSmoke(PlaybackSmokeParams(data_path=playback_path))
        # Use simulator dims
        H, W = sim.data.shape[2], sim.data.shape[3]
        sim_res = sim.resolution
    else:
        print("Mode: SIMULATION (Procedural)")
        H = int(y_size / res)
        W = int(x_size / res)
        sim_res = res
        
    print(f"Collecting {num_episodes} episodes of {max_steps} steps. Grid: {H}x{W}")
    
    # Generate coordinate grid once
    # Coordinates for the centers of the grid cells? Or nodes? 
    # Usually linspace 0 to x_size.
    xs = np.linspace(0, x_size, W)
    ys = np.linspace(0, y_size, H)
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    # Shape (H, W). Flatten to (N, 2)
    grid_coords = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)
    
    datasize = 0
    pbar = tqdm(total=num_samples)
    
    for ep in range(num_episodes):
        if datasize >= num_samples:
            break

        if playback_mode:
            # PlaybackSmoke reset uses episode_idx to select/cycle episodes
            sim.reset(episode_idx=ep)
        else:
            # Create new simulation with random params
            smoke_blobs = []
            for _ in range(cfg.data_collection.num_smoke_blobs):
                for _ in range(100): # Attempts to find a valid position
                    x_c = x_size * np.random.rand()
                    y_c = y_size * np.random.rand()
                    
                    # Check distance to existing blobs
                    valid_pos = True
                    for blob in smoke_blobs:
                        dist = np.sqrt((x_c - blob.x_pos)**2 + (y_c - blob.y_pos)**2)
                        if dist < 4.0:
                            valid_pos = False
                            break
                    
                    if valid_pos:
                        smoke_blobs.append(
                            SmokeBlobParams(
                                x_pos=x_c,
                                y_pos=y_c,
                                intensity=cfg.data_collection.smoke.intensity,
                                spread_rate=np.random.uniform(2.0, 4.0)
                            )
                        )
                        break
            
            params = DynamicSmokeParams(
                x_size=x_size, 
                y_size=y_size, 
                smoke_blob_params=smoke_blobs, 
                resolution=res,
                average_wind_speed=10.0,
                smoke_emission_rate=2.0,
                smoke_diffusion_rate=2.0,
                smoke_decay_rate=1.5,
                buoyancy_factor=1.2
            )
            sim = DynamicSmoke(params)
        
        for step in range(max_steps):
            if datasize >= num_samples:
                break

            # Capture state
            # Global grid: (H, W) or similar
            smoke_grid = sim.get_smoke_map()
            
            # Prepare sample data
            # Flatten smoke grid corresponding to grid_coords
            smoke_values = smoke_grid.flatten().astype(np.float32)
            
            # Dummy actions/state
            # Assuming action dim 2, state dim 3 (robot pose) from standard config
            actions = np.zeros(2, dtype=np.float32)
            state = np.zeros(3, dtype=np.float32)
            
            done = (step == max_steps - 1)
            
            sample = {
                'actions': actions,
                'state': state,
                'smoke_values': smoke_values,
                'done': float(done)
            }
            
            replay_buffer.add(**sample)
            datasize += 1
            pbar.update(1)
            
            # Step simulation
            sim.step()
            
    pbar.close()
    
    print(f"Saving to {output_path}...")
    replay_buffer.save_to_file(output_path, smoke_value_positions=grid_coords)
    print(f"Done. Size: {os.path.getsize(output_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
