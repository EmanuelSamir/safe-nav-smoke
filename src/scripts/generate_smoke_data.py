import numpy as np
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from simulator.dynamic_smoke import DynamicSmoke, DynamicSmokeParams, SmokeBlobParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode (visualize, do not save)")
    args = parser.parse_args()

    # Parameters
    NUM_EPISODES = 3 if args.test else 400
    EPISODE_STEPS = 100
    OUTPUT_PATH = "data/global_source_400_100_2nd.npz"
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Smoke Simulation Parameters
    # Using 50x50 world with resolution 1.0 results in 50x50 grid
    x_size = 30
    y_size = 20
    resolution = 0.2
    
    # Storage
    # Grid shape will be (y_res, x_res)
    H = int(y_size / resolution)
    W = int(x_size / resolution)
    print(f"Grid shape: {H}x{W}")
    
    # Using float16 to save space if needed, or float32. 
    # Smoke values are likely small floats.
    all_data = np.zeros((NUM_EPISODES, EPISODE_STEPS, H, W), dtype=np.float32)
    
    print(f"Generating {NUM_EPISODES} episodes of {EPISODE_STEPS} steps...")
    
    if args.test:
        fig, ax = plt.subplots(figsize=(8, 6))
        print("Running in TEST mode: Visualizing episodes...")

    start_time = time.time()
    
    for ep in tqdm(range(NUM_EPISODES), desc="Episodes"):
        # Randomize blobs for this episode
        num_blobs = np.random.randint(4, 8)  # Random number of blobs
        episode_blobs = []
        for _ in range(num_blobs):
            for _ in range(100): # Attempts to find a valid position
                x_c = np.random.uniform(4, x_size - 4)
                y_c = np.random.uniform(4, y_size - 4)
                
                # Check distance to existing blobs
                valid_pos = True
                for blob in episode_blobs:
                    dist = np.sqrt((x_c - blob.x_pos)**2 + (y_c - blob.y_pos)**2)
                    if dist < 5.0:
                        valid_pos = False
                        break
                
                if valid_pos:
                    episode_blobs.append(SmokeBlobParams(
                        x_pos=x_c,
                        y_pos=y_c,
                        intensity=1.0,
                        spread_rate=np.random.uniform(1.0, 3.0)
                    ))
                    break
        
        # Create new simulator instance for this episode to bake in the new blobs
        params = DynamicSmokeParams(
            x_size=x_size,
            y_size=y_size,
            smoke_blob_params=episode_blobs,
            resolution=resolution,
            average_wind_speed=8.0,
            smoke_emission_rate=2.0,
            smoke_diffusion_rate=2.0,
            smoke_decay_rate=1.5,
            buoyancy_factor=1.2
        )
        sim = DynamicSmoke(params)
        
        for step in range(EPISODE_STEPS):
            # Record current state
            if not args.test:
                all_data[ep, step] = sim.get_smoke_map()
            
            # Visualization in test mode
            if args.test:
                sim.plot_smoke_map(fig=fig, ax=ax)
                plt.pause(0.01)

            # Advance simulation
            sim.step()
            
    print(f"Generation complete in {time.time() - start_time:.2f}s")
    
    if args.test:
        print("Test mode complete. No data saved.")
        plt.show()
        return

    print(f"Saving to {OUTPUT_PATH}...")
    
    # Save compressed
    np.savez_compressed(
        OUTPUT_PATH,
        smoke_data=all_data,
        x_size=x_size,
        y_size=y_size,
        resolution=resolution,
        dt=0.1 # standard dt
    )
    
    print(f"Saved {(os.path.getsize(OUTPUT_PATH) / 1024 / 1024):.2f} MB")

if __name__ == "__main__":
    main()
