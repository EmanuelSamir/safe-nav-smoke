import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from envs.simulator.playback_schema import SmokeDataSchema
from envs.simulator.smoke import BlobParams, Smoke, SmokeParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (visualize, do not save)"
    )
    args = parser.parse_args()

    # Parameters
    NUM_EPISODES = 1 if args.test else 100
    EPISODE_STEPS = 200
    OUTPUT_PATH = "data/planning_smoke_200_steps_100_episodes"
    dt = 0.1

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

    tailored_blobs = {
        "case_1": {
            "x_pos": [10.0, 10.0, 10.0, 20.0, 20.0],
            "y_pos": [3.0, 10.0, 17.0, 6.0, 14.0],
        },
        "case_2": {
            "x_pos": [20.0, 20.0, 20.0, 10.0, 10.0],
            "y_pos": [3.0, 10.0, 17.0, 6.0, 14.0],
        },
        "case_3": {
            "x_pos": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "y_pos": [3.0, 8.0, 13.0, 7.0, 12.0, 17.0],
        },
        "case_4": {
            "x_pos": [20.0, 20.0, 20.0, 10.0, 10.0, 10.0],
            "y_pos": [3.0, 8.0, 13.0, 7.0, 12.0, 17.0],
        },
        "case_5": {
            "x_pos": [8.0, 8.0, 15.0, 22.0, 22.0],
            "y_pos": [4.0, 16.0, 10.0, 4.0, 16.0],
        },
        "case_6": {
            "x_pos": [8.0, 15.0, 15.0, 15.0, 22.0],
            "y_pos": [10.0, 4.0, 10.0, 16.0, 10.0],
        },
    }

    for ep in tqdm(range(NUM_EPISODES), desc="Episodes"):
        # Randomize blobs for this episode
        case_idx = np.random.randint(1, len(tailored_blobs) + 1)
        case_blobs = tailored_blobs[f"case_{case_idx}"]
        num_blobs = len(case_blobs["x_pos"])
        episode_blobs = []
        for i in range(num_blobs):
            spread_rate = np.random.uniform(1.0, 3.0)
            episode_blobs.append(
                BlobParams(
                    x_pos=case_blobs["x_pos"][i],
                    y_pos=case_blobs["y_pos"][i],
                    intensity=1.0,
                    spread_rate=spread_rate,
                )
            )

        # Create new simulator instance for this episode to bake in the new blobs
        params = SmokeParams(
            x_size=x_size,
            y_size=y_size,
            smoke_blob_params=episode_blobs,
            resolution=resolution,
            average_wind_speed=8.0,
            smoke_emission_rate=2.0,
            smoke_diffusion_rate=2.0,
            smoke_decay_rate=1.5,
            buoyancy_factor=1.2,
        )
        sim = Smoke(params)

        for step in range(EPISODE_STEPS):
            # Record current state
            if not args.test:
                all_data[ep, step] = sim.get_smoke_map()

            # Visualization in test mode
            if args.test:
                sim.plot_smoke_map(fig=fig, ax=ax)
                plt.pause(0.01)

            # Advance simulation
            sim.step(dt=dt)

    print(f"Generation complete in {time.time() - start_time:.2f}s")

    if args.test:
        print("Test mode complete. No data saved.")
        plt.show()
        return

    print(f"Saving to {OUTPUT_PATH} (Hugging Face Dataset format)...")

    # Create dataset dictionary
    # Instead of one big 4D array, we save per episode for better HF handling
    dataset_dict = {
        SmokeDataSchema.SMOKE_DATA: [all_data[ep].tolist() for ep in range(NUM_EPISODES)],
        SmokeDataSchema.X_SIZE: [float(x_size)] * NUM_EPISODES,
        SmokeDataSchema.Y_SIZE: [float(y_size)] * NUM_EPISODES,
        SmokeDataSchema.RESOLUTION: [float(resolution)] * NUM_EPISODES,
        SmokeDataSchema.DT: [float(dt)] * NUM_EPISODES,
        "episode_id": list(range(NUM_EPISODES)),
    }

    from datasets import Dataset

    ds = Dataset.from_dict(dataset_dict)
    ds.save_to_disk(OUTPUT_PATH)

    print(f"Saved dataset to {OUTPUT_PATH}. Features: {ds.features}")
    print(f"Data structure: Each row is one episode with {EPISODE_STEPS} steps of {H}x{W} maps.")


if __name__ == "__main__":
    main()
