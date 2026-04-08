import os
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from envs.simulator.playback_schema import SmokeDataSchema
from envs.simulator.smoke import BlobParams, Smoke, SmokeParams


@hydra.main(version_base=None, config_path="configs/experiments", config_name="playback_generation")
def main(cfg: DictConfig):
    # Parameters from config
    num_episodes = cfg.get("num_episodes", 10)
    episode_steps = cfg.get("episode_steps", 100)
    output_path = cfg.get("output_path", "data/playback_smoke_v1.npz")

    # Grid settings
    x_size = cfg.get("x_size", 30.0)
    y_size = cfg.get("y_size", 30.0)
    resolution = cfg.get("resolution", 0.2)
    dt = cfg.get("dt", 0.1)

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Storage dimensions
    H = int(y_size / resolution)
    W = int(x_size / resolution)
    print(f"Generating grid: {H}x{W} ({y_size}m x {x_size}m at {resolution}m resolution)")

    # Storage array
    all_data = np.zeros((num_episodes, episode_steps, H, W), dtype=np.float32)

    start_time = time.time()

    for ep in tqdm(range(num_episodes), desc="Episodes"):
        # 1. Randomize blobs for this episode based on config ranges
        num_blobs = np.random.randint(cfg.num_blobs_range[0], cfg.num_blobs_range[1] + 1)
        episode_blobs = []

        for _ in range(num_blobs):
            for _ in range(100):  # Attempts to find valid non-overlapping position
                x_c = np.random.uniform(2.0, x_size - 2.0)
                y_c = np.random.uniform(2.0, y_size - 2.0)

                # Enforce min distance between centers if requested
                min_dist = cfg.get("blob_min_dist", 5.0)
                valid_pos = True
                for blob in episode_blobs:
                    dist = np.sqrt((x_c - blob.x_pos) ** 2 + (y_c - blob.y_pos) ** 2)
                    if dist < min_dist:
                        valid_pos = False
                        break

                if valid_pos:
                    episode_blobs.append(
                        BlobParams(
                            x_pos=x_c,
                            y_pos=y_c,
                            intensity=float(cfg.get("blob_intensity", 1.0)),
                            spread_rate=np.random.uniform(
                                cfg.blob_spread_range[0], cfg.blob_spread_range[1]
                            ),
                        )
                    )
                    break

        # 2. Setup the Simulator with randomized blobs and config parameters
        params = SmokeParams(
            x_size=x_size,
            y_size=y_size,
            smoke_blob_params=episode_blobs,
            resolution=resolution,
            average_wind_speed=float(cfg.get("wind_speed", 8.0)),
            smoke_emission_rate=float(cfg.get("emission_rate", 2.0)),
            smoke_diffusion_rate=float(cfg.get("diffusion_rate", 2.0)),
            smoke_decay_rate=float(cfg.get("decay_rate", 1.5)),
            buoyancy_factor=float(cfg.get("buoyancy", 1.2)),
        )
        sim = Smoke(params)

        # 3. Simulate episode steps and record map
        for step in range(episode_steps):
            all_data[ep, step] = sim.get_smoke_map()
            sim.step(dt=dt)

    print(f"Generation complete in {time.time() - start_time:.2f}s")
    print(f"Saving to {output_path} (Hugging Face Dataset format)...")

    # Create dataset dictionary
    # Instead of one big 4D array, we save per episode for better HF handling
    dataset_dict = {
        SmokeDataSchema.SMOKE_DATA: [all_data[ep].tolist() for ep in range(num_episodes)],
        SmokeDataSchema.X_SIZE: [float(x_size)] * num_episodes,
        SmokeDataSchema.Y_SIZE: [float(y_size)] * num_episodes,
        SmokeDataSchema.RESOLUTION: [float(resolution)] * num_episodes,
        SmokeDataSchema.DT: [float(dt)] * num_episodes,
        "episode_id": list(range(num_episodes)),
    }

    from datasets import Dataset

    ds = Dataset.from_dict(dataset_dict)

    # Add metadata to the dataset info if possible, or just as a separate file
    # For simplicity in this environment, we save to disk folder
    ds.save_to_disk(output_path)

    print(f"Saved dataset to {output_path}. Features: {ds.features}")
    print(f"Data structure: Each row is one episode with {episode_steps} steps of {H}x{W} maps.")


if __name__ == "__main__":
    main()
