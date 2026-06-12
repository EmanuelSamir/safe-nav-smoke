import os
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from env.simulator.playback_schema import SmokeDataSchema
from env.simulator.smoke import BlobParams, Smoke, SmokeParams


@hydra.main(
    version_base=None, config_path="../configs/data_collection", config_name="playback_physics"
)
def main(cfg: DictConfig):
    # Parameters from config
    num_episodes = cfg.num_episodes
    episode_steps = cfg.episode_steps
    output_path = cfg.output_path

    # Grid settings
    x_size = cfg.x_size
    y_size = cfg.y_size
    resolution = cfg.resolution
    dt = cfg.dt

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Storage dimensions
    H = int(y_size / resolution)
    W = int(x_size / resolution)
    print(f"Generating grid: {H}x{W} ({y_size}m x {x_size}m at {resolution}m resolution)")

    start_time = time.time()

    def episode_generator():
        for ep in tqdm(range(num_episodes), desc="Episodes"):
            # 1. Randomize blobs for this episode based on config ranges
            num_blobs = np.random.randint(cfg.num_blobs_range[0], cfg.num_blobs_range[1] + 1)
            episode_blobs = []

            for _ in range(num_blobs):
                for _ in range(100):  # Attempts to find valid non-overlapping position
                    x_c = np.random.uniform(2.0, x_size - 2.0)
                    y_c = np.random.uniform(2.0, y_size - 2.0)

                    # Enforce min distance between centers if requested
                    min_dist = cfg.blob_min_dist
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
                                intensity=float(cfg.blob_intensity),
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
                average_wind_speed=float(cfg.wind_speed),
                smoke_emission_rate=float(cfg.emission_rate),
                smoke_diffusion_rate=float(cfg.diffusion_rate),
                smoke_decay_rate=float(cfg.decay_rate),
                buoyancy_factor=float(cfg.buoyancy),
            )
            sim = Smoke(params)

            # 3. Simulate episode steps and record map
            episode_data = np.zeros((episode_steps, H, W), dtype=np.float32)
            for step in range(episode_steps):
                episode_data[step] = sim.get_smoke_map()
                sim.step(dt=dt)

            yield {
                SmokeDataSchema.SMOKE_DATA: episode_data.tolist(),
                SmokeDataSchema.X_SIZE: float(x_size),
                SmokeDataSchema.Y_SIZE: float(y_size),
                SmokeDataSchema.RESOLUTION: float(resolution),
                SmokeDataSchema.DT: float(dt),
                "episode_id": ep,
            }

    print(f"Generating data using streaming generator (Hugging Face Dataset format)...")

    from datasets import Dataset

    ds = Dataset.from_generator(episode_generator)

    print(f"Generation complete in {time.time() - start_time:.2f}s. Saving to {output_path}...")
    ds.save_to_disk(output_path)

    print(f"Saved dataset to {output_path}. Features: {ds.features}")
    print(f"Data structure: Each row is one episode with {episode_steps} steps of {H}x{W} maps.")


if __name__ == "__main__":
    main()
