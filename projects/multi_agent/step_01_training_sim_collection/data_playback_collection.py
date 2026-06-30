import os
import sys
import time

import matplotlib
import numpy as np
import yaml
from tqdm import tqdm

# It's better practice to run the script via `python -m` from the root, 
# but if this is strictly needed, keep it near the top before local imports.
sys.path.append(os.getcwd())

from projects.single_agent.step_01_training_sim_collection.schema import DataCollectionConfig
from src.env.simulator.smoke_data_schema import SmokeDataSchema
from src.env.simulator.schemas import BlobConfig, SmokeConfig
from src.env.simulator.smoke import Smoke

import argparse

def load_config() -> DataCollectionConfig:
    """Loads the local YAML configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_config.yaml", help="Path to YAML config file")
    args, _ = parser.parse_known_args()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f) or {}
    # Use Pydantic's model_validate for parsing dicts (idiomatic Pydantic v2)
    return DataCollectionConfig.model_validate(yaml_data)

# Initialize configuration
cfg = load_config()

# Configure matplotlib backend before importing pyplot
if not cfg.test:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main():
    # Parameters from config
    test_mode = cfg.test
    num_episodes = cfg.num_episodes
    episode_steps = cfg.episode_steps
    output_path = cfg.output_path

    # Grid settings
    x_size = cfg.smoke_params.x_size
    y_size = cfg.smoke_params.y_size
    resolution = cfg.smoke_params.resolution
    dt = cfg.smoke_params.dt

    def create_randomized_sim():
        num_blobs = np.random.randint(cfg.num_blobs_range[0], cfg.num_blobs_range[1] + 1)
        episode_blobs = []

        for _ in range(num_blobs):
            for _ in range(cfg.max_spawn_attempts):
                x_c = np.random.uniform(cfg.spawn_margin, x_size - cfg.spawn_margin)
                y_c = np.random.uniform(cfg.spawn_margin, y_size - cfg.spawn_margin)

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
                        BlobConfig(
                            x_pos=x_c,
                            y_pos=y_c,
                            intensity=float(cfg.blob_intensity),
                            spread_rate=np.random.uniform(
                                cfg.blob_spread_range[0], cfg.blob_spread_range[1]
                            ),
                        )
                    )
                    break

        # Clone the default smoke parameters from config and attach the randomized blobs
        params = cfg.smoke_params.model_copy(deep=True)
        params.blobs = episode_blobs
        return Smoke(cfg=params)

    if test_mode:
        fig, ax = plt.subplots(figsize=(8, 6))
        print("Running in TEST mode: Visualizing episode...")

        sim = create_randomized_sim()

        for step in range(episode_steps):
            sim.plot_smoke_map(fig=fig, ax=ax)
            plt.pause(0.01)
            sim.step(dt=dt)

        print("Test mode complete. No data saved.")
        plt.show()
        return

    # Prevent overwriting existing data
    assert not os.path.exists(output_path), f"Output path '{output_path}' already exists! Stopping to prevent overwrite."

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Storage dimensions
    H = int(y_size / resolution)
    W = int(x_size / resolution)
    print(f"Generating grid: {H}x{W} ({y_size}m x {x_size}m at {resolution}m resolution)")

    start_time = time.time()

    def episode_generator():
        for ep in tqdm(range(num_episodes), desc="Episodes"):
            # 1. Setup the Simulator with randomized blobs and config parameters
            sim = create_randomized_sim()

            # 2. Simulate episode steps and record map
            episode_data = np.zeros((episode_steps, H, W), dtype=np.float32)
            for step in range(episode_steps):
                episode_data[step] = sim.get_smoke_map()
                sim.step(dt=dt)

            yield {
                SmokeDataSchema.SMOKE_DATA: episode_data,
                SmokeDataSchema.X_SIZE: float(x_size),
                SmokeDataSchema.Y_SIZE: float(y_size),
                SmokeDataSchema.RESOLUTION: float(resolution),
                SmokeDataSchema.DT: float(dt),
                "episode_id": ep,
            }

    print("Generating data using streaming generator (Hugging Face Dataset format)...")

    import datasets
    from datasets import Dataset

    features = datasets.Features(
        {
            SmokeDataSchema.SMOKE_DATA: datasets.Array3D(
                shape=(episode_steps, H, W), dtype="float32"
            ),
            SmokeDataSchema.X_SIZE: datasets.Value("float32"),
            SmokeDataSchema.Y_SIZE: datasets.Value("float32"),
            SmokeDataSchema.RESOLUTION: datasets.Value("float32"),
            SmokeDataSchema.DT: datasets.Value("float32"),
            "episode_id": datasets.Value("int32"),
        }
    )

    ds = Dataset.from_generator(episode_generator, features=features, writer_batch_size=cfg.writer_batch_size)

    print(f"Generation complete in {time.time() - start_time:.2f}s. Saving to {output_path}...")
    ds.save_to_disk(output_path)

    print(f"Saved dataset to {output_path}. Features: {ds.features}")
    print(f"Data structure: Each row is one episode with {episode_steps} steps of {H}x{W} maps.")


if __name__ == "__main__":
    main()
