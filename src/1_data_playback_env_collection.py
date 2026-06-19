import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from omegaconf import DictConfig, OmegaConf

# Early config parsing to configure matplotlib backend before any other imports
cli_args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
cli_cfg = OmegaConf.from_cli(cli_args)

# Load base config
config_name = "playback_env"
config_path = os.path.join(
    os.path.dirname(__file__), "../configs/data_collection", f"{config_name}.yaml"
)
base_cfg = OmegaConf.load(config_path)
merged_cfg = OmegaConf.merge(base_cfg, cli_cfg)
test_mode = merged_cfg.get("test", False)

import matplotlib

if not test_mode:
    matplotlib.use("Agg")
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from env.simulator.playback_schema import SmokeDataSchema
from env.simulator.smoke import BlobParams, Smoke, SmokeParams


@hydra.main(version_base=None, config_path="../configs/data_collection", config_name="playback_env")
def main(cfg: DictConfig):
    # Parameters from config
    test_mode = cfg.test
    num_episodes = 1 if test_mode else cfg.num_episodes
    episode_steps = cfg.episode_steps
    output_path = cfg.output_path
    dt = cfg.dt

    # Grid settings
    x_size = cfg.x_size
    y_size = cfg.y_size
    resolution = cfg.resolution

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Storage
    # Grid shape will be (y_res, x_res)
    H = int(y_size / resolution)
    W = int(x_size / resolution)
    print(f"Grid shape: {H}x{W}")

    start_time = time.time()

    tailored_blobs = {
        "case_1": {
            "x_pos": [11.5, 11.5, 11.5, 23.5, 23.5],
            "y_pos": [7.5, 15.0, 22.5, 11.25, 18.75],
        },
        "case_2": {
            "x_pos": [23.5, 23.5, 23.5, 11.5, 11.5],
            "y_pos": [7.5, 15.0, 22.5, 11.25, 18.75],
        },
        "case_3": {
            "x_pos": [11.5, 11.5, 11.5, 23.5, 23.5, 23.5],
            "y_pos": [7.5, 15.0, 22.5, 7.5, 15.0, 22.5],
        },
        "case_4": {
            "x_pos": [23.5, 23.5, 23.5, 11.5, 11.5, 11.5],
            "y_pos": [7.5, 15.0, 22.5, 7.5, 15.0, 22.5],
        },
        "case_5": {
            "x_pos": [10.5, 10.5, 17.5, 24.5, 24.5],
            "y_pos": [7.5, 22.5, 15.0, 7.5, 22.5],
        },
        "case_6": {
            "x_pos": [10.5, 17.5, 17.5, 17.5, 24.5],
            "y_pos": [15.0, 7.5, 15.0, 22.5, 15.0],
        },
    }

    if test_mode:
        fig, ax = plt.subplots(figsize=(8, 6))
        print("Running in TEST mode: Visualizing episodes...")

        case_idx = np.random.randint(1, len(tailored_blobs) + 1)
        case_blobs = tailored_blobs[f"case_{case_idx}"]
        num_blobs = len(case_blobs["x_pos"])
        episode_blobs = []
        for i in range(num_blobs):
            spread_rate = np.random.uniform(1.5, 3.0)
            episode_blobs.append(
                BlobParams(
                    x_pos=case_blobs["x_pos"][i],
                    y_pos=case_blobs["y_pos"][i],
                    intensity=1.0,
                    spread_rate=spread_rate,
                )
            )

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

        for step in range(episode_steps):
            sim.plot_smoke_map(fig=fig, ax=ax)
            plt.pause(0.01)
            sim.step(dt=dt)

        print("Test mode complete. No data saved.")
        plt.show()
        return

    def episode_generator():
        for ep in tqdm(range(num_episodes), desc="Episodes"):
            # Randomize blobs for this episode
            case_idx = np.random.randint(1, len(tailored_blobs) + 1)
            case_blobs = tailored_blobs[f"case_{case_idx}"]
            num_blobs = len(case_blobs["x_pos"])
            episode_blobs = []
            for i in range(num_blobs):
                spread_rate = np.random.uniform(1.5, 3.0)
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
                average_wind_speed=float(cfg.wind_speed),
                smoke_emission_rate=float(cfg.emission_rate),
                smoke_diffusion_rate=float(cfg.diffusion_rate),
                smoke_decay_rate=float(cfg.decay_rate),
                buoyancy_factor=float(cfg.buoyancy),
            )
            sim = Smoke(params)

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

    ds = Dataset.from_generator(episode_generator, features=features, writer_batch_size=50)

    print(f"Generation complete in {time.time() - start_time:.2f}s. Saving to {output_path}...")
    ds.save_to_disk(output_path)

    print(f"Saved dataset to {output_path}. Features: {ds.features}")
    print(f"Data structure: Each row is one episode with {episode_steps} steps of {H}x{W} maps.")


if __name__ == "__main__":
    main()
