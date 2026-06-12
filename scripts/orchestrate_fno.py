import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional
from prefect import flow, task


@task(name="Collect Physics Smoke Data")
def collect_data(data_path: str, num_episodes: int = 500, episode_steps: int = 100):
    print(f"Starting data collection to {data_path}...")
    cmd = [
        sys.executable,
        "src/0_data_playback_physics_collection.py",
        f"output_path={data_path}",
        f"num_episodes={num_episodes}",
        f"episode_steps={episode_steps}",
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Data collection completed successfully.")


@task(name="Train FNO with NLL Loss")
def train_fno_nll(data_path: str, max_epochs: int = 250, max_samples: int = None, downsample_factor: int = 1):
    print("Starting FNO training with NLL Loss...")
    cmd = [
        sys.executable,
        "src/training/train_fno.py",
        "training.loss.name=nll",
        f"training.data.data_path={data_path}",
        f"training.optimizer.max_epochs={max_epochs}",
        f"training.data.downsample_factor={downsample_factor}",
    ]
    if max_samples is not None:
        cmd.append(f"training.data.max_samples={max_samples}")

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("FNO NLL training completed successfully.")


@task(name="Train FNO with Energy Score Loss")
def train_fno_es(data_path: str, max_epochs: int = 250, max_samples: int = None, downsample_factor: int = 1):
    print("Starting FNO training with Energy Score Loss...")
    cmd = [
        sys.executable,
        "src/training/train_fno.py",
        "training.loss.name=energy_score",
        f"training.data.data_path={data_path}",
        f"training.optimizer.max_epochs={max_epochs}",
        f"training.data.downsample_factor={downsample_factor}",
    ]
    if max_samples is not None:
        cmd.append(f"training.data.max_samples={max_samples}")

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("FNO Energy Score training completed successfully.")


@flow(name="Probabilistic FNO Pipeline")
def fno_pipeline(
    dry_run: bool = False,
    data_path: Optional[str] = None,
    skip_collection: bool = False,
    num_episodes: int = 500,
    episode_steps: int = 100,
    max_epochs: int = 250,
    downsample_factor: int = 1,
):
    # Determine the dataset path
    if data_path is None:
        data_path = "data/physics_smoke_dry_run" if dry_run else "data/physics_smoke"

    # Determine flow parameters based on dry-run flag
    if dry_run:
        print("!!! DRY-RUN MODE ACTIVE !!!")
        num_episodes = 2
        episode_steps = 40
        max_epochs = 1
        max_samples = 2
    else:
        max_samples = None

    # Step 1: Collect Data
    if not skip_collection:
        collect_data(data_path, num_episodes=num_episodes, episode_steps=episode_steps)
    else:
        print(f"Skipping data collection. Training will use dataset at: {data_path}")

    # Step 2: Train with NLL Loss
    train_fno_nll(data_path, max_epochs=max_epochs, max_samples=max_samples, downsample_factor=downsample_factor)

    # Step 3: Train with Energy Score Loss
    train_fno_es(data_path, max_epochs=max_epochs, max_samples=max_samples, downsample_factor=downsample_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orchestrate FNO Data Collection and Training using Prefect."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a quick dry-run with tiny data and epoch count.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom output and training dataset path.",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection and train directly on the specified data path.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=500,
        help="Number of simulation episodes to collect (default: 500).",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=100,
        help="Number of simulation steps per episode (default: 100).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=250,
        help="Maximum epochs to train the FNO models (default: 250).",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help="Spatial downsample factor applied to the smoke grid before training (default: 1).",
    )

    args = parser.parse_args()
    fno_pipeline(
        dry_run=args.dry_run,
        data_path=args.data_path,
        skip_collection=args.skip_collection,
        num_episodes=args.num_episodes,
        episode_steps=args.episode_steps,
        max_epochs=args.max_epochs,
        downsample_factor=args.downsample_factor,
    )
