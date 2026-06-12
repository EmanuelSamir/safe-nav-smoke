import os
import sys

import numpy as np
from datasets import load_from_disk

# Add project root to path
sys.path.append(os.getcwd())


def analyze_dataset(path):
    if not os.path.exists(path):
        return None

    analysis = {
        "name": os.path.basename(path),
        "type": "Unknown",
        "num_episodes": 0,
        "episode_steps": 0,
        "grid_shape": "N/A",
        "mean_value": 0.0,
        "max_value": 0.0,
        "nonzero_fraction": 0.0,
        "features": [],
        "errors": None,
    }

    try:
        if os.path.isdir(path):
            # Check if it's a Hugging Face dataset
            try:
                ds = load_from_disk(path)
                analysis["type"] = "HuggingFace Dataset"
                analysis["features"] = list(ds.features.keys())

                # Check for smoke_data
                if "smoke_data" in ds.features:
                    ds_np = ds.with_format("numpy")[:]
                    smoke_data = ds_np["smoke_data"]

                    analysis["num_episodes"] = smoke_data.shape[0]
                    analysis["episode_steps"] = smoke_data.shape[1]
                    analysis["grid_shape"] = f"{smoke_data.shape[2]}x{smoke_data.shape[3]}"

                    analysis["mean_value"] = float(smoke_data.mean())
                    analysis["max_value"] = float(smoke_data.max())
                    analysis["nonzero_fraction"] = float((smoke_data > 0.01).mean())
                elif "obs_full_map" in ds.features:
                    analysis["type"] = "HF Replay Buffer (Transition based)"
                    analysis["num_episodes"] = len(ds)
                    # Get sample map
                    sample_map = np.array(ds[0]["obs_full_map"])
                    analysis["grid_shape"] = f"{sample_map.shape[0]}x{sample_map.shape[1]}"
            except Exception as e:
                analysis["errors"] = f"Failed to load as HF dataset: {str(e)}"
        elif path.endswith(".npz"):
            try:
                loader = np.load(path)
                analysis["type"] = "NumPy NPZ"
                analysis["features"] = list(loader.keys())
                if "smoke_data" in loader:
                    smoke_data = loader["smoke_data"]
                    analysis["num_episodes"] = smoke_data.shape[0]
                    analysis["episode_steps"] = smoke_data.shape[1]
                    analysis["grid_shape"] = f"{smoke_data.shape[2]}x{smoke_data.shape[3]}"
                    analysis["mean_value"] = float(smoke_data.mean())
                    analysis["max_value"] = float(smoke_data.max())
                    analysis["nonzero_fraction"] = float((smoke_data > 0.01).mean())
            except Exception as e:
                analysis["errors"] = f"Failed to load as NPZ: {str(e)}"
    except Exception as e:
        analysis["errors"] = f"Unexpected error: {str(e)}"

    return analysis


def main():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        return

    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]
    results = []

    print("=" * 80)
    print("ANALYZING DATASETS FOR TRAINING SUITABILITY")
    print("=" * 80)

    for path in sorted(subdirs):
        # Ignore system files like .DS_Store
        if os.path.basename(path).startswith("."):
            continue

        analysis = analyze_dataset(path)
        if analysis:
            results.append(analysis)

    # Print comparison
    print(
        f"{'Dataset Name':<42} | {'Type':<20} | {'Ep/Items':<8} | {'Steps':<6} | {'Grid':<8} | {'MaxVal':<6} | {'NonZero%':<8}"
    )
    print("-" * 115)

    valid_candidates = []
    for r in results:
        if r["errors"]:
            print(f"{r['name']:<42} | ERROR: {r['errors'][:60]}")
            continue

        print(
            f"{r['name']:<42} | {r['type'][:20]:<20} | {r['num_episodes']:<8} | {r['episode_steps']:<6} | {r['grid_shape']:<8} | {r['max_value']:<6.2f} | {r['nonzero_fraction'] * 100:<7.1f}%"
        )

        # Candidate selection logic:
        # Must have smoke_data, non-zero values, and substantial episodes
        if (
            r["type"] in ["HuggingFace Dataset", "NumPy NPZ"]
            and r["num_episodes"] > 0
            and r["nonzero_fraction"] > 0.01
        ):
            valid_candidates.append(r)

    print("=" * 80)
    print("RECOMMENDATION FOR TRAINING:")
    print("=" * 80)

    if not valid_candidates:
        print("No valid smoke sequential datasets found. Make sure to generate playbacks first!")
        return

    # Sort candidates by number of episodes * steps (total sequence length)
    valid_candidates.sort(key=lambda x: x["num_episodes"] * x["episode_steps"], reverse=True)

    best = valid_candidates[0]
    print(f"The best dataset found for training is: '{best['name']}'")
    print(f"  - Total episodes: {best['num_episodes']}")
    print(f"  - Steps per episode: {best['episode_steps']}")
    print(f"  - Grid dimensions: {best['grid_shape']}")
    print(f"  - Non-zero density fraction: {best['nonzero_fraction'] * 100:.1f}%")
    print(f"  - Max smoke value: {best['max_value']:.2f}")
    print(
        "\nReasoning: It has the largest volume of sequential smoke maps and rich activity (high non-zero fraction)."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
