import numpy as np
import os
import psutil

process = psutil.Process()

folder = "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d"
files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
print(f"Total files: {len(files)}")
for i, f in enumerate(files):
    path = os.path.join(folder, f)
    data = np.load(path)
    # just access shapes
    time_steps = data["time_steps"]
    gt = data["gt"] if "gt" in data else None
    sample = data["sample"] if "sample" in data else None
    print(f"{i}: {f} GT {gt.shape if gt is not None else None} SAMPLE {sample.shape if sample is not None else None} Mem: {process.memory_info().rss / 1e6:.1f}MB")
