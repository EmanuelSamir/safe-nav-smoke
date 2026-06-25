import time
from collections import defaultdict

import pandas as pd


class TimeTracker:
    """Class to measure execution times for code blocks using `with`.

    Stores a history of times by name, and can return averages, totals, etc.
    All times are recorded and reported in milliseconds (ms).
    """

    def __init__(self):
        """Initialize the TimeTracker."""
        self.times = defaultdict(list)  # {block_name: [durations]}
        self._start_times = {}  # internal, stores start time per active block

    def track(self, name: str):
        """Usage.

        with timer.track("block_name"):
            <code to measure>
        """
        return _TimeBlock(self, name)

    def record(self, name: str, duration: float):
        """Saves the time in the log."""
        self.times[name].append(duration)

    def summary(self):
        """Returns avg and std per block in milliseconds."""
        return {
            k: {
                "mean (ms)": sum(v) / len(v),
                "std (ms)": (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5,
                "count": len(v),
                "total (ms)": sum(v),
            }
            for k, v in self.times.items()
        }

    def pretty_print(self):
        """Prints a nicely formatted table of the tracked times."""
        if not self.times:
            print("TimeTracker is empty.")
            return

        df = pd.DataFrame(self.summary()).T
        print("\n" + "="*65)
        print(" TimeTracker Summary (All times in ms)")
        print("="*65)
        print(df.to_markdown(floatfmt=".3f"))
        print("="*65 + "\n")

    def as_dict(self):
        """Returns the raw dictionary."""
        return dict(self.times)

    def reset(self):
        """Clears all records."""
        self.times.clear()
        self._start_times.clear()

    def to_csv(self, filepath: str):
        """Save times to a CSV file."""
        df = pd.DataFrame(self.summary())
        df.to_csv(filepath)

        print(f"Time summary saved to {filepath}")


class _TimeBlock:
    """Internal context manager that measures time between __enter__ and __exit__."""

    def __init__(self, tracker: TimeTracker, name: str):
        self.tracker = tracker
        self.name = name
        self.start = None

    def __enter__(self):
        self._sync_torch()
        self._sync_jax()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sync_torch()
        self._sync_jax()
        end = time.perf_counter()
        # Convert seconds to milliseconds
        duration_ms = (end - self.start) * 1000.0
        self.tracker.record(self.name, duration_ms)

    def _sync_torch(self):
        """Synchronize PyTorch GPU/MPS operations if PyTorch is available."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        except ImportError:
            pass

    def _sync_jax(self):
        """Synchronize JAX GPU/TPU operations if JAX is available."""
        try:
            import jax
            # Enqueue a dummy operation and wait for it.
            # Because JAX executes sequentially on the stream, this implicitly
            # waits for all previously dispatched JAX operations to finish!
            jax.device_put(0.0).block_until_ready()
        except ImportError:
            pass


if __name__ == "__main__":
    tracker = TimeTracker()
    with tracker.track("test1"):
        time.sleep(0.02)
    with tracker.track("test2"):
        time.sleep(0.02)
    with tracker.track("test1"):
        time.sleep(0.01)
    tracker.pretty_print()
