import time
from collections import defaultdict
import numpy as np

class TimeTracker:
    """
    Class to measure execution times for code blocks using `with`.
    Stores a history of times by name, and can return averages, totals, etc.
    """

    def __init__(self):
        self.times = defaultdict(list)   # {block_name: [durations]}
        self._start_times = {}           # internal, stores start time per active block

    def track(self, name: str):
        """
        Usage:
            with timer.track("block_name"):
                <code to measure>
        """
        return _TimeBlock(self, name)

    def record(self, name: str, duration: float):
        """Saves the time in the log."""
        self.times[name].append(duration)

    def summary(self):
        """Returns avg and std per block."""
        return {
            k: {
                "mean": sum(v)/len(v),
                "std": (sum((x - sum(v)/len(v))**2 for x in v)/len(v))**0.5,
                "count": len(v),
                "total": sum(v),
            }
            for k, v in self.times.items()
        }

    def as_dict(self):
        """Returns the raw dictionary."""
        return dict(self.times)
    
    def reset(self):
        """Clears all records."""
        self.times.clear()
        self._start_times.clear()


class _TimeBlock:
    """Internal context manager that measures time between __enter__ and __exit__."""
    def __init__(self, tracker: TimeTracker, name: str):
        self.tracker = tracker
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        duration = end - self.start
        self.tracker.record(self.name, duration)
        
def stats_no_near_zero(data, tol=1e-3, return_counts=False):
    arr = np.asarray(data, dtype=float)
    mask = ~np.isclose(arr, 0.0, atol=tol)
    filtered = arr[mask]
    
    if filtered.size > 0:
        mean = np.mean(filtered)
        std = np.std(filtered, ddof=0) if filtered.size > 1 else 0.0
    else:
        mean, std = np.nan, np.nan
    
    if return_counts:
        return mean, std, np.sum(~mask), np.sum(mask)
    else:
        return mean, std

if __name__ == "__main__":
    tracker = TimeTracker()
    with tracker.track("test1"):
        time.sleep(2)
    with tracker.track("test2"):
        time.sleep(2)
    with tracker.track("test1"):
        time.sleep(1)
    print(tracker.summary())