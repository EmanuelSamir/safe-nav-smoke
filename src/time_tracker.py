import time
from collections import defaultdict
import numpy as np

class TimeTracker:
    """
    Clase para medir tiempos de ejecución por bloques de código usando `with`.
    Guarda un historial de tiempos por nombre, y puede devolver promedios, totales, etc.
    """

    def __init__(self):
        self.times = defaultdict(list)   # {block_name: [durations]}
        self._start_times = {}           # interno, guarda inicio por bloque activo

    def track(self, name: str):
        """
        Uso:
            with timer.track("nombre_bloque"):
                <código a medir>
        """
        return _TimeBlock(self, name)

    def record(self, name: str, duration: float):
        """Guarda el tiempo en el registro."""
        self.times[name].append(duration)

    def summary(self):
        """Devuelve promedio y std por bloque."""
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
        """Devuelve el diccionario crudo."""
        return dict(self.times)
    
    def reset(self):
        """Limpia todos los registros."""
        self.times.clear()
        self._start_times.clear()


class _TimeBlock:
    """Context manager interno que mide el tiempo entre __enter__ y __exit__."""
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