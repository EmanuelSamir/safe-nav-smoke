
import pandas as pd
import numpy as np
from dataclasses import is_dataclass, asdict
import json
from collections import defaultdict

class LoggerMetrics:
    def __init__(self):
        self.values = defaultdict(list)

    def add_value(self, key: str, value: float):
        self.values[key].append(value)

    def get_last_value(self, key: str):
        if self.values.get(key) is None:
            return 0.0
        return self.values[key][-1]

    def get_values(self, key: str = None):
        if key is None:
            return self.values
        return self.values.get(key, [])

    def dump_to_csv(self, filepath: str):
        df = pd.DataFrame(self.values)
        df.to_csv(filepath, index=False)
    
    def reset(self):
        self.values.clear()


def dataclass_json_dump(obj, path):
    def convert(x):

        # ----- Dataclass -----
        if is_dataclass(x):
            return convert(asdict(x))

        # ----- Numpy array -----
        if isinstance(x, np.ndarray):
            return x.tolist()

        # ----- Numpy scalar (np.float32, np.int64, etc.) -----
        if isinstance(x, (np.generic,)):
            return x.item()

        # ----- Dict -----
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}

        # ----- List / Tuple -----
        if isinstance(x, (list, tuple)):
            return [convert(i) for i in x]

        # ----- Tipos JSON válidos -----
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x

        # ----- Clases no dataclass -----
        # NO convertirlas, solo representarlas como string para evitar crash.
        return str(x)

    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=4)