import os
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_from_disk

from src.env.simulator.smoke_data_schema import SmokeDataSchema


class ReplayEnv:
    """A generic replay buffer that stores samples as a dictionary of deques.

    Infers structure from SmokeDataSchema if keys are not provided.
    """

    def __init__(self, buffer_size: int, data_keys: Optional[List[str]] = None):
        """Initializes the buffer.

        :param buffer_size: The maximum size of the buffer.
        :param data_keys: A list of strings with the keys of the data to store.
                         If None, uses SmokeDataSchema.get_replay_buffer_keys().
        """
        self.buffer_size = int(buffer_size)
        self.data_keys = (
            data_keys if data_keys is not None else SmokeDataSchema.get_replay_buffer_keys()
        )
        self.buffer: Dict[str, deque] = {}
        self.global_data: Dict[str, Any] = {}
        self.current_size = 0
        self.reset()

    @classmethod
    def from_hf_dataset(cls, dataset_path: str, buffer_size: Optional[int] = None):
        """Creates and populates a ReplayEnv from a Hugging Face dataset on disk."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        ds = load_from_disk(dataset_path)
        ds = ds.with_format("numpy")
        keys = list(ds.features.keys())
        size = buffer_size if buffer_size is not None else len(ds)

        replay = cls(buffer_size=size, data_keys=keys)
        print(f"Loading {len(ds)} samples from HF dataset into ReplayEnv...")
        for row in ds:
            replay.add(**row)
        return replay

    def reset(self):
        """Initializes all deques for the specified keys."""
        self.buffer = {key: deque(maxlen=self.buffer_size) for key in self.data_keys}
        self.current_size = 0

    def add(self, **kwargs: Any):
        """Adds a sample to the buffer using keyword arguments.

        Example: buffer.add(state=s, action=a, done=d)
        """
        # Ensure only defined keys are passed
        if not set(kwargs.keys()).issubset(set(self.data_keys)):
            extra = set(kwargs.keys()) - set(self.data_keys)
            raise ValueError(f"Unknown keys provided: {extra}. Expected: {self.data_keys}")

        for key, value in kwargs.items():
            self.buffer[key].append(value)

        # Update current size
        self.current_size = len(self.buffer[self.data_keys[0]]) if self.data_keys else 0

    def get_from_index(self, index: int) -> Dict[str, Any]:
        """Retrieves a full sample from a specific index."""
        if index >= self.current_size:
            raise IndexError("Index out of current buffer limits.")

        data = {key: self.buffer[key][index] for key in self.data_keys if key in self.buffer}
        return data

    def save_to_file(self, filepath: str, **kwargs):
        """Saves the buffer content to an .npz file."""
        data_to_save = {}
        for key in self.data_keys:
            if key not in self.buffer or len(self.buffer[key]) == 0:
                continue
            try:
                temp_array = np.array(self.buffer[key])
            except Exception:
                temp_array = np.array(list(self.buffer[key]), dtype=object)
            data_to_save[key] = temp_array

        np.savez(
            filepath,
            **data_to_save,
            buffer_size=self.buffer_size,
            data_keys=self.data_keys,
            **kwargs,
        )
        print(f"Buffer successfully saved to: {filepath}")

    def load_from_file(self, filepath: str):
        """Loads the buffer content from an .npz file."""
        loaded = np.load(filepath, allow_pickle=True)

        if "data_keys" in loaded:
            self.data_keys = list(loaded["data_keys"])
            self.buffer_size = int(loaded.get("buffer_size", len(loaded[self.data_keys[0]])))

        self.reset()
        for key in loaded.files:
            if key in ("data_keys", "buffer_size"):
                continue

            if key in self.data_keys:
                data_array = loaded[key]
                if data_array.dtype == object:
                    self.buffer[key] = deque(
                        [np.array(i) for i in data_array], maxlen=self.buffer_size
                    )
                else:
                    self.buffer[key] = deque(data_array, maxlen=self.buffer_size)
            else:
                self.global_data[key] = loaded[key]

        self.current_size = len(self.buffer[self.data_keys[0]]) if self.data_keys else 0

    def full(self) -> bool:
        return self.current_size >= self.buffer_size

    def __len__(self) -> int:
        return self.current_size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get_from_index(index)
