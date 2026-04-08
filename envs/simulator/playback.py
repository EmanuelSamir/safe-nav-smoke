import os
from dataclasses import dataclass

import numpy as np
from datasets import load_from_disk
from scipy.ndimage import map_coordinates

from envs.simulator.playback_schema import SmokeDataSchema


@dataclass
class PlaybackParams:
    data_path: str


class Playback:
    """Simulator that plays back pre-calculated smoke simulation data from an HF Dataset."""

    def __init__(self, params: PlaybackParams):
        """Initializes the playback simulator.

        :param params: Configuration containing data_path (HF Dataset directory).
        """
        self.params = params
        if not os.path.exists(params.data_path):
            raise FileNotFoundError(f"Smoke dataset not found: {params.data_path}")

        print(f"Loading smoke data from dataset at {params.data_path}...")
        self.dataset = load_from_disk(params.data_path)
        self.dataset = self.dataset.with_format("numpy")

        # Consistent parameters from the first row
        first_row = self.dataset[0]
        self.x_size = float(first_row.get(SmokeDataSchema.X_SIZE, 30.0))
        self.y_size = float(first_row.get(SmokeDataSchema.Y_SIZE, 30.0))
        self.resolution = float(first_row.get(SmokeDataSchema.RESOLUTION, 0.2))
        self.dt = float(first_row.get(SmokeDataSchema.DT, 0.1))

        # Grid parameters
        sample_map = np.array(first_row[SmokeDataSchema.SMOKE_DATA][0])
        self.H, self.W = sample_map.shape
        self.num_episodes = len(self.dataset)
        self.max_steps = len(first_row[SmokeDataSchema.SMOKE_DATA])

        # State
        self.current_episode_idx = -1
        self.current_step_idx = 0
        self.current_episode_data = None  # Lazily loaded per episode

        print(f"Playback ready: {self.num_episodes} episodes, {self.max_steps} steps per episode")
        print(
            f"Grid: ({self.H}, {self.W}) at {self.resolution}m resolution | World: {self.x_size}mx{self.y_size}m"
        )

    def reset(self, episode_idx=None):
        """Resets the simulator to the start of an episode."""
        if episode_idx is not None:
            self.current_episode_idx = episode_idx % self.num_episodes
        else:
            self.current_episode_idx = (self.current_episode_idx + 1) % self.num_episodes

        # Lazy load episode data into memory for fast step() access
        # we convert to numpy array for fast slicing
        episode_row = self.dataset[self.current_episode_idx]
        self.current_episode_data = np.array(
            episode_row[SmokeDataSchema.SMOKE_DATA], dtype=np.float32
        )
        self.current_step_idx = 0

    def step(self, dt=None):
        """Advances the simulation to the next pre-recorded frame."""
        if self.current_step_idx < self.max_steps - 1:
            self.current_step_idx += 1

    def get_smoke_map(self):
        """Returns the current 2D smoke density map."""
        if self.current_episode_data is None:
            self.reset(0)
        return self.current_episode_data[self.current_step_idx]

    def get_smoke_density(self, pos: np.ndarray) -> np.ndarray:
        """Samples smoke density at world coordinates using interpolation."""
        if self.current_episode_data is None:
            self.reset(0)

        if pos.ndim == 1:
            pos = pos.reshape(1, 2)

        # Map (x, y) to grid indices
        x_coords = (pos[:, 0] / self.resolution) - 0.5
        y_coords = (pos[:, 1] / self.resolution) - 0.5
        coords = np.stack([y_coords, x_coords])

        grid = self.current_episode_data[self.current_step_idx]
        values = map_coordinates(grid, coords, order=1, mode="constant", cval=0.0)

        return values.reshape(-1, 1)

    def get_smoke_extent(self):
        """Returns the [xmin, xmax, ymin, ymax] extent of the world."""
        return [0, self.x_size, 0, self.y_size]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test with standard path
    try:
        params = PlaybackParams(data_path="data/playback_smoke_v1")
        sim = Playback(params)
        sim.reset()

        smoke_map = sim.get_smoke_map()
        plt.imshow(smoke_map, origin="lower", extent=sim.get_smoke_extent())
        plt.title("Playback Test (HF Dataset)")
        plt.show()
    except Exception as e:
        print(f"Test failed: {e}")
