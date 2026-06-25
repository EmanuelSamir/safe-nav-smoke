import os

import numpy as np
from datasets import load_from_disk
from scipy.ndimage import map_coordinates

from src.env.simulator.base_smoke_simulator import BaseSmokeSimulator
from src.env.simulator.schemas import PlaybackConfig
from src.env.simulator.smoke_data_schema import SmokeDataSchema


class Playback(BaseSmokeSimulator):
    """Simulator that plays back pre-calculated smoke simulation data from an HF Dataset."""

    def __init__(self, cfg: PlaybackConfig):
        """Initializes the playback simulator.

        :param cfg: Configuration containing data_path (HF Dataset directory).
        """
        self.cfg = cfg
        if not os.path.exists(cfg.data_path):
            raise FileNotFoundError(f"Smoke dataset not found: {cfg.data_path}")

        print(f"Loading smoke data from dataset at {cfg.data_path}...")
        self.dataset = load_from_disk(cfg.data_path)
        self.dataset = self.dataset.with_format("numpy")

        # Consistent parameters from the first row
        first_row = self.dataset[0]
        self.cfg.x_size = float(first_row.get(SmokeDataSchema.X_SIZE, 30.0))
        self.cfg.y_size = float(first_row.get(SmokeDataSchema.Y_SIZE, 30.0))
        self.cfg.resolution = float(first_row.get(SmokeDataSchema.RESOLUTION, 0.2))
        self.cfg.dt = float(first_row.get(SmokeDataSchema.DT, 0.1))

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
            f"Grid: ({self.H}, {self.W}) at {self.cfg.resolution}m resolution | World: {self.cfg.x_size}mx{self.cfg.y_size}m"
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
        x_coords = (pos[:, 0] / self.cfg.resolution) - 0.5
        y_coords = (pos[:, 1] / self.cfg.resolution) - 0.5
        coords = np.stack([y_coords, x_coords])

        grid = self.current_episode_data[self.current_step_idx]
        values = map_coordinates(grid, coords, order=1, mode="constant", cval=0.0)

        return values.reshape(-1, 1)

    def get_smoke_extent(self):
        """Returns the [xmin, xmax, ymin, ymax] extent of the world."""
        return [0, self.cfg.x_size, 0, self.cfg.y_size]

    def get_smoke_map_tensor(self):
        """Returns the current 2D smoke density map as a native PyTorch tensor."""
        import torch

        from src.utils.config_utils import get_device

        return torch.tensor(self.get_smoke_map(), dtype=torch.float32, device=get_device())


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test with standard path
    try:
        cfg = PlaybackConfig(data_path="data/smoke_env_100ep")
        sim = Playback(cfg)
        sim.reset()

        fig, ax = plt.subplots()
        for _ in range(100):
            sim.step()
            sim.plot_smoke_map(fig=fig, ax=ax)
            print(np.round(sim.get_smoke_density(np.array([[10, 40], [40, 10]])), 2))
            plt.draw()
            plt.pause(0.1)

        plt.show()
    except Exception as e:
        print(f"Test failed: {e}")
