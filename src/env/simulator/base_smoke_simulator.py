import abc

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.simulator.schemas import BaseSimConfig


class BaseSmokeSimulator(abc.ABC):
    """Abstract base class for smoke simulators (physics-based or playback)."""

    def __init__(self, cfg: BaseSimConfig) -> None:
        """Initialize the simulator."""
        self.cfg = cfg

    @abc.abstractmethod
    def reset(self, **kwargs) -> None:
        """Reset the simulator to its initial state."""
        pass

    @abc.abstractmethod
    def step(self, dt: float = 0.1) -> None:
        """Advance the simulation by dt."""
        pass

    @abc.abstractmethod
    def get_smoke_map(self) -> np.ndarray:
        """Returns the current 2D smoke density map as a numpy array."""
        pass

    @abc.abstractmethod
    def get_smoke_density(self, pos: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Samples smoke density at the given coordinates."""
        pass

    @abc.abstractmethod
    def get_smoke_extent(self) -> list[float]:
        """Returns the [xmin, xmax, ymin, ymax] extent of the world."""
        pass

    @abc.abstractmethod
    def get_smoke_map_tensor(self) -> torch.Tensor:
        """Returns the current 2D smoke density map as a PyTorch tensor."""
        pass

    def plot_smoke_map(self, fig: plt.Figure = None, ax: plt.Axes = None) -> None:
        """Plot the current smoke map."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        extent = self.get_smoke_extent()
        smoke_arr = self.get_smoke_map()

        if hasattr(ax, "images") and ax.images:
            ax.images[0].set_array(smoke_arr)
        else:
            ax_ = ax.imshow(smoke_arr, cmap="gray", extent=extent, origin="lower", vmin=0, vmax=1)
            fig.colorbar(ax_, label="Smoke Density")
            ax.set_title("Smoke Map")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

        fig.canvas.draw()
