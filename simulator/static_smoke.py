import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

from src.utils import *

@dataclass
class SmokeBlobParams:
    x_pos: int
    y_pos: int
    intensity: float
    spread_rate: float

class StaticSmoke:
    def __init__(self, x_size:float, y_size:float, smoke_blob_params: list[SmokeBlobParams], resolution:float=0.1):
        self.x_size = x_size
        self.y_size = y_size
        self.resolution = resolution

        self.y_discrete_size, self.x_discrete_size = get_index_bounds(self.x_size, self.y_size, self.resolution)

        self.smoke_map = self.build_smoke_map(smoke_blob_params)

    def build_smoke_map(self, smoke_blob_params: list[SmokeBlobParams]):
        smoke_map = np.zeros((self.y_discrete_size, self.x_discrete_size))

        for blob in smoke_blob_params:
            y_pos_proj, x_pos_proj = world_to_index(blob.x_pos, blob.y_pos, self.x_size, self.y_size, self.resolution)

            y, x = np.ogrid[-y_pos_proj:smoke_map.shape[0]-y_pos_proj,
                            -x_pos_proj:smoke_map.shape[1]-x_pos_proj]
            
            sigma = blob.spread_rate / self.resolution
            gaussian = blob.intensity * np.exp(-(x*x + y*y)/(2.0*sigma**2))
            
            smoke_map += gaussian

        return smoke_map
        
    def get_smoke_density(self, x, y):
        y_proj, x_proj = world_to_index(x, y, self.x_size, self.y_size, self.resolution)
        return self.smoke_map[y_proj, x_proj]

    def get_smoke_map(self):
        return self.smoke_map
    
    def plot_smoke_map(self, fig: plt.Figure = None, ax: plt.Axes = None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if ax.images:
            ax.images[0].set_array(self.smoke_map)
        else:
            ax_ = ax.imshow(self.smoke_map, cmap='gray', extent=[0, self.x_size, 0, self.y_size], origin='lower')
            #fig.colorbar(ax_, label='Smoke Density')
            ax.set_title('Static Smoke Map')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')

        fig.canvas.draw()


if __name__ == "__main__":
    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=2.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=20, y_pos=20, intensity=1.5, spread_rate=5.0),
        SmokeBlobParams(x_pos=60, y_pos=20, intensity=2.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=50, y_pos=10, intensity=1.5, spread_rate=5.0),
    ]
    
    smoke_simulator = StaticSmoke(x_size=80, y_size=50, resolution=0.1, smoke_blob_params=smoke_blob_params)
    print(smoke_simulator.get_smoke_density(70, 20))
    smoke_simulator.plot_smoke_map()
    plt.show()






