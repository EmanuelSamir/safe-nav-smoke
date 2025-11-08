import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.path import Path

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
        self.smoke_blob_params = smoke_blob_params

        self.smoke_map = self.build_smoke_map(smoke_blob_params)

    def reset(self):
        self.smoke_map = self.build_smoke_map(self.smoke_blob_params)

    def step(self):
        pass

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

    def get_smoke_density_at_point(self, x, y) -> float:
        """
        x: float
        y: float
        return: float
        """
        if x < 0 or x > self.x_size or y < 0 or y > self.y_size:
            return 0
        y_proj, x_proj = world_to_index(x, y, self.x_size, self.y_size, self.resolution)
        return self.smoke_map[y_proj, x_proj]

    def get_smoke_density_within_polygon(self, polygon: np.ndarray, return_location: bool = False) -> np.ndarray:
        """
        polygon: nx2 array
        return: (n,1) array, optional: (n,2) array
        """
        assert polygon.shape[1] == 2 and polygon.ndim == 2, "Polygon must be a nx2 array"

        polygon_in_discrete = [world_to_index(p[0], p[1], self.x_size, self.y_size, self.resolution) for p in polygon]

        cols, rows = np.meshgrid(np.arange(self.x_discrete_size), np.arange(self.y_discrete_size))

        pts = np.column_stack([rows.ravel(), cols.ravel()])

        path = Path(polygon_in_discrete)
        inside = path.contains_points(pts)
        inside_map = inside.reshape(self.y_discrete_size, self.x_discrete_size)

        values = self.smoke_map[inside_map]

        if return_location:
            pts_in_discrete = pts[inside]
            pts_in_world = [index_to_world(p[0], p[1], self.x_size, self.y_size, self.resolution) for p in pts_in_discrete]
            pts_in_world = np.array(pts_in_world)
            return values.reshape(-1, 1), pts_in_world

        return values.reshape(-1, 1)
        
    def get_smoke_density(self, pos: np.ndarray) -> np.ndarray:
        """
        pos: nx2 array or 1x2 array
        return: (n,1) array
        """

        if pos.ndim == 1:
            return np.array([[self.get_smoke_density_at_point(pos[0], pos[1])]])

        assert pos.shape[1] == 2 and pos.shape.dim == 2, "Position must be a nx2 array"
        
        densities = []
        for i in range(pos.shape[0]):
            x, y = pos[i]
            density = self.get_smoke_density_at_point(x, y)
            densities.append(density)
        return np.array(densities).reshape(-1, 1)

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
        # SmokeBlobParams(x_pos=60, y_pos=20, intensity=2.0, spread_rate=3.0),
        # SmokeBlobParams(x_pos=50, y_pos=10, intensity=1.5, spread_rate=5.0),
    ]
    
    smoke_simulator = StaticSmoke(x_size=80, y_size=50, resolution=0.8, smoke_blob_params=smoke_blob_params)

    center = np.array([10, 40])
    fov_size = 5
    square = np.array([[center[0] - fov_size / 2, center[1] - fov_size / 2], [center[0] + fov_size / 2, center[1] - fov_size / 2], [center[0] + fov_size / 2, center[1] + fov_size / 2], [center[0] - fov_size / 2, center[1] + fov_size / 2]])
    print(np.round(smoke_simulator.get_smoke_density_within_polygon(square), 2))
    print(np.round(smoke_simulator.get_smoke_density(np.array([10, 40])), 2))
    smoke_simulator.plot_smoke_map()
    plt.show()






