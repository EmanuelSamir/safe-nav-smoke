import numpy as np
from dataclasses import dataclass, field
from learning.base_model import BaseModel
from learning.gaussian_process import GaussianProcess
from itertools import product
import matplotlib.pyplot as plt
from simulator.static_smoke import StaticSmoke, SmokeBlobParams
from simulator.dynamic_smoke import DynamicSmoke, DynamicSmokeParams
from simulator.sensor import DownwardsSensorParams
from tqdm import tqdm
from src.utils import *
import scipy.stats as stats
from envs.smoke_env_dyn import EnvParams, RobotParams, DynamicSmokeEnv
from simulator.sensor import GlobalSensorParams, DownwardsSensorParams, PointSensorParams
import time

class InferenceRegion:
    def __init__(self, region_type: str):
        if type(self) is InferenceRegion:
            raise TypeError("InferenceRegion cannot be instantiated directly.")
        self.region_type = region_type

    def get_region_bounds(self, x_robot_pos: float, y_robot_pos: float):
        """
        Get the region bounds.
        Args:
            x_robot_pos: float
            y_robot_pos: float
        Returns:
            region_bounds: tuple[tuple[float, float], tuple[float, float]]
        """
        raise NotImplementedError("get_region_bounds is not implemented for this region type.")
    
    def get_inference_region(self, x_robot_pos: float, y_robot_pos: float):
        """
        Get the inference region based on the robot position.
        Args:
            x_robot_pos: float
            y_robot_pos: float
        Returns:
            inference_region: InferenceRegion
        """
        raise NotImplementedError("get_inference_region is not implemented for this region type.")

    def map_region_values_to_2d(self, region_values: np.ndarray):
        """
        Maps the region to a 2D array.
        Args:
            region_values: np.ndarray, shape (n, 1)
        Returns:
            region_values_2d: np.ndarray, shape (nx, ny)
            nx is the number of x points in the region
            ny is the number of y points in the region
        """
        raise NotImplementedError("map_region_values_to_2d is not implemented for this region type.")

class GlobalRegion(InferenceRegion):
    def __init__(self, world_size: tuple[float, float], resolution: float):
        super().__init__("global")
        self.world_size = world_size
        self.resolution = resolution

        self.x_size = int(self.world_size[0] / self.resolution)
        self.y_size = int(self.world_size[1] / self.resolution)
        self.inference_region = self.build_inference_region(self.x_size, self.y_size)

    def get_region_bounds(self, x_robot_pos: float, y_robot_pos: float):
        return ((0, self.world_size[0]), (0, self.world_size[1]))

    def build_inference_region(self, x_size: int, y_size: int):
        """
        Creates a region of size world_size with resolution.
        Returns:
            inference_region: np.ndarray, shape (n, 2)
            inference_region[i, 0] is the x coordinate of the i-th point
            inference_region[i, 1] is the y coordinate of the i-th point
        """
        inference_region = np.meshgrid(np.linspace(0, self.world_size[0], x_size),
                                            np.linspace(0, self.world_size[1], y_size))
        inference_region = np.stack([inference_region[0].ravel(), inference_region[1].ravel()], axis=-1)
        return inference_region
    
    def get_inference_region(self, x_robot_pos: float, y_robot_pos: float):
        return self.inference_region

    def map_region_values_to_2d(self, region_values: np.ndarray):
        return region_values.reshape(self.y_size, self.x_size)

class LocalRegion(InferenceRegion):
    def __init__(self, range_bound: tuple[float, float], resolution: float):
        super().__init__("local")
        """
        range_bound: tuple[float, float]
        range_bound[0] is the range in x direction
        range_bound[1] is the range in y direction
          | range_bound[1]
        --*-- range_bound[0]
          |
        """
        self.range_bound = range_bound
        self.resolution = resolution
        self.x_size = int(2 * self.range_bound[0] / self.resolution)
        self.y_size = int(2 * self.range_bound[1] / self.resolution)
        self.non_shifted_region = self.build_non_shifted_region(self.x_size, self.y_size)

    def get_region_bounds(self, x_robot_pos: float, y_robot_pos: float):
        return ((x_robot_pos - self.range_bound[0], x_robot_pos + self.range_bound[0]), 
                (y_robot_pos - self.range_bound[1], y_robot_pos + self.range_bound[1]))

    def build_non_shifted_region(self, x_size: int, y_size: int):
        """
        Creates a region of size range_bound with resolution.
        Returns:
            inference_region: np.ndarray, shape (n, 2)
            inference_region[i, 0] is the x coordinate of the i-th point
            inference_region[i, 1] is the y coordinate of the i-th point
        """
        inference_region = np.meshgrid(np.linspace(-self.range_bound[0], self.range_bound[0], x_size),
                                        np.linspace(-self.range_bound[1], self.range_bound[1], y_size))
        inference_region = np.stack([inference_region[0].ravel(), inference_region[1].ravel()], axis=-1)
        return inference_region

    def get_inference_region(self, x_robot_pos: float, y_robot_pos: float):
        return self.non_shifted_region + np.array([x_robot_pos, y_robot_pos])

    def map_region_values_to_2d(self, region_values: np.ndarray):
        return region_values.reshape(self.y_size, self.x_size)

@dataclass
class RiskMapParams:
    inference_region: InferenceRegion = field(default_factory=lambda: LocalRegion(range_bound=(1.0, 1.0), resolution=0.5))

    # TODO: Add case when starting from zero in time for changing inferencer. To normalize time

    # Available map rule types:
    # - 'mean': mean map rule
    # - 'cvar': cvar map rule
    map_rule_type: str = 'cvar'

    # Used for cvar map rule
    cvar_alpha: float = 0.95
    gamma: float = 0.75
    beta: float = 0.20

class RiskMapBuilder():
    def __init__(self, params: RiskMapParams):
        """
        Builds a risk map based on the given parameters.

        Args:
            params: RiskMapParams
        """
        self.params = params
        self.inference_region = params.inference_region

        # TODO: Check if the initial map is correct?
        # self.failure_map = np.ones(self.map_cell_size) # (x, y, t), to show in image, need to flip y axis
        self.map_assets = {'mean_map': None, 'std_map': None, 'risk_map': None, 'coords_map': None}

    def build_map(self, forecaster: BaseModel, x_robot_pos: float, y_robot_pos: float, time: float = None):
        """
        Builds a risk map based on the given forecaster. Returns the risk map in the form of a numpy array filled with 0s and 1s.

        Args:
            forecaster: BaseModel
        Returns:
            risk_map: np.ndarray
        """
        inference_region = self.inference_region.get_inference_region(x_robot_pos, y_robot_pos)
        self.map_assets['coords_map'] = inference_region

        space_coords = self.map_assets['coords_map']
        if time is not None:
            time_coords = np.full((space_coords.shape[0], 1), time)
            space_time_coords = np.concatenate([space_coords, time_coords], axis=-1)
        else:
            space_coords = space_coords - np.array([x_robot_pos, y_robot_pos])
            space_time_coords = space_coords

        pred, std = forecaster.predict(space_time_coords)
        pred = pred.squeeze()
        std = std.squeeze()

        # TODO: Scale mean and cvar so that the values are between 0 and 1 and have meaningful values.
        self.map_assets['mean_map'] = self.inference_region.map_region_values_to_2d(pred)
        self.map_assets['std_map'] = self.inference_region.map_region_values_to_2d(std)
        cvar = self.cvar_map(pred, std)
        self.map_assets['cvar_map'] = self.inference_region.map_region_values_to_2d(cvar)

        if self.params.map_rule_type == 'cvar':
            self.map_assets['risk_map'] = cvar
        elif self.params.map_rule_type == 'mean':
            self.map_assets['risk_map'] = pred
        elif self.params.map_rule_type == 'std':
            self.map_assets['risk_map'] = std
        else:
            raise ValueError(f"Invalid map rule type: {self.params.map_rule_type}")
        
        return self.map_assets['risk_map']
    
    def cvar_map(self, map_mean: np.ndarray, map_std: np.ndarray, tricked: bool = False):
        """
        Risk map is a map where the value is the risk of the region.
        Args:
            map_mean: np.ndarray, shape (nx, ny)
            map_std: np.ndarray, shape (nx, ny)
            tricked: bool, whether to trick the cvar map
        Returns:
            cvar_map: np.ndarray, shape (nx, ny)
            cvar_map[i, j] is the cvar of the region at (i, j)
        """
        map_mean = map_mean.squeeze()
        map_std = self.params.beta * map_std.squeeze()
        w = np.exp(- self.params.gamma * map_std) 
        cvar = map_mean + w * map_std * stats.norm.pdf(stats.norm.ppf(self.params.cvar_alpha)) / (1 - self.params.cvar_alpha)

        if tricked:
            tricked_cvar = np.ones_like(cvar)
            tricked_cvar[~np.isclose(map_mean, 0.5) | ~np.isclose(map_std, 1.0)] = cvar[~np.isclose(map_mean, 0.5) | ~np.isclose(map_std, 1.0)]
            return tricked_cvar
        else:
            return cvar
    
    def plot_map(self, risk_map: np.ndarray, x_robot_pos: float, y_robot_pos: float, fig: plt.Figure = None, ax: plt.Axes = None, title: str = None):
        created_fig = fig is None or ax is None
        if created_fig:
            fig, ax = plt.subplots()

        plot_map = self.inference_region.map_region_values_to_2d(risk_map)

        map_array_im = plot_map

        (x_min, x_max), (y_min, y_max) = self.inference_region.get_region_bounds(x_robot_pos, y_robot_pos)

        if ax.images:
            img = ax.images[0]
            img.set_array(map_array_im)
            img.set_extent([x_min, x_max, y_min, y_max])
        else:
            img = ax.imshow(
                map_array_im,
                extent=[x_min, x_max, y_min, y_max],
                origin='lower',
                cmap='Spectral', 
                vmin=0,
                vmax=1
            )
            if title is not None:
                ax.set_title(title)
            else:
                ax.set_title(r'Risk Map at $t+H$ time step')
            ax.set_xticks([])
            ax.set_yticks([])
            # colorbar = fig.colorbar(ax_, ax=ax)
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)

        fig.canvas.draw_idle()
        # fig.colorbar(img, ax=ax, label='Risk')

        plt.show(block=created_fig)

def get_value_in_map_from_coords(coords: np.ndarray, map_coords: np.ndarray, map_values: np.ndarray) -> np.ndarray:
    """
    Get the closest value(s) in map_values for given coordinates.
    
    Args:
        coords: np.ndarray, shape (m, 2)
        map_coords: np.ndarray, shape (n, 2)
        map_values: np.ndarray, shape (n,)
    
    Returns:
        values: np.ndarray, shape (m,)
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be of shape (m, 2)"
    assert map_coords.ndim == 2 and map_coords.shape[1] == 2, "map_coords must be of shape (n, 2)"
    assert map_values.shape[0] == map_coords.shape[0], "map_values must have same length as map_coords"

    # Compute pairwise distances (m x n)
    dists = np.linalg.norm(coords[:, None, :] - map_coords[None, :, :], axis=-1)
    # For each coord, find index of closest map point
    nearest_idx = np.argmin(dists, axis=1)
    # Retrieve corresponding values
    return map_values[nearest_idx]

if __name__ == "__main__":
    x_size, y_size = 60, 50
    params = RiskMapParams( inference_region=GlobalRegion(world_size=(x_size, y_size), resolution=1.0),
                            map_rule_type='cvar')
    builder = RiskMapBuilder(params)

    env_params = EnvParams.load_from_yaml("envs/env_cfg.yaml")

    robot_params = RobotParams.load_from_yaml("agents/dubins2d_fixed_velocity_cfg.yaml")
    world_x_size = 60
    world_y_size = 50
    env_params.world_x_size = world_x_size
    env_params.world_y_size = world_y_size
    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=1.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=20, y_pos=20, intensity=1.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=15, y_pos=45, intensity=1.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=40, y_pos=20, intensity=1.0, spread_rate=8.0)
    ]
    sensor_params = GlobalSensorParams(world_x_size=world_x_size, world_y_size=world_y_size)
    env_params.sensor_params = sensor_params
    smoke_params = DynamicSmokeParams(x_size=world_x_size, y_size=world_y_size, smoke_blob_params=smoke_blob_params, resolution=0.5)

    env = DynamicSmokeEnv(env_params, robot_params, smoke_params)
    initial_state = {"location": np.array([45, 15]), "angle": 0, "smoke_density": 0, "velocity": 4.0}
    state, _ = env.reset(initial_state=initial_state)
    env.smoke_simulator.step(dt=0.2)

    gp = GaussianProcess(online=True)

    obs = env._get_obs()

    # Calibration kernel and map building

    t0 = 0.0
    X_input = np.concatenate([obs["smoke_density_location"], np.full((obs["smoke_density_location"].shape[0], 1), t0)], axis=-1)
    y_observe = obs["smoke_density"]

    gp.track_data(X_input, y_observe)
    gp.update()

    builder.build_map(gp, 40, 20, 0.1)
    builder.plot_map(builder.map_assets['mean_map'], 40, 20)
    builder.plot_map(builder.map_assets['std_map'], 40, 20)
    builder.plot_map(builder.map_assets['cvar_map'], 40, 20)

    # TODO: Calibration of GP parameters and map building
