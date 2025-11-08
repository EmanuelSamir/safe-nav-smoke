import numpy as np
from dataclasses import dataclass
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
@dataclass
class TimeDynamicFailureMapParams:
    resolution: float

    time_steps_horizon: int
    max_displacement_in_dt: float
    dt: float

    # Available map rule types:
    # - 'threshold': threshold map rule
    # TODO: Add more map rule types
    map_rule_type: str

    # Used for threshold map rule
    map_rule_threshold: float

    # Used for cvar map rule
    cvar_alpha: float = 0.95
    gamma: float = 2.5
    beta: float = 0.0

class TimeDynamicFailureMapBuilder():  
    def __init__(self, params: TimeDynamicFailureMapParams):
        """
        Builds a failure map based on the given parameters.

        Args:
            params: TimeDynamicFailureMapParams
        """
        self.params = params
        map_y_size = 2 * int(params.time_steps_horizon * params.max_displacement_in_dt / self.params.resolution)
        map_x_size = 2 * int(params.time_steps_horizon * params.max_displacement_in_dt / self.params.resolution)
        map_t_size = params.time_steps_horizon

        self.map_cell_size = (np.array([map_x_size, map_y_size, map_t_size])).astype(int)

        # TODO: Check if the initial map is correct?
        self.failure_map = np.ones(self.map_cell_size) # (x, y, t), to show in image, need to flip y axis

        self.map_assets = {'continuous_map': None, 'std_map': None, 'cvar_map': None, 'coords_space': None}

    def get_value_from_coords(self, coords: np.ndarray):
        """
        Get the value from the coords space.
        Args:
            coords: np.ndarray, shape (n, 3)
        Returns:
            value: bool, shape (n, 1)
        """
        try:
            if self.map_assets.get('coords_space') is None:
                raise ValueError("Coords space is not set")

            assert coords.shape[1] == 3, "Coords must be a 3D array"

            index = np.argmin(np.linalg.norm(self.map_assets['coords_space'] - coords, axis=-1))
            return self.failure_map[index]
        except Exception as e:
            print(f"Error getting value from coords: {e}. Returning None.")
            return None

    def build_map(self, forecaster: BaseModel, x_robot_pos: float, y_robot_pos: float, time: float):
        """
        Builds a failure map based on the given forecaster. Returns the failure map in the form of a numpy array filled with 0s and 1s.

        Args:
            forecaster: BaseModel
        Returns:
            failure_map: np.ndarray
        """
        x_bounds = (x_robot_pos - self.params.time_steps_horizon * self.params.max_displacement_in_dt / self.params.resolution,
                    x_robot_pos + self.params.time_steps_horizon * self.params.max_displacement_in_dt / self.params.resolution)
        y_bounds = (y_robot_pos - self.params.time_steps_horizon * self.params.max_displacement_in_dt / self.params.resolution,
                    y_robot_pos + self.params.time_steps_horizon * self.params.max_displacement_in_dt / self.params.resolution)

        x_space = np.linspace(x_bounds[0], x_bounds[1], self.map_cell_size[0])
        y_space = np.linspace(y_bounds[0], y_bounds[1], self.map_cell_size[1])
        t_space = np.linspace(time, time + self.params.time_steps_horizon * self.params.dt, self.map_cell_size[2])

        X, Y, T = np.meshgrid(x_space, y_space, t_space, indexing='ij')
        self.map_assets['coords_space'] = np.stack([X, Y, T], axis=-1)
        coords = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=-1)

        try:
            pred, std = forecaster.predict(coords)

            self.map_assets['continuous_map'] = pred.reshape(self.map_cell_size)

            if self.params.map_rule_type == 'cvar':
                self.map_assets['std_map'] = std.reshape(self.map_cell_size)

            self.failure_map = self.rule_based_map(self.map_assets)
        except Exception as e:
            print(f"Error predicting: {e}. Returning last failure map.")

        return self.failure_map
    
    def rule_based_map(self, map_assets: dict):
        """
        Failure map is a boolean map where 1s are empty and 0s are failure.
        """
        if self.params.map_rule_type == 'threshold' and map_assets.get('continuous_map') is not None:
            continuous_map = map_assets['continuous_map']
            failure_map = (continuous_map < self.params.map_rule_threshold).astype(bool)
        elif self.params.map_rule_type == 'cvar' and map_assets.get('std_map') is not None:
            std_map = map_assets['std_map']
            continuous_map = map_assets['continuous_map']
            # Option 1: Trick to set mean 0.5 and variance 1.0 to empty as default
            w = np.exp(- self.params.gamma * std_map)
            cvar = continuous_map + w * std_map * stats.norm.pdf(stats.norm.ppf(self.params.cvar_alpha)) / (1 - self.params.cvar_alpha)
            tricked_cvar = np.ones_like(cvar)
            tricked_cvar[~np.isclose(continuous_map, 0.5) | ~np.isclose(std_map, 1.0)] = cvar[~np.isclose(continuous_map, 0.5) | ~np.isclose(std_map, 1.0)]
            
            # Option 2: Penalized mean and variance
            # mu_penalized = continuous_map * np.exp(- self.params.beta * std_map)
            # w = np.exp(- self.params.gamma * std_map)
            # cvar = mu_penalized + w * std_map * stats.norm.pdf(stats.norm.ppf(self.params.cvar_alpha)) / (1 - self.params.cvar_alpha)
            self.map_assets['cvar_map'] = tricked_cvar.reshape(self.map_cell_size)
            failure_map = (tricked_cvar < self.params.map_rule_threshold).astype(bool)
        else:
            raise ValueError(f"Invalid map rule type: {self.params.map_rule_type} for map assets: {map_assets.keys()}")

        return failure_map
    
    def plot_failure_map(self, fig: plt.Figure = None, ax: plt.Axes = None, t_step: int = 0):
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        assert t_step < self.map_cell_size[2], "t_step is out of range"
        failure_map = self.failure_map[..., t_step] # (x, y)
        print("Failure map shape:", failure_map.shape)
        # map_array_im = failure_map.reshape(self.map_cell_size[1], self.map_cell_size[0])
        # map_array_im = np.flip(map_array_im, axis=0)
        map_array_im = failure_map.T

        if ax.images:
            ax.images[0].set_array(map_array_im)
        else:
            ax_ = ax.imshow(map_array_im, vmin=0.0, vmax=1.0, extent=[0, self.map_cell_size[0], 0, self.map_cell_size[1]], origin='lower', cmap='gray')
            #fig.colorbar(ax_, label='Failure', shrink=0.5)
            ax.set_xlim(0, self.map_cell_size[0])
            ax.set_ylim(0, self.map_cell_size[1])
            ax.set_title('Failure Map')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        fig.canvas.draw()


if __name__ == "__main__":
    x_size, y_size = 60, 50
    params = TimeDynamicFailureMapParams(resolution=0.5, 
                                        time_steps_horizon=10, 
                                        max_displacement_in_dt=1.0, 
                                        dt=0.1, map_rule_type='cvar', 
                                        map_rule_threshold=0.6, cvar_alpha=0.95)
    builder = TimeDynamicFailureMapBuilder(params)
    # print("Initial failure map shape:", builder.failure_map.shape)
    # builder.plot_failure_map()

    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=1.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=10, y_pos=20, intensity=1.0, spread_rate=5.0),
        SmokeBlobParams(x_pos=60, y_pos=20, intensity=1.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=50, y_pos=10, intensity=1.0, spread_rate=5.0),
    ]

    sensor_params = DownwardsSensorParams(world_x_size=x_size, world_y_size=y_size)
    smoke_params = DynamicSmokeParams(x_size=x_size, y_size=y_size, smoke_blob_params=smoke_blob_params, resolution=0.4, fov_sensor_params=sensor_params)

    smoke_simulator = DynamicSmoke(params=smoke_params)

    gp = GaussianProcess()

    sample_size = 2

    for i in tqdm(range(sample_size)):
        X_sample = np.array([np.random.uniform(0, x_size, 100), np.random.uniform(0, y_size, 100), np.array(100*[i*0.1])]).T
        y_observe = smoke_simulator.get_smoke_density(X_sample[:, :2])
        gp.track_data(X_sample, y_observe)
        smoke_simulator.step(dt=0.1)
        gp.update()
    plt.show()
    y_true = smoke_simulator.get_smoke_map()

    plt.figure()

    failure_map = builder.build_map(gp, 10, 20, 2.0)

    plt.imshow(y_true, extent=[0, x_size, 0, y_size], origin='lower', cmap='gray', zorder=-10)
    plt.figure()
    print("CVAR map shape:", builder.map_assets['cvar_map'].shape)
    cvar_map_t_0 = builder.map_assets['continuous_map'][..., 0]
    plt.imshow(cvar_map_t_0, origin='lower', cmap='gray', zorder=-10)
    print(builder.map_assets["coords_space"])
    plt.colorbar()
    print("Failure map shape:", failure_map.shape)
    fig, ax = plt.subplots()
    builder.plot_failure_map(fig=fig, ax=ax)
    plt.show()