import numpy as np
from dataclasses import dataclass
from src.models.base_model import BaseModel
from src.models.gaussian_process import GaussianProcess
from itertools import product
import matplotlib.pyplot as plt
from simulator.static_smoke import StaticSmoke, SmokeBlobParams
from simulator.dynamic_smoke import DynamicSmoke, DynamicSmokeParams
from simulator.sensor import DownwardsSensorParams
from tqdm import tqdm
from src.utils import *
import scipy.stats as stats
@dataclass
class FailureMapParams:
    x_size: int
    y_size: int
    resolution: float

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

class FailureMapBuilder():  
    def __init__(self, params: FailureMapParams):
        """
        Builds a failure map based on the given parameters.

        Args:
            params: FailureMapParams
        """
        self.params = params

        cell_y_size, cell_x_size = get_index_bounds(self.params.x_size, self.params.y_size, self.params.resolution)
        self.domain_cells = np.array([cell_x_size, cell_y_size])

        # TODO: Check if the initial map is correct?
        self.failure_map = np.ones([cell_y_size, cell_x_size])

        x = np.linspace(0, self.params.x_size, cell_x_size, endpoint=False)
        y = np.linspace(0, self.params.y_size, cell_y_size, endpoint=False)
        self.xy_coords = np.flip(np.array(list(product(y, x))), axis=1)
        self.map_assets = {'continuous_map': None, 'std_map': None, 'cvar_map': None}


    def build_map(self, forecaster: BaseModel, time: float):
        """
        Builds a failure map based on the given forecaster. Returns the failure map in the form of a numpy array filled with 0s and 1s.

        Args:
            forecaster: BaseModel
        Returns:
            failure_map: np.ndarray
        """
        try:
            y_pred, std_pred = forecaster.predict(np.concatenate([self.xy_coords, np.full((self.xy_coords.shape[0], 1), time)], axis=1))

            self.map_assets['continuous_map'] = y_pred.reshape(self.domain_cells[1], self.domain_cells[0])

            if self.params.map_rule_type == 'cvar':
                self.map_assets['std_map'] = std_pred.reshape(self.domain_cells[1], self.domain_cells[0])

            self.failure_map = self.rule_based_map(self.map_assets)
        except Exception as e:
            print(f"Error predicting: {e}. Returning last failure map.")

        return self.failure_map
        
    def build_continuous_map(self, forecaster: BaseModel):
        try:
            y_pred, _ = forecaster.predict(self.xy_coords)
            continuous_map = y_pred.reshape(self.domain_cells[1], self.domain_cells[0])
            return continuous_map
        except Exception as e:
            print(f"Error predicting: {e}. Returning last failure map instead.")
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
            self.map_assets['cvar_map'] = tricked_cvar.reshape(self.domain_cells[1], self.domain_cells[0])
            failure_map = (tricked_cvar < self.params.map_rule_threshold).astype(bool)
        else:
            raise ValueError(f"Invalid map rule type: {self.params.map_rule_type} for map assets: {map_assets.keys()}")

        return failure_map
    
    def plot_failure_map(self, fig: plt.Figure = None, ax: plt.Axes = None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if ax.images:
            ax.images[0].set_array(self.failure_map)
        else:
            ax_ = ax.imshow(self.failure_map, vmin=0.0, vmax=1.0, extent=[0, self.params.x_size, 0, self.params.y_size], origin='lower', cmap='gray')
            #fig.colorbar(ax_, label='Failure', shrink=0.5)
            ax.set_xlim(0, self.params.x_size)
            ax.set_ylim(0, self.params.y_size)
            ax.set_title('Failure Map')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        fig.canvas.draw()


if __name__ == "__main__":
    x_size, y_size = 60, 50
    params = FailureMapParams(x_size=x_size, y_size=y_size, resolution=1.0, map_rule_type='cvar', map_rule_threshold=0.6, cvar_alpha=0.95)
    builder = FailureMapBuilder(params)
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

    failure_map = builder.build_map(gp, sample_size*0.1)
    print("CVAR map shape:", builder.map_assets['cvar_map'].shape)
    plt.imshow(builder.map_assets['cvar_map'], extent=[0, x_size, 0, y_size], origin='lower', cmap='gray', zorder=-10)
    plt.colorbar()
    print("Failure map shape:", failure_map.shape)
    fig, ax = plt.subplots()
    builder.plot_failure_map(fig=fig, ax=ax)
    plt.show()