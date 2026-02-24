import warnings
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
# Optional if using online version of GP
from goppy import OnlineGP, SquaredExponentialKernel, Matern32Kernel

import scipy.stats as stats
from collections import deque
from simulator.dynamic_smoke import SmokeBlobParams, DynamicSmoke, DynamicSmokeParams, DownwardsSensorParams

from src.models.base_model import BaseModel
from src.utils import *
from tqdm import tqdm

class Kernel:
    RBF = RBF(length_scale=5.0)
    Matern = Matern(length_scale=1.0, nu=1.3)
    ConstantKernel = C()

class OnlineKernel:
    Matern = Matern32Kernel(lengthscales=[2.0])

N_RESTARTS_OPTIMIZER = 10
NORMALIZE_Y = False

class GaussianProcess(BaseModel):
    def __init__(self, kernel: Kernel = Kernel.Matern, online_kernel: OnlineKernel = OnlineKernel.Matern, online: bool = False, history_size: int = None):
        super().__init__(history_size=history_size)

        self.online = online
        # Cumulative variables for std
        self.n_seen = 0
        self.mean_y = 0.0
        self.M2 = 0.0  # Sum of squares of deviations

        if not self.online:
            # self.kernel = kernel
            # self.model = GaussianProcessRegressor(kernel=self.kernel, 
            #                                     optimizer='fmin_l_bfgs_b', 
            #                                     n_restarts_optimizer=N_RESTARTS_OPTIMIZER, 
            #                                     normalize_y=NORMALIZE_Y)
            self.kernel = online_kernel
            self.model = OnlineGP(self.kernel, noise_var=0.01)
            self.first_update = False
        else:
            self.kernel = online_kernel
            self.model = OnlineGP(self.kernel, noise_var=0.01)
            self.first_update = False
        
    def update(self):
        if self.input_history and self.output_history:
            X = np.array(self.input_history)
            y = np.array(self.output_history)

            for yi in y:
                self.n_seen += 1
                delta = yi - self.mean_y
                self.mean_y += delta / self.n_seen
                self.M2 += delta * (yi - self.mean_y)

            if y.ndim == 1:
                y = y.reshape(-1, 1)

            if self.online:
                if not self.first_update:
                    self.model.fit(X, y)
                    self.first_update = True
                else:
                    self.model.add(X, y)

                # Flush history
                self.input_history = deque()
                self.output_history = deque()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model.fit(X, y)

    def predict(self, x):
        if not self.online:
            res = self.model.predict(x, what=['mean', 'mse'])
            y_pred = res['mean']
            std = np.sqrt(res['mse']) 
        else:
            res = self.model.predict(x, what=['mean', 'mse'])
            y_pred = res['mean']
            std = np.sqrt(res['mse']) 
        return y_pred, std

    def score(self, x, y_true):
        y_pred, _ = self.predict(x)
        return mean_squared_error(y_true, y_pred)
    
def plot_static_map(gp: GaussianProcess, world_x_size: float, world_y_size: float, resolution: float, fig: plt.Figure = None, ax: plt.Axes = None, plot_type: str = 'mean'):
    y_indices, x_indices = get_index_bounds(world_x_size, world_y_size, resolution)
    
    x = np.linspace(0, world_x_size, num=x_indices, endpoint=False)
    y = np.linspace(0, world_y_size, num=y_indices, endpoint=False)
    xy = np.array(list(product(y, x)))

    if plot_type == 'mean':
        y_pred, _ = gp.predict(xy)
        im_ = y_pred.reshape(y_indices, x_indices)
    elif plot_type == 'std':
        _, std = gp.predict(xy)
        im_ = std.reshape(y_indices, x_indices)
    elif plot_type == 'cvar':
        y_pred, std = gp.predict(xy)
        mu = y_pred.reshape(y_indices, x_indices)
        std = std.reshape(y_indices, x_indices)
        im_ = mu + std * stats.norm.pdf(stats.norm.ppf(0.95)) / (1 - 0.95)
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if ax.images:
        ax.images[0].set_array(im_)
    else:
        ax_ = ax.imshow(im_, cmap='gray', vmin=0, vmax=1.0, extent=[0, world_x_size, 0, world_y_size], origin='lower')
        ax.set_title('Predicted Map')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        #fig.colorbar(ax_, label='Estimated Smoke Density', shrink=0.5)

    fig.canvas.draw()

def plot_dynamic_map(gp: GaussianProcess, world_x_size: float, world_y_size: float, time: float, resolution: float, fig: plt.Figure = None, ax: plt.Axes = None, plot_type: str = 'mean'):
    num_rows, num_cols = get_index_bounds(world_x_size, world_y_size, resolution)
    
    x = np.linspace(0, world_x_size, num=num_cols, endpoint=False)
    y = np.linspace(0, world_y_size, num=num_rows, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')            # X,Y shape: (num_rows, num_cols)

    # 3) Flatten in C order (row-major) -> matches default reshape
    xy = np.column_stack([X.ravel(order='C'), Y.ravel(order='C')])  # (num_rows*num_cols, 2)

    # 4) Build input for the GP
    x_input = np.concatenate([xy, np.full((xy.shape[0], 1), time)], axis=1)

    if plot_type == 'mean':
        y_pred, _ = gp.predict(x_input)
        im_ = y_pred.reshape(num_rows, num_cols)
    elif plot_type == 'std':
        _, std = gp.predict(x_input)
        im_ = std.reshape(num_rows, num_cols)
    elif plot_type == 'cvar':
        y_pred, std = gp.predict(x_input)
        mu = y_pred.reshape(num_rows, num_cols)
        std = std.reshape(num_rows, num_cols)
        im_ = mu + std * stats.norm.pdf(stats.norm.ppf(0.95)) / (1 - 0.95)
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if ax.images:
        ax.images[0].set_array(im_)
    else:
        ax_ = ax.imshow(im_, cmap='gray', extent=[0, world_x_size, 0, world_y_size], origin='lower', zorder=-10)
        ax.set_title('Predicted Map')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(ax_, label=plot_type, shrink=0.5)

    fig.canvas.draw()

    
if __name__ == '__main__':
    # TODO: Change to dynamic smoke
    world_x_size, world_y_size = 80, 50

    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=4.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=10, y_pos=20, intensity=1.0, spread_rate=5.0),
        SmokeBlobParams(x_pos=60, y_pos=20, intensity=1.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=50, y_pos=10, intensity=1.0, spread_rate=5.0),
    ]

    sensor_params = DownwardsSensorParams(world_x_size=world_x_size, world_y_size=world_y_size)
    smoke_params = DynamicSmokeParams(x_size=world_x_size, y_size=world_y_size, smoke_blob_params=smoke_blob_params, resolution=0.3, fov_sensor_params=sensor_params)
    smoke_simulator = DynamicSmoke(params=smoke_params)

    gp = GaussianProcess(online=True)

    steps = 30
    sample_size_per_step = 20
    dt = 0.1
    t = 0.0
    for step in tqdm(range(steps)):
        X_sample = np.concatenate([np.random.uniform(5, 15, sample_size_per_step).reshape(-1, 1),
                                   np.random.uniform(28, 38, sample_size_per_step).reshape(-1, 1),
                                   np.full((sample_size_per_step, 1), t)], axis=1)
        t += dt
        y_observe = smoke_simulator.get_smoke_density(X_sample[:, :2])
        gp.track_data(X_sample, y_observe)
        smoke_simulator.step(dt=dt)
        gp.update()

    y_true = smoke_simulator.get_smoke_map()

    map_resolution = 1.0
    f, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].imshow(y_true, vmin=0, vmax=y_true.max(), extent=[0, world_x_size, 0, world_y_size], origin='lower', cmap='gray')
    ax[0, 0].set_title('Ground truth')
    plot_dynamic_map(gp, world_x_size, world_y_size, t, map_resolution, fig=f, ax=ax[0, 1], plot_type='mean')
    ax[0, 1].set_title('Predicted')
    plot_dynamic_map(gp, world_x_size, world_y_size, t, map_resolution, fig=f, ax=ax[1, 0], plot_type='std')
    ax[1, 0].set_title('Std')
    plot_dynamic_map(gp, world_x_size, world_y_size, t, map_resolution, fig=f, ax=ax[1, 1], plot_type='cvar')
    ax[1, 1].set_title('Cvar')
    plt.show()


