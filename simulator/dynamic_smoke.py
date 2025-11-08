import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.path import Path
from phi import flow
import phi.field

from src.utils import *
from simulator.sensor import DownwardsSensorParams, DownwardsSensor

@dataclass
class SmokeBlobParams:
    x_pos: int
    y_pos: int
    intensity: float
    spread_rate: float

@dataclass
class DynamicSmokeParams:
    x_size: float
    y_size: float
    smoke_blob_params: list[SmokeBlobParams]
    resolution: float

    # Parameters for the smoke simulation
    average_wind_speed: float = 15.0
    smoke_emission_rate: float = 0.7
    smoke_diffusion_rate: float = 0.3
    smoke_decay_rate: float = 1.0
    fov_sensor_params: DownwardsSensorParams | None = None

class DynamicSmoke:
    def __init__(self, params: DynamicSmokeParams | None = None):
        self.params = params

        assert self.params.x_size >= self.params.resolution or self.params.y_size >= self.params.resolution, "Resolution must be smaller than the size of the world"
        x_resolution = int(self.params.x_size / self.params.resolution)
        y_resolution = int(self.params.y_size / self.params.resolution)

        self.scalar_resolution = self.params.resolution
        self.spatial_resolution = flow.spatial(x=x_resolution, y=y_resolution)
        self.bounds = flow.Box(x=self.params.x_size, y=self.params.y_size)
        self.smoke_blob_params = self.params.smoke_blob_params

        self.sensor = DownwardsSensor(self.params.fov_sensor_params) if self.params.fov_sensor_params is not None else None

        self.smoke_map = self.build_smoke_map(self.params.smoke_blob_params)
        self.velocity = self.build_velocity()

        smoke_top = flow.CenteredGrid(1, flow.extrapolation.BOUNDARY,resolution=self.spatial_resolution, bounds=self.bounds)
        smoke_zero = flow.CenteredGrid(0, flow.extrapolation.BOUNDARY,resolution=self.spatial_resolution, bounds=self.bounds)
        self.smoke_top = smoke_top
        self.smoke_zero = smoke_zero

    def reset(self):
        self.smoke_map = self.build_smoke_map(self.params.smoke_blob_params)
        self.velocity = self.build_velocity()

    def step(self, dt: float = 0.1):
        wind_force = 0.1 * self.params.average_wind_speed * flow.StaggeredGrid(flow.Noise(smoothness=0.8), flow.extrapolation.ZERO,resolution=self.spatial_resolution, bounds=self.bounds) 
        wind_force, _ = flow.fluid.make_incompressible(wind_force, (), flow.Solve(rank_deficiency=0))

        self.velocity += wind_force
        self.velocity = flow.advect.semi_lagrangian(self.velocity, self.velocity, dt=dt)
        self.velocity, _ = flow.fluid.make_incompressible(self.velocity, (), flow.Solve(rank_deficiency=0))

        kappa = self.params.smoke_diffusion_rate * self.scalar_resolution**2 / 4 / dt #0.3        # diffusion coeff
        tau = self.params.smoke_decay_rate        # decay time-constant
        emit_rate = self.params.smoke_emission_rate   # emission strength (scale as you like)

        # 1) Source / inflow (scaled by dt)
        # assume self.inflow_map is a [0..1] mask or field with source strength
        self.smoke_map = self.smoke_map + emit_rate * self.inflow_map

        # 2) Advection
        # Mac Cormack only
        self.smoke_map = flow.advect.mac_cormack(self.smoke_map, self.velocity, dt=dt)

        # mac cormack + boundry handling
        # pred = flow.advect.semi_lagrangian(self.smoke_map, self.velocity, dt=dt)
        # back = flow.advect.semi_lagrangian(pred,         -self.velocity, dt=dt)
        # mc   = pred + 0.5 * (self.smoke_map - back)
        # lo = flow.field.minimum(self.smoke_map, pred)
        # hi = flow.field.maximum(self.smoke_map, pred)
        # self.smoke_map = flow.field.maximum(flow.field.minimum(mc, hi), lo)

        # 3) Diffusion
        self.smoke_map = flow.diffuse.explicit(self.smoke_map, kappa, dt=dt)

        # 4) Exponential decay
        self.smoke_map = self.smoke_map * flow.math.exp(-dt / tau)

        # 5) Clamp to valid range
        self.smoke_map = phi.field.maximum(
                phi.field.minimum(self.smoke_map, self.smoke_top),
                self.smoke_zero
        )

    def build_smoke_map(self, smoke_blob_params: list[SmokeBlobParams]):
        inflow_map = flow.CenteredGrid(0, flow.extrapolation.BOUNDARY,resolution=self.spatial_resolution, bounds=self.bounds)  # sampled at cell centers

        for blob in smoke_blob_params:
            loc = flow.tensor([(blob.x_pos, blob.y_pos)], flow.batch('inflow_loc'), flow.channel(vector='x,y'))
            sphere = flow.Sphere(center=loc, radius=blob.spread_rate)
            inflow = flow.CenteredGrid(flow.Sphere(center=loc, radius=blob.spread_rate), flow.extrapolation.BOUNDARY,resolution=self.spatial_resolution, bounds=self.bounds)
            inflow_map += inflow

        self.inflow_map = inflow_map
        return inflow_map

    def build_velocity(self):
        velocity = self.params.average_wind_speed * flow.StaggeredGrid(flow.Noise(smoothness=0.5), flow.extrapolation.ZERO,resolution=self.spatial_resolution, bounds=self.bounds)
        velocity, _ = flow.fluid.make_incompressible(velocity, (), flow.Solve(rank_deficiency=0))
        return velocity

    def get_smoke_density_at_point(self, x, y) -> float:
        """
        x: float
        y: float
        return: float
        """

        pos = flow.tensor([(x, y)], flow.batch('inflow_loc'), flow.channel(vector='x,y'))

        pos_smoke = self.smoke_map.sample(pos).numpy("inflow_loc")

        return pos_smoke[0]
        
    def get_smoke_density(self, pos: np.ndarray) -> np.ndarray:
        """
        pos: nx2 array or 1x2 array
        return: (n,1) array
        """

        if pos.ndim == 1:
            return np.array([[self.get_smoke_density_at_point(pos[0], pos[1])]])

        assert pos.shape[1] == 2 and pos.ndim == 2, "Position must be a nx2 array"
        
        densities = []
        for i in range(pos.shape[0]):
            x, y = pos[i]
            density = self.get_smoke_density_at_point(x, y)
            densities.append(density)
        return np.array(densities).reshape(-1, 1)

    def get_smoke_density_downwards_sensor(self, pos: np.ndarray, return_location: bool = False) -> np.ndarray:
        assert self.sensor is not None, "Sensor must have been initialized"

        sensor_output = self.sensor.read(self.get_smoke_density, curr_pos=pos)
        if return_location:
            return sensor_output["sensor_readings"], sensor_output["sensor_position_readings"]
        return sensor_output["sensor_readings"]

    def get_smoke_map(self):
        smoke_arr = self.smoke_map.values.numpy(('y','x', 'inflow_loc'))
        smoke_arr = smoke_arr.squeeze()
        return smoke_arr

    def get_smoke_extent(self):
        b = self.smoke_map.bounds
        extent = [b.lower[0].numpy(), b.upper[0].numpy(), b.lower[1].numpy(), b.upper[1].numpy()]
        return extent
    
    def plot_smoke_map(self, fig: plt.Figure = None, ax: plt.Axes = None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        extent = self.get_smoke_extent()
        smoke_arr = self.get_smoke_map()

        if ax.images:
            ax.images[0].set_array(smoke_arr)
        else:
            ax_ = ax.imshow(smoke_arr, cmap='gray', extent=extent, origin='lower')
            fig.colorbar(ax_, label='Smoke Density')
            ax.set_title('Static Smoke Map')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')

        if self.sensor is not None:
            pairs_to_plot = self.sensor.grid_pairs_positions + np.array([10, 30])
            ax.scatter(pairs_to_plot[:, 0], pairs_to_plot[:, 1], color='red', s=0.1)

        fig.canvas.draw()


if __name__ == "__main__":
    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=1.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=20, y_pos=20, intensity=1.0, spread_rate=5.0),
        SmokeBlobParams(x_pos=20, y_pos=40, intensity=1.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=30, y_pos=30, intensity=1.0, spread_rate=5.0),
    ]
    
    world_x_size = 60
    world_y_size = 50
    sensor_params = DownwardsSensorParams(world_x_size=world_x_size, world_y_size=world_y_size)
    smoke_params = DynamicSmokeParams(x_size=world_x_size, y_size=world_y_size, smoke_blob_params=smoke_blob_params, resolution=0.3, fov_sensor_params=sensor_params)
    smoke_simulator = DynamicSmoke(params=smoke_params)


    fig, ax = plt.subplots()
    for i in range(100):
        smoke_simulator.step(dt=0.1)
        smoke_simulator.plot_smoke_map(fig=fig, ax=ax)
        print(np.round(smoke_simulator.get_smoke_density(np.array([[10, 40],[40, 10]])), 2))
        print(np.round(smoke_simulator.get_smoke_density_downwards_sensor(np.array([10, 30]), return_location=False), 2).shape)
        plt.draw()
        plt.pause(0.1)

    plt.show()




    # def get_smoke_density_within_polygon(self, polygon: np.ndarray, return_location: bool = False) -> np.ndarray:
    #     """
    #     polygon: nx2 array
    #     return: (n,1) array, optional: (n,2) array
    #     """
    #     assert polygon.shape[1] == 2 and polygon.ndim == 2, "Polygon must be a nx2 array"

    #     polygon_in_discrete = [world_to_index(p[0], p[1], self.x_size, self.y_size, 1) for p in polygon]

    #     cols, rows = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size))

    #     pts = np.column_stack([rows.ravel(), cols.ravel()])

    #     path = Path(polygon_in_discrete)
    #     inside = path.contains_points(pts)
    #     inside_map = inside.reshape(self.y_size, self.x_size)

    #     smoke_map = self.smoke_map.values.numpy(('x', 'y', 'inflow_loc')).squeeze().T
    #     values = smoke_map[inside_map]

    #     if return_location:
    #         pts_in_discrete = pts[inside]
    #         pts_in_world = [index_to_world(p[0], p[1], self.x_size, self.y_size, 1) for p in pts_in_discrete]
    #         pts_in_world = np.array(pts_in_world)
    #         return values.reshape(-1, 1), pts_in_world

    #     return values.reshape(-1, 1)




