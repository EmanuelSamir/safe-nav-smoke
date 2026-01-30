import numpy as np
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.path import Path
from phi import flow
import phi.field

from src.utils import *
from simulator.sensor import DownwardsSensorParams, DownwardsSensor, PointSensor, GlobalSensor, BaseSensorParams

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
    average_wind_speed: float = 8.0
    smoke_emission_rate: float = 2.0
    smoke_diffusion_rate: float = 2.0 #1.0 #0.5
    smoke_decay_rate: float = 1.5
    buoyancy_factor: float = 1.2
    # sensor_params: BaseSensorParams | None = None

class DynamicSmoke:
    def __init__(self, params: Optional[DynamicSmokeParams] = None):
        self.params = params

        assert self.params.x_size >= self.params.resolution or self.params.y_size >= self.params.resolution, "Resolution must be smaller than the size of the world"
        x_resolution = int(self.params.x_size / self.params.resolution)
        y_resolution = int(self.params.y_size / self.params.resolution)

        self.scalar_resolution = self.params.resolution
        self.spatial_resolution = flow.spatial(x=x_resolution, y=y_resolution)
        self.bounds = flow.Box(x=self.params.x_size, y=self.params.y_size)
        self.smoke_blob_params = self.params.smoke_blob_params

        self.inflow_bank = []
        for _ in range(5):
            self.inflow_bank.append(self.build_smoke_map(self.params.smoke_blob_params))
        
        self.smoke_map = self.inflow_bank[0]
        self.smoke_map = flow.diffuse.explicit(self.smoke_map, diffusivity=0.1, dt=0.1)

        self.velocity = self.build_velocity()

        smoke_top = flow.CenteredGrid(1, flow.extrapolation.BOUNDARY,resolution=self.spatial_resolution, bounds=self.bounds)
        smoke_zero = flow.CenteredGrid(0, flow.extrapolation.BOUNDARY,resolution=self.spatial_resolution, bounds=self.bounds)
        self.smoke_top = smoke_top
        self.smoke_zero = smoke_zero

    def reset(self):
        self.smoke_map = self.inflow_bank[0]
        self.smoke_map = flow.diffuse.explicit(self.smoke_map, diffusivity=0.1, dt=0.1)
        self.velocity = self.build_velocity()

    def step(self, dt: float = 0.1):
        # 1. --- FUERZAS DE VELOCIDAD (NUEVO) ---
        # El humo (densidad) genera fuerza hacia arriba (Flotabilidad)
        # Resampleamos el humo al centro de las celdas de velocidad (Staggered)
        smoke_centered = self.smoke_map.at(self.velocity)
        
        # Creamos una fuerza hacia arriba (eje y=1) proporcional a la densidad del humo
        # Esto crea los remolinos "tipo hongo"
        buoyancy_force = smoke_centered * (0, self.params.buoyancy_factor) 
        
        # Aplicamos la fuerza a la velocidad
        self.velocity = self.velocity + buoyancy_force * dt

        # 2. --- ADVECCIÓN DE VELOCIDAD ---
        self.velocity = flow.advect.semi_lagrangian(self.velocity, self.velocity, dt=dt)
        self.velocity, _ = flow.fluid.make_incompressible(self.velocity, (), flow.Solve(rank_deficiency=0))

        # --- Parámetros ---
        diffusion_amount = self.params.smoke_diffusion_rate 
        tau = self.params.smoke_decay_rate
        emit_rate = self.params.smoke_emission_rate

        # 3. --- SOURCE / INFLOW ---
        # Usamos tu banco de texturas random
        idx = np.random.randint(len(self.inflow_bank))
        # Multiplicamos por dt para consistencia física
        self.smoke_map = self.smoke_map + emit_rate * self.inflow_bank[idx] * dt

        # 4. --- ADVECCIÓN DE HUMO ---
        self.smoke_map = flow.advect.semi_lagrangian(self.smoke_map, self.velocity, dt=dt)

        # 4) Decaimiento Exponencial
        self.smoke_map = self.smoke_map * flow.math.exp(-dt / tau)

        # 5) Clamp (Limpieza final)
        self.smoke_map = phi.field.maximum(
                phi.field.minimum(self.smoke_map, self.smoke_top),
                self.smoke_zero
        )

    def build_smoke_map(self, smoke_blob_params: list[SmokeBlobParams]):
        # Inicializamos el mapa vacío
        inflow_map = flow.CenteredGrid(0, flow.extrapolation.BOUNDARY, resolution=self.spatial_resolution, bounds=self.bounds)

        for blob in smoke_blob_params:
            # 1. Definir la ubicación y la forma base (La Esfera/Mascara)
            loc = flow.tensor([(blob.x_pos, blob.y_pos)], flow.batch('inflow_loc'), flow.channel(vector='x,y'))
            sphere_shape = flow.Sphere(center=loc, radius=blob.spread_rate)
            
            # Convertimos la esfera a una rejilla (0 fuera, 1 dentro)
            sphere_mask = flow.CenteredGrid(sphere_shape, flow.extrapolation.BOUNDARY, resolution=self.spatial_resolution, bounds=self.bounds)

            # 2. Generar el Ruido (La Textura)
            # scale: controla qué tan "grandes" son los grumos de humo. 
            # smoothness: suaviza el ruido para que parezca humo y no estática de TV.
            noise_grid = flow.CenteredGrid(
                flow.Noise(scale=blob.spread_rate * 0.5, smoothness=0.8), 
                flow.extrapolation.BOUNDARY, 
                resolution=self.spatial_resolution, 
                bounds=self.bounds
            )

            # 3. Combinar: Normalizamos el ruido y lo multiplicamos por la esfera
            # El ruido viene de -1 a 1. Lo pasamos a 0 a 1 con (noise + 1) / 2
            texture = (noise_grid + 1) / 2
            
            # Multiplicamos: (Forma Esfera) * (Textura Ruido) * (Intensidad)
            blob_inflow = sphere_mask * texture * blob.intensity
            
            inflow_map += blob_inflow
        
        return inflow_map

    def build_velocity(self):
        velocity = self.params.average_wind_speed * flow.StaggeredGrid(flow.Noise(smoothness=0.4), flow.extrapolation.ZERO,resolution=self.spatial_resolution, bounds=self.bounds)
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
            pos = pos.reshape(1, 2)

        assert pos.shape[1] == 2 and pos.ndim == 2, "Position must be a nx2 array"
        
        # Vectorized sampling using phi.flow
        # Create a single tensor with all query points in a 'points' batch dimension
        pos_tensor = flow.tensor(pos, flow.batch('points'), flow.channel(vector='x,y'))
        
        # Sample the smoke map at these coordinates (interpolated)
        sampled_values = self.smoke_map.sample(pos_tensor)
        
        # Use single string for dimension order as suggested by phiml error message
        values = sampled_values.numpy('inflow_loc,points')
        
        return values.reshape(-1, 1)

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
            ax_ = ax.imshow(smoke_arr, cmap='gray', extent=extent, origin='lower', vmin=0, vmax=1)
            fig.colorbar(ax_, label='Smoke Density')
            ax.set_title('Static Smoke Map')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')

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
    smoke_params = DynamicSmokeParams(x_size=world_x_size, y_size=world_y_size, smoke_blob_params=smoke_blob_params, resolution=0.3)
    smoke_simulator = DynamicSmoke(params=smoke_params)


    fig, ax = plt.subplots()
    for i in range(100):
        smoke_simulator.step(dt=0.1)
        smoke_simulator.plot_smoke_map(fig=fig, ax=ax)
        print(np.round(smoke_simulator.get_smoke_density(np.array([[10, 40],[40, 10]])), 2))
        plt.draw()
        plt.pause(0.1)

    plt.show()




