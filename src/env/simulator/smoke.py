import matplotlib.pyplot as plt
import numpy as np
import phi.field
import torch
from phi.torch import flow

from src.env.simulator.base_smoke_simulator import BaseSmokeSimulator
from src.env.simulator.schemas import BlobConfig, SmokeConfig


class Smoke(BaseSmokeSimulator):
    def __init__(self, cfg: SmokeConfig):
        """Initialize the smoke.

        :param cfg: Parameters for the smoke simulation.
        """
        if torch.cuda.is_available():
            device_name = "GPU"
        else:
            device_name = "CPU"
        flow.TORCH.set_default_device(device_name)

        self.cfg = cfg
        self.blob_cfg_list = cfg.blobs

        assert self.cfg.x_size >= self.cfg.resolution or self.cfg.y_size >= self.cfg.resolution, (
            "Resolution must be smaller than the size of the world"
        )
        x_resolution = int(self.cfg.x_size / self.cfg.resolution)
        y_resolution = int(self.cfg.y_size / self.cfg.resolution)

        self.scalar_resolution = self.cfg.resolution
        self.spatial_resolution = flow.spatial(x=x_resolution, y=y_resolution)
        self.bounds = flow.Box(x=self.cfg.x_size, y=self.cfg.y_size)

        self.inflow_bank = []
        for _ in range(self.cfg.inflow_bank_count):
            self.inflow_bank.append(self.build_smoke_map(self.blob_cfg_list))

        self.smoke_map = self.inflow_bank[0]
        self.smoke_map = flow.diffuse.explicit(self.smoke_map, diffusivity=0.1, dt=0.1)

        self.velocity = self.build_velocity()

        smoke_top = flow.CenteredGrid(
            1, flow.extrapolation.BOUNDARY, resolution=self.spatial_resolution, bounds=self.bounds
        )
        smoke_zero = flow.CenteredGrid(
            0, flow.extrapolation.BOUNDARY, resolution=self.spatial_resolution, bounds=self.bounds
        )
        self.smoke_top = smoke_top
        self.smoke_zero = smoke_zero

    def reset(self):
        self.smoke_map = self.inflow_bank[0]
        self.smoke_map = flow.diffuse.explicit(self.smoke_map, diffusivity=0.1, dt=0.1)
        self.velocity = self.build_velocity()

    def step(self, dt: float = 0.1):
        # 1. --- VELOCITY FORCES (NEW) ---
        # Smoke (density) generates upward force (Buoyancy)
        # We resample the smoke to the center of the velocity cells (Staggered)
        smoke_centered = self.smoke_map.at(self.velocity)

        # We create an upward force (y-axis = 1) proportional to the smoke density
        # This creates the "mushroom-like" plumes
        buoyancy_force = smoke_centered * (0, self.cfg.buoyancy_factor)

        # Apply force to velocity
        self.velocity = self.velocity + buoyancy_force * dt

        # 2. --- VELOCITY ADVECTION ---
        self.velocity = flow.advect.semi_lagrangian(self.velocity, self.velocity, dt=dt)
        self.velocity, _ = flow.fluid.make_incompressible(
            self.velocity,
            (),
            flow.Solve(
                rank_deficiency=0,
                rel_tol=1e-4,
                abs_tol=1e-4,
                max_iterations=2000,
                suppress=(flow.math.NotConverged,),
            ),
        )

        # --- Parameters ---
        tau = self.cfg.smoke_decay_rate
        emit_rate = self.cfg.smoke_emission_rate

        # 3. --- SOURCE / INFLOW ---
        # Use random texture bank
        idx = int(flow.math.random_uniform(low=0, high=len(self.inflow_bank)))

        # Multiply by dt for physical consistency
        self.smoke_map = self.smoke_map + emit_rate * self.inflow_bank[idx] * dt

        # 4. --- SMOKE ADVECTION ---
        self.smoke_map = flow.advect.semi_lagrangian(self.smoke_map, self.velocity, dt=dt)

        # 4.5 Diffusion (implicit solver: unconditionally stable, no CFL constraint)
        if self.cfg.smoke_diffusion_rate > 0:
            self.smoke_map = flow.diffuse.implicit(
                self.smoke_map, diffusivity=self.cfg.smoke_diffusion_rate, dt=dt
            )

        # 4) Exponential Decay
        if self.cfg.smoke_decay_rate > 0:
            self.smoke_map = self.smoke_map * flow.math.exp(-dt / tau)

        # 5) Clamp (Final cleanup)
        self.smoke_map = phi.field.maximum(
            phi.field.minimum(self.smoke_map, self.smoke_top), self.smoke_zero
        )

    def build_smoke_map(self, blob_cfg_list: list[BlobConfig]):
        # Initialize empty map
        inflow_map = flow.CenteredGrid(
            0, flow.extrapolation.BOUNDARY, resolution=self.spatial_resolution, bounds=self.bounds
        )

        for blob in blob_cfg_list:
            # 1. Define location and base shape (Sphere/Mask)
            loc = flow.tensor(
                [(blob.x_pos, blob.y_pos)], flow.batch("inflow_loc"), flow.channel(vector="x,y")
            )
            sphere_shape = flow.Sphere(center=loc, radius=blob.spread_rate)

            # Convert sphere to a grid (0 outside, 1 inside)
            sphere_mask = flow.CenteredGrid(
                sphere_shape,
                flow.extrapolation.BOUNDARY,
                resolution=self.spatial_resolution,
                bounds=self.bounds,
            )

            # 2. Generate Noise (Texture)
            # scale: controls how "large" the smoke clumps are.
            # smoothness: smooths the noise to look like smoke and not TV static.
            noise_grid = flow.CenteredGrid(
                flow.Noise(scale=blob.spread_rate * 0.5, smoothness=0.8),
                flow.extrapolation.BOUNDARY,
                resolution=self.spatial_resolution,
                bounds=self.bounds,
            )

            # 3. Combine: Normalize noise and multiply by the sphere
            # Noise is from -1 to 1. Map to 0 to 1 with (noise + 1) / 2
            texture = (noise_grid + 1) / 2

            # Multiply: (Sphere Shape) * (Noise Texture) * (Intensity)
            blob_inflow = sphere_mask * texture * blob.intensity

            inflow_map += blob_inflow

        return inflow_map

    def build_velocity(self):
        velocity = self.cfg.average_wind_speed * flow.StaggeredGrid(
            flow.Noise(smoothness=0.4),
            flow.extrapolation.ZERO,
            resolution=self.spatial_resolution,
            bounds=self.bounds,
        )
        velocity, _ = flow.fluid.make_incompressible(
            velocity,
            (),
            flow.Solve(
                rank_deficiency=0,
                rel_tol=1e-4,
                abs_tol=1e-4,
                max_iterations=2000,
                suppress=(flow.math.NotConverged,),
            ),
        )
        return velocity

    def get_smoke_map_tensor(self) -> torch.Tensor:
        """Returns the current 2D smoke density map as a native PyTorch tensor on the active device.

        Returns:
            torch.Tensor: A 2D tensor of shape (H, W) representing smoke densities.
        """
        # The underlying tensor has dimensions ('y', 'x', 'inflow_loc')
        tensor = self.smoke_map.values.native(("y", "x", "inflow_loc"))
        return tensor.squeeze()

    def get_smoke_density(self, pos: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Function to get smoke density at multiple points natively on GPU/device.

        pos: nx2 array/tensor or 1x2 array/tensor
        return: (n,1) array/tensor
        """
        is_numpy = isinstance(pos, np.ndarray)
        if is_numpy:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
        else:
            pos_t = pos

        if pos_t.ndim == 1:
            pos_t = pos_t.reshape(1, 2)

        assert pos_t.shape[1] == 2 and pos_t.ndim == 2, "Position must be a nx2 array/tensor"

        # Map physical positions to normalized coordinates [-1, 1] for grid_sample.
        # In grid_sample, coordinates must be in range [-1, 1], where:
        # -1 represents the border of the first grid cell, 1 represents the border of the last.
        x_norm = 2.0 * (pos_t[:, 0] / self.cfg.x_size) - 1.0
        y_norm = 2.0 * (pos_t[:, 1] / self.cfg.y_size) - 1.0

        # grid_sample expects grid format of [x, y] coordinates
        grid_coords = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)

        # Get map tensor (H, W) -> (B, C, H, W)
        smoke_map_t = self.get_smoke_map_tensor().unsqueeze(0).unsqueeze(0)

        # Perform bilinear interpolation in GPU memory
        sampled = torch.nn.functional.grid_sample(
            smoke_map_t, grid_coords, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        values = sampled.view(-1, 1)

        if is_numpy:
            return values.cpu().numpy()
        return values

    def get_smoke_map(self):
        smoke_arr = self.smoke_map.values.numpy(("y", "x", "inflow_loc"))
        smoke_arr = smoke_arr.squeeze()
        return smoke_arr

    def get_smoke_extent(self):
        b = self.smoke_map.bounds
        extent = [b.lower[0].numpy(), b.upper[0].numpy(), b.lower[1].numpy(), b.upper[1].numpy()]
        return extent


def run_smoke_test() -> None:
    """Run a smoke simulation test."""
    smoke_cfg = SmokeConfig()

    blob_cfg_list = [
        BlobConfig(x_pos=10, y_pos=15, intensity=1.0, spread_rate=4.0),
        BlobConfig(x_pos=20, y_pos=20, intensity=1.0, spread_rate=5.0),
        BlobConfig(x_pos=20, y_pos=15, intensity=1.0, spread_rate=3.0),
        BlobConfig(x_pos=25, y_pos=5, intensity=1.0, spread_rate=5.0),
    ]
    smoke_cfg = SmokeConfig(blobs=blob_cfg_list)

    smoke = Smoke(cfg=smoke_cfg)

    fig, ax = plt.subplots()
    for _ in range(100):
        smoke.step(dt=0.1)
        smoke.plot_smoke_map(fig=fig, ax=ax)
        print(np.round(smoke.get_smoke_density(np.array([[10, 40], [40, 10]])), 2))
        plt.draw()
        plt.pause(0.1)

    plt.show()


if __name__ == "__main__":
    run_smoke_test()
