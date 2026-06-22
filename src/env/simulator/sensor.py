import math
from dataclasses import dataclass

import hydra
import torch


@dataclass
class SensorOutput:
    readings: torch.Tensor
    positions: torch.Tensor


@dataclass
class BaseSensorParams:
    world_x_size: float
    world_y_size: float
    sensor_type: str


@dataclass
class GlobalSensorParams(BaseSensorParams):
    density_reading_per_unit_length: float


@dataclass
class DownwardsSensorParams(BaseSensorParams):
    density_reading_per_unit_length: float

    x_fov_size: float
    y_fov_size: float


@dataclass
class Camera1DSensorParams(BaseSensorParams):
    fov_size_degrees: float
    num_rays: int
    step_size: float
    opacity_threshold: float
    max_range: float


class BaseSensor:
    def __init__(self, params: BaseSensorParams):
        """Initialize the base sensor.

        Args:
            params: Parameters for the sensor.
        """
        self.params = params

    def projection_bounds(self, pos_x: float, pos_y: float) -> torch.Tensor:
        raise NotImplementedError("Projection bounds must be implemented in the subclass")

    def read(self, simulator, curr_pos: torch.Tensor) -> SensorOutput:
        raise NotImplementedError("Read must be implemented in the subclass")


class DownwardsSensor(BaseSensor):
    def __init__(self, params: DownwardsSensorParams):
        self.params = params

        density = self.params.density_reading_per_unit_length
        if density == 0:
            density = 1.0

        self.nx = max(1, round(density * self.params.x_fov_size))
        self.ny = max(1, round(density * self.params.y_fov_size))

    def projection_bounds(self, pos_x: float, pos_y: float) -> torch.Tensor:
        half_x = self.params.x_fov_size / 2
        half_y = self.params.y_fov_size / 2
        return torch.tensor(
            [
                [pos_x - half_x, pos_y - half_y],
                [pos_x + half_x, pos_y - half_y],
                [pos_x + half_x, pos_y + half_y],
                [pos_x - half_x, pos_y + half_y],
            ],
            dtype=torch.float32,
        )

    def read(self, simulator, curr_pos: torch.Tensor) -> SensorOutput:
        grid = simulator.get_smoke_map_tensor()  # shape: (H, W)
        curr_pos = curr_pos.to(grid.device)
        device = curr_pos.device
        center_pos = curr_pos[:2]

        resolution = getattr(simulator, "resolution", getattr(simulator.params, "resolution", None))
        H, W = grid.shape

        density = self.params.density_reading_per_unit_length
        if density == 0:
            # Exact crop slicing without interpolation
            nx = max(1, round(self.params.x_fov_size / resolution))
            ny = max(1, round(self.params.y_fov_size / resolution))

            center_grid_x = center_pos[0] / resolution
            center_grid_y = center_pos[1] / resolution

            start_x = round(center_grid_x.item() - nx / 2)
            start_y = round(center_grid_y.item() - ny / 2)

            x_indices = torch.arange(start_x, start_x + nx, device=device, dtype=torch.float32)
            y_indices = torch.arange(start_y, start_y + ny, device=device, dtype=torch.float32)

            y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing="ij")
            positions = torch.stack(
                [(x_grid + 0.5) * resolution, (y_grid + 0.5) * resolution], dim=-1
            ).view(-1, 2)

            # Filter valid physical bounds
            in_bounds_phys = (
                (positions[:, 0] >= 0)
                & (positions[:, 0] <= self.params.world_x_size)
                & (positions[:, 1] >= 0)
                & (positions[:, 1] <= self.params.world_y_size)
            )
            valid_positions = positions[in_bounds_phys]

            if valid_positions.shape[0] == 0:
                return SensorOutput(
                    readings=torch.zeros((0, 1), device=device),
                    positions=torch.zeros((0, 2), device=device),
                )

            # Map coordinates to grid indices
            grid_x = torch.round(valid_positions[:, 0] / resolution - 0.5).long()
            grid_y = torch.round(valid_positions[:, 1] / resolution - 0.5).long()

            # Clamp to prevent indexing errors (out of bounds values will get 0)
            in_bounds_grid = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)

            readings = torch.zeros((valid_positions.shape[0], 1), device=device, dtype=grid.dtype)
            readings[in_bounds_grid, 0] = grid[grid_y[in_bounds_grid], grid_x[in_bounds_grid]]

            return SensorOutput(readings=readings, positions=valid_positions)
        else:
            # Resampling using grid_sample bilinear interpolation
            dx = self.params.x_fov_size / self.nx
            dy = self.params.y_fov_size / self.ny

            x_range = (
                torch.arange(self.nx, device=device, dtype=torch.float32) - self.nx / 2 + 0.5
            ) * dx
            y_range = (
                torch.arange(self.ny, device=device, dtype=torch.float32) - self.ny / 2 + 0.5
            ) * dy

            y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
            relative_grid = torch.stack([x_grid.ravel(), y_grid.ravel()], dim=-1)

            positions = center_pos + relative_grid

            in_bounds = (
                (positions[:, 0] >= 0)
                & (positions[:, 0] <= self.params.world_x_size)
                & (positions[:, 1] >= 0)
                & (positions[:, 1] <= self.params.world_y_size)
            )
            valid_positions = positions[in_bounds]

            if valid_positions.shape[0] == 0:
                return SensorOutput(
                    readings=torch.zeros((0, 1), device=device),
                    positions=torch.zeros((0, 2), device=device),
                )

            x_norm = 2.0 * (valid_positions[:, 0] / self.params.world_x_size) - 1.0
            y_norm = 2.0 * (valid_positions[:, 1] / self.params.world_y_size) - 1.0
            grid_coords = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)

            grid_unsqueezed = grid.unsqueeze(0).unsqueeze(0)
            sampled = torch.nn.functional.grid_sample(
                grid_unsqueezed,
                grid_coords,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            readings = sampled.view(-1, 1)
            return SensorOutput(readings=readings, positions=valid_positions)


class GlobalSensor(BaseSensor):
    def __init__(self, params: GlobalSensorParams):
        self.params = params

    def projection_bounds(self, pos_x: float, pos_y: float) -> torch.Tensor:
        return torch.tensor(
            [
                [0.0, 0.0],
                [self.params.world_x_size, 0.0],
                [self.params.world_x_size, self.params.world_y_size],
                [0.0, self.params.world_y_size],
            ],
            dtype=torch.float32,
        )

    def read(self, simulator, curr_pos: torch.Tensor) -> SensorOutput:
        grid = simulator.get_smoke_map_tensor()  # shape: (H, W)
        device = grid.device

        # If density_reading_per_unit_length is 0, we do not resample
        if self.params.density_reading_per_unit_length == 0:
            readings = grid.ravel().unsqueeze(-1)
            ny, nx = grid.shape
            dx = self.params.world_x_size / nx
            dy = self.params.world_y_size / ny
        else:
            nx = max(
                1, round(self.params.density_reading_per_unit_length * self.params.world_x_size)
            )
            ny = max(
                1, round(self.params.density_reading_per_unit_length * self.params.world_y_size)
            )

            dx = self.params.world_x_size / nx
            dy = self.params.world_y_size / ny

            # Resample using torch.nn.functional.interpolate
            grid_unsqueezed = grid.unsqueeze(0).unsqueeze(0)
            resampled = torch.nn.functional.interpolate(
                grid_unsqueezed, size=(ny, nx), mode="bilinear", align_corners=False
            )
            readings = resampled.squeeze().ravel().unsqueeze(-1)

        # Build positions grid
        x_range = (torch.arange(nx, device=device, dtype=torch.float32) + 0.5) * dx
        y_range = (torch.arange(ny, device=device, dtype=torch.float32) + 0.5) * dy
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
        positions = torch.stack([x_grid.ravel(), y_grid.ravel()], dim=-1)

        return SensorOutput(readings=readings, positions=positions)


class Camera1DSensor(BaseSensor):
    def __init__(self, params: Camera1DSensorParams):
        self.params = params
        self.fov_size_rad = math.radians(self.params.fov_size_degrees)

    def projection_bounds(self, pos_x: float, pos_y: float) -> torch.Tensor:
        return torch.tensor(
            [
                [pos_x - self.params.max_range, pos_y - self.params.max_range],
                [pos_x + self.params.max_range, pos_y - self.params.max_range],
                [pos_x + self.params.max_range, pos_y + self.params.max_range],
                [pos_x - self.params.max_range, pos_y + self.params.max_range],
            ],
            dtype=torch.float32,
        )

    def read(self, simulator, curr_pos: torch.Tensor) -> SensorOutput:
        # curr_pos: [x, y, theta]
        grid = simulator.get_smoke_map_tensor()
        curr_pos = curr_pos.to(grid.device)
        device = curr_pos.device
        pos_x, pos_y, theta = curr_pos[0], curr_pos[1], curr_pos[2]

        # Calculate the angle of each ray within the field of view
        angles = torch.linspace(
            theta - self.fov_size_rad / 2,
            theta + self.fov_size_rad / 2,
            self.params.num_rays,
            device=device,
        )
        ray_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        # Pre-compute all distances to sample along each ray
        max_steps = int(self.params.max_range / self.params.step_size)
        ray_distances = (
            torch.arange(1, max_steps + 1, device=device, dtype=torch.float32)
            * self.params.step_size
        )

        # Generate sampling points: (num_rays, max_steps, 2)
        ray_offsets = ray_dirs.unsqueeze(1) * ray_distances.unsqueeze(0).unsqueeze(2)
        ray_points = curr_pos[:2] + ray_offsets

        # Batch query all generated point densities directly from the full grid
        flat_ray_points = ray_points.view(-1, 2)
        x_norm = 2.0 * (flat_ray_points[:, 0] / self.params.world_x_size) - 1.0
        y_norm = 2.0 * (flat_ray_points[:, 1] / self.params.world_y_size) - 1.0
        grid_coords = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)

        grid_unsqueezed = grid.unsqueeze(0).unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(
            grid_unsqueezed, grid_coords, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        densities = sampled.view(self.params.num_rays, max_steps)

        # Mask out-of-bounds densities
        in_bounds = (
            (ray_points[:, :, 0] >= 0)
            & (ray_points[:, :, 0] <= self.params.world_x_size)
            & (ray_points[:, :, 1] >= 0)
            & (ray_points[:, :, 1] <= self.params.world_y_size)
        )
        densities = torch.where(in_bounds, densities, torch.zeros_like(densities))

        # Integrate densities over the distance of the ray
        accumulated_density = torch.cumsum(densities * self.params.step_size, dim=1)

        # For each ray, find where the density exceeds the visibility (opacity threshold)
        is_opaque = accumulated_density >= self.params.opacity_threshold
        any_opaque = torch.any(is_opaque, dim=1)
        first_opaque_step = torch.argmax(is_opaque.to(torch.int8), dim=1)

        # Retrieve the accumulated density at the limit
        idx = torch.where(any_opaque, first_opaque_step, torch.tensor(max_steps - 1, device=device))
        ray_images = accumulated_density[torch.arange(self.params.num_rays, device=device), idx]

        # The returned positions represent the origin of the sensor readings
        sensor_position_readings = curr_pos[:2].unsqueeze(0).repeat(self.params.num_rays, 1)

        return SensorOutput(readings=ray_images, positions=sensor_position_readings)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def run_tests(cfg) -> None:
    from hydra import compose

    from src.env.simulator.smoke import BlobParams, Smoke

    smoke_params = cfg.simulator

    # Initialize Smoke
    blob_params_list = [
        BlobParams(x_pos=10, y_pos=20, intensity=1.0, spread_rate=4.0),
    ]
    smoke = Smoke(params=smoke_params, blob_params_list=blob_params_list)
    smoke.step(dt=0.1)

    # Define the sensor configurations to test
    sensor_configs = ["global", "downwards", "camera1d"]
    curr_pos = torch.tensor([10.0, 10.0, 0.0])

    for stype in sensor_configs:
        print(f"\n--- Testing Sensor Config File: {stype}.yaml ---")

        # Programmatically compose the config overriding the active sensor config
        composed_cfg = compose(config_name="config", overrides=[f"env/sensors@sensor={stype}"])
        sensor_params = composed_cfg.sensor

        if stype == "global":
            sensor_class = GlobalSensor
        elif stype == "downwards":
            sensor_class = DownwardsSensor
        elif stype == "camera1d":
            sensor_class = Camera1DSensor
        else:
            raise ValueError(f"Unknown sensor type: {stype}")

        # Instantiate and run the sensor
        sensor = sensor_class(sensor_params)
        res = sensor.read(smoke, curr_pos)

        print(f"{sensor_class.__name__} readings shape:", res.readings.shape)
        print(f"{sensor_class.__name__} positions shape:", res.positions.shape)
        if stype == "camera1d":
            print("Camera1D readings:", res.readings)


if __name__ == "__main__":
    run_tests()
