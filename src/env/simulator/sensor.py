from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SensorOutput:
    readings: np.ndarray
    positions: np.ndarray


@dataclass
class BaseSensorParams:
    world_x_size: float = 20.0
    world_y_size: float = 20.0
    sensor_type: str = "base"


@dataclass
class DownwardsSensorParams(BaseSensorParams):
    fov_size_degrees: float = 15
    points_in_range: int = 8
    height: float = 12.0
    sensor_type: str = "downwards"


@dataclass
class PointSensorParams(BaseSensorParams):
    sensor_type: str = "point"


@dataclass
class GlobalSensorParams(BaseSensorParams):
    density_reading_per_unit_length: float = (
        5.0  # density reading per unit length in the range of the sensor
    )
    sensor_type: str = "global"


@dataclass
class Camera1DSensorParams(BaseSensorParams):
    fov_size_degrees: float = 90.0
    num_rays: int = 64
    step_size: float = 0.5
    opacity_threshold: float = 2.0
    max_range: float = 8.0
    sensor_type: str = "camera_1d"


class BaseSensor:
    def __init__(self, params: BaseSensorParams):
        self.params = params

    def projection_bounds(self, pos_x: float, pos_y: float) -> np.ndarray:
        raise NotImplementedError("Projection bounds must be implemented in the subclass")

    def read(self, function_to_get_values: Callable, curr_pos: np.ndarray) -> SensorOutput:
        raise NotImplementedError("Read must be implemented in the subclass")


class PointSensor(BaseSensor):
    def __init__(self, params: PointSensorParams):
        self.params = params

    def projection_bounds(self, pos_x: float, pos_y: float) -> np.ndarray:
        arr = np.array([(pos_x, pos_y)])
        return np.repeat(arr, 4, axis=0)

    def read(self, function_to_get_values: Callable, curr_pos: np.ndarray) -> SensorOutput:
        assert curr_pos.shape[0] >= 2 and curr_pos.ndim == 1, (
            "Current position must be at least a 2-element array"
        )
        sensor_position_readings = curr_pos[:2]
        try:
            sensor_readings = function_to_get_values(sensor_position_readings)
        except Exception as e:
            raise ValueError(
                "function_to_get_values failed. It must take a 2D array and return a 1D array"
            ) from e
        return SensorOutput(readings=sensor_readings, positions=sensor_position_readings)


class DownwardsSensor(BaseSensor):
    def __init__(self, params: DownwardsSensorParams):
        self.params = params
        self.fov_size_rad = np.deg2rad(self.params.fov_size_degrees)
        self.grid_pairs_positions = self.build_reading_grids()

    def projection_bounds(self, pos_x: float, pos_y: float) -> np.ndarray:
        half_fov_size = self.fov_size_rad / 2
        projection_distance = np.tan(half_fov_size) * self.params.height
        return np.array(
            [
                (pos_x - projection_distance, pos_y - projection_distance),
                (pos_x + projection_distance, pos_y - projection_distance),
                (pos_x + projection_distance, pos_y + projection_distance),
                (pos_x - projection_distance, pos_y + projection_distance),
            ]
        )

    def build_reading_grids(self) -> np.ndarray:
        """Returns a grid of the reading of the sensor in the world from zero position.

        Returns:
            np.ndarray: A grid of the reading of the sensor in the world from zero position.
        """
        half_fov_size = self.fov_size_rad / 2

        angle_diff = np.linspace(-half_fov_size, half_fov_size, self.params.points_in_range)

        angle_x_diff, angle_y_diff = np.meshgrid(angle_diff, angle_diff)
        x_diff = np.tan(angle_x_diff) * self.params.height
        y_diff = np.tan(angle_y_diff) * self.params.height

        pairs_positions = np.column_stack([x_diff.ravel(), y_diff.ravel()])

        return pairs_positions

    def read(self, function_to_get_values: Callable, curr_pos: np.ndarray) -> SensorOutput:
        assert curr_pos.shape[0] >= 2 and curr_pos.ndim == 1, (
            "Current position must be at least a 2-element array"
        )
        sensor_position_readings = curr_pos[:2] + self.grid_pairs_positions

        # Filter readings out of bounds
        sensor_position_readings = sensor_position_readings[
            np.logical_and(
                np.logical_and(
                    sensor_position_readings[:, 0] >= 0,
                    sensor_position_readings[:, 0] <= self.params.world_x_size,
                ),
                np.logical_and(
                    sensor_position_readings[:, 1] >= 0,
                    sensor_position_readings[:, 1] <= self.params.world_y_size,
                ),
            )
        ]

        try:
            sensor_readings = function_to_get_values(sensor_position_readings)
        except Exception as e:
            raise ValueError(
                "function_to_get_values failed. It must take a 2D array and return a 1D array"
            ) from e

        return SensorOutput(readings=sensor_readings, positions=sensor_position_readings)


class GlobalSensor(BaseSensor):
    def __init__(self, params: GlobalSensorParams):
        self.params = params
        self.grid_pairs_positions = self.build_reading_grids()

    def build_reading_grids(self) -> np.ndarray:
        """Returns a grid of the reading of the sensor in the world from zero position, so
        it is not calculated at each time step.
        """
        nx = max(1, round(self.params.density_reading_per_unit_length * self.params.world_x_size))
        ny = max(1, round(self.params.density_reading_per_unit_length * self.params.world_y_size))

        dx = self.params.world_x_size / nx
        dy = self.params.world_y_size / ny

        x_range = (np.arange(nx) + 0.5) * dx
        y_range = (np.arange(ny) + 0.5) * dy

        x_grid, y_grid = np.meshgrid(x_range, y_range)
        return np.column_stack([x_grid.ravel(), y_grid.ravel()])

    def projection_bounds(self, pos_x: float, pos_y: float) -> np.ndarray:
        """Returns the bounds of the projection of the sensor in the world"""
        return np.array(
            [
                (0, 0),
                (self.params.world_x_size, 0),
                (self.params.world_x_size, self.params.world_y_size),
                (0, self.params.world_y_size),
            ]
        )

    def read(self, function_to_get_values: Callable, curr_pos: np.ndarray) -> SensorOutput:
        assert curr_pos.shape[0] >= 2 and curr_pos.ndim == 1, (
            "Current position must be at least a 2-element array"
        )

        try:
            sensor_readings = function_to_get_values(self.grid_pairs_positions)
        except Exception as e:
            raise ValueError(
                "function_to_get_values failed. It must take a 2D array and return a 1D array"
            ) from e

        return SensorOutput(readings=sensor_readings, positions=self.grid_pairs_positions)


class Camera1DSensor(BaseSensor):
    def __init__(self, params: Camera1DSensorParams):
        self.params = params
        self.fov_size_rad = np.deg2rad(self.params.fov_size_degrees)

    def projection_bounds(self, pos_x: float, pos_y: float) -> np.ndarray:
        return np.array(
            [
                (pos_x - self.params.max_range, pos_y - self.params.max_range),
                (pos_x + self.params.max_range, pos_y - self.params.max_range),
                (pos_x + self.params.max_range, pos_y + self.params.max_range),
                (pos_x - self.params.max_range, pos_y + self.params.max_range),
            ]
        )

    def read(self, function_to_get_values: Callable, curr_pos: np.ndarray) -> SensorOutput:
        """Emulates a 1D camera by raymarching efficiently across the map."""
        assert curr_pos.shape[0] >= 3 and curr_pos.ndim == 1, (
            "Current position must include [x, y, theta]"
        )
        pos_x, pos_y, theta = curr_pos[0], curr_pos[1], curr_pos[2]

        # Calculate the angle of each ray within the field of view
        angles = np.linspace(
            theta - self.fov_size_rad / 2, theta + self.fov_size_rad / 2, self.params.num_rays
        )
        ray_dirs = np.column_stack((np.cos(angles), np.sin(angles)))

        # Pre-compute all distances to sample along each ray
        max_steps = int(self.params.max_range / self.params.step_size)
        ray_distances = np.arange(1, max_steps + 1) * self.params.step_size

        # Expand arrays to generate a shape of (num_rays, max_steps, 2) covering all points
        ray_offsets = ray_dirs[:, np.newaxis, :] * ray_distances[np.newaxis, :, np.newaxis]
        ray_points = curr_pos[:2] + ray_offsets

        # Batch query all generated point densities from the active simulation
        flat_ray_points = ray_points.reshape(-1, 2)
        try:
            densities = function_to_get_values(flat_ray_points)
        except Exception as e:
            raise ValueError("function_to_get_values failed. Check inputs.") from e

        densities = densities.reshape(self.params.num_rays, max_steps)

        # Ignore densities if the ray travels outside the simulator map limits
        in_bounds = (
            (ray_points[:, :, 0] >= 0)
            & (ray_points[:, :, 0] <= self.params.world_x_size)
            & (ray_points[:, :, 1] >= 0)
            & (ray_points[:, :, 1] <= self.params.world_y_size)
        )
        densities[~in_bounds] = 0.0

        # Numerically integrate the densities over the distance of the ray
        accumulated_density = np.cumsum(densities * self.params.step_size, axis=1)

        # For each ray, find where the density exceeds the visibility (opacity threshold)
        ray_images = np.zeros(self.params.num_rays)
        is_opaque = accumulated_density >= self.params.opacity_threshold

        for i in range(self.params.num_rays):
            limit_reached = np.where(is_opaque[i])[0]
            if len(limit_reached) > 0:
                first_opaque_step = limit_reached[0]
                ray_images[i] = accumulated_density[i, first_opaque_step]
            else:
                ray_images[i] = accumulated_density[i, -1]

        # The returned positions represent the origin of the sensor readings
        sensor_position_readings = np.repeat(
            np.array([[pos_x, pos_y]]), self.params.num_rays, axis=0
        )
        return SensorOutput(readings=ray_images, positions=sensor_position_readings)
