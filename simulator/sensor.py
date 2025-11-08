from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class DownwardsSensorParams:
    world_x_size: float = 20.0
    world_y_size: float = 20.0
    fov_size_degrees: float = 50
    points_in_range: int = 8
    height: float = 9.0

class DownwardsSensor:
    def __init__(self, params: DownwardsSensorParams):
        self.params = params
        self.fov_size_rad = np.deg2rad(self.params.fov_size_degrees)
        self.grid_pairs_positions = self.build_reading_grids()

    def projection_bounds(self, pos_x: float, pos_y: float) -> np.ndarray:
        """
        Returns the bounds of the projection of the sensor in the world
        """
        half_fov_size = self.fov_size_rad / 2
        projection_distance = np.tan(half_fov_size) * self.params.height
        return np.array([(pos_x - projection_distance, pos_y - projection_distance),
                         (pos_x + projection_distance, pos_y - projection_distance),
                         (pos_x + projection_distance, pos_y + projection_distance),
                         (pos_x - projection_distance, pos_y + projection_distance)])

    def build_reading_grids(self) -> np.ndarray:
        """
        Returns a grid of the reading of the sensor in the world from zero position, so
        it is not calculated at each time step.
        """
        half_fov_size = self.fov_size_rad / 2

        angle_diff = np.linspace(-half_fov_size, half_fov_size, self.params.points_in_range)

        angle_x_diff, angle_y_diff = np.meshgrid(angle_diff, angle_diff)
        x_diff = np.tan(angle_x_diff) * self.params.height
        y_diff = np.tan(angle_y_diff) * self.params.height

        pairs_positions = np.column_stack([x_diff.ravel(), y_diff.ravel()]) 
        
        return pairs_positions

    def read(self, function_to_get_values: Callable, curr_pos: np.ndarray) -> np.ndarray:
        assert curr_pos.shape[0] == 2 and curr_pos.ndim == 1, "Current position must be a 2, array"
        sensor_position_readings = curr_pos + self.grid_pairs_positions

        # Filter readings out of bounds
        sensor_position_readings = sensor_position_readings[np.logical_and(np.logical_and(
            sensor_position_readings[:, 0] >= 0, sensor_position_readings[:, 0] <= self.params.world_x_size),
            np.logical_and(
                sensor_position_readings[:, 1] >= 0, sensor_position_readings[:, 1] <= self.params.world_y_size)
            )]

        try:
            sensor_readings = function_to_get_values(sensor_position_readings)
        except:
            raise ValueError("Function to get values must be a function that takes a 2D array and returns a 1D array")

        return {"sensor_readings": sensor_readings,
                "sensor_position_readings": sensor_position_readings}
