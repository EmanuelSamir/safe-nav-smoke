import numpy as np


class SpaceCarvingMapper:
    """Reconstructs a 2D occupancy/density map from 1D sensor scanlines.

    Uses 'Space Carving' logic where each ray's integrated density is projected
    back into a 2D wedge on a grid.
    """

    def __init__(self, world_x, world_y, resolution):
        self.res = resolution
        self.nx = int(np.ceil(world_x / resolution))
        self.ny = int(np.ceil(world_y / resolution))
        self.grid = np.zeros((self.ny, self.nx))

        # Grid point coordinates (centers of the pixels)
        x = np.arange(self.nx) * resolution + (resolution / 2.0)
        y = np.arange(self.ny) * resolution + (resolution / 2.0)
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")

        # Internal smoothing factor
        self.alpha = 0.5

    def update(self, pose, readings, sensor_params):
        """Updates the internal grid with new sensor readings.

        pose: [x, y, theta] in world coordinates
        readings: 1D array of integrated density from Camera1DSensor
        sensor_params: Parameters including FOV and max range
        """
        pos_x, pos_y, theta = pose
        fov_rad = np.deg2rad(sensor_params.fov_size_degrees)
        max_range = sensor_params.max_range
        num_rays = sensor_params.num_rays

        # Angular step for each ray
        d_theta = fov_rad / num_rays
        start_angle = theta - fov_rad / 2

        # Relative coordinates of all grid points to robot
        DX = self.X - pos_x
        DY = self.Y - pos_y
        DistSq = DX**2 + DY**2
        Angles = np.arctan2(DY, DX)

        # Mask for distance: only update points within sensor range
        dist_mask = DistSq <= max_range**2

        current_map = np.zeros_like(self.grid)
        update_mask = np.zeros(self.grid.shape, dtype=bool)

        for i in range(num_rays):
            r_val = readings[i]
            # Angle range for this ray
            angle_start = start_angle + i * d_theta
            angle_end = start_angle + (i + 1) * d_theta

            # Normalize angles to [-pi, pi] for comparison
            # We use a trick: (a - b + pi) % (2*pi) - pi gives the signed difference in range [-pi, pi]
            diff_start = (Angles - angle_start + np.pi) % (2 * np.pi) - np.pi
            diff_end = (Angles - angle_end + np.pi) % (2 * np.pi) - np.pi

            # If the sign of the difference changes, the point is within the wedge
            # Or simplified: check if Angles is between start and end
            # This handles wrap-around correctly if we are careful
            rel_angle = (Angles - start_angle + np.pi) % (2 * np.pi) - np.pi
            angle_mask = (rel_angle >= i * d_theta - d_theta / 2) & (
                rel_angle <= (i + 1) * d_theta + d_theta / 2
            )

            ray_mask = dist_mask & angle_mask
            current_map[ray_mask] = r_val
            update_mask |= ray_mask

        # Exponential moving average update only for points seen in this scan
        self.grid[update_mask] = (1 - self.alpha) * self.grid[
            update_mask
        ] + self.alpha * current_map[update_mask]

    def get_map(self):
        return self.grid

    def reset(self):
        self.grid.fill(0.0)
