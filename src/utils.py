import numpy as np
import warnings

def clip_world(x: float, y: float, x_max: float, y_max: float):
    """
    Clips the world coordinates (x, y) to keep them within the map.
    Note: Map starts at (0, 0) and ends at (x_max, y_max)
    """
    epsilon = 1e-6
    x_clipped = np.clip(x, 0, x_max - epsilon)
    y_clipped = np.clip(y, 0, y_max - epsilon)

    if x != x_clipped or y != y_clipped:
        warnings.warn(f"Clipping world coords: ({x}, {y}) → ({x_clipped}, {y_clipped})")
    return x_clipped, y_clipped

def clip_index(row: int, col: int, x_max: int, y_max: int, resolution: float):
    """
    Clips the indices of a matrix to keep them within the valid bounds.
    Note: Matrix starts at (0, 0) and ends at (num_rows - 1, num_cols - 1)
    """
    num_rows, num_cols = get_index_bounds(x_max, y_max, resolution)

    row_clipped = np.clip(row, 0, num_rows - 1)
    col_clipped = np.clip(col, 0, num_cols - 1)

    if row != row_clipped or col != col_clipped:
        warnings.warn(f"Clipping index: ({row}, {col}) → ({row_clipped}, {col_clipped})")

    return row_clipped, col_clipped

def world_to_index(x: float, y: float, x_max: float, y_max: float, resolution: float):
    """
    Converts world coordinates (x, y) in meters to indices (row, col) of a matrix.
    Note: Matrix starts at (0, y_max) and ends at (x_max, 0)
    """
    x, y = clip_world(x, y, x_max, y_max)

    col = int(x / resolution)
    row = int(y / resolution)

    row, col = clip_index(row, col, x_max, y_max, resolution)

    return row, col

def index_to_world(row: int, col: int, x_max: float, y_max: float, resolution: float):
    """
    Converts indices (row, col) of a matrix to world coordinates (x, y) of the center of the cell.
    Note: Matrix starts at (0, y_max) and ends at (x_max, 0)
    """
    row, col = clip_index(row, col, x_max, y_max, resolution)

    x = (col + 0.5) * resolution
    y = (row + 0.5) * resolution

    x, y = clip_world(x, y, x_max, y_max)

    return x, y

def get_index_bounds(x_max: float, y_max: float, resolution: float):
    """
    Returns the bounds of the index space in the form of a tuple (num_rows, num_cols).
    Note: Matrix starts at (0, y_max) and ends at (x_max, 0)
    """
    return int(np.ceil(y_max / resolution)), int(np.ceil(x_max / resolution))

def get_world_bounds(num_rows: int, num_cols: int, resolution: float):
    """
    Returns the bounds of the world space in the form of a tuple (x_max, y_max).
    Note: World starts at (0, 0) and ends at (x_max, y_max)
    """
    return num_cols * resolution, num_rows * resolution