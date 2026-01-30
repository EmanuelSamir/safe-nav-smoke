from abc import ABC, abstractmethod
from collections import deque
import warnings 
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for an online forecasting model that tracks data during inference.
    """
    def __init__(self, history_size: int = None):
        """
        Initializes the model with a rolling buffer for tracking past data.
        
        Args:
            x_bounds: Tuple of two numpy arrays representing the bounds of the input space.
            history_size: Number of past data points to store.
        """
        self.history_size = history_size
        self.input_history = deque(maxlen=history_size) if history_size else deque()  # Stores recent inputs
        self.output_history = deque(maxlen=history_size) if history_size else deque()  # Stores recent predictions

    @abstractmethod
    def update(self):
        """
        Update the model with a data point tracks.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Make a forecast for the next time step.

        Args:
            x: The current input features.

        Returns:
            The predicted value.
        """
        pass

    def track_data(self, x, y_true):
        """
        x: n_x array or mxn_x array
        y_true: n_y array or mxn_y array
        n_x: number of input features
        n_y: number of output features
        m: number of data points
        Stores the latest input and prediction in the rolling buffer.

        Args:
            x: The input features.
            y_true: The model's prediction.
        """

        if x.ndim == 2 and y_true.ndim == 2:
            assert x.shape[0] == y_true.shape[0], "Input and output must have the same number of data points"
            for i in range(x.shape[0]):
                self.input_history.append(x[i])
                self.output_history.append(y_true[i])
        elif x.ndim == 1 and y_true.ndim == 1:
            # assert x.shape[0] == y_true.shape[0], "Input and output must have the same number of data points"
            self.input_history.append(x)
            self.output_history.append(y_true)
        else:
            raise ValueError("Input and output must be a 1D or 2D array and have the same number of data points")

    


    def score(self, x, y_true):
        """
        Score the model on the given data.
        """
        pass