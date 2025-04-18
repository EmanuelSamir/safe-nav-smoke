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
        Stores the latest input and prediction in the rolling buffer.

        Args:
            x: The input features.
            y_true: The model's prediction.
        """
        self.input_history.append(x)
        self.output_history.append(y_true)


    def score(self, x, y_true):
        """
        Score the model on the given data.
        """
        pass