import logging
from typing import Any

from src.models.conv_lstm import ConvLSTMModel
from src.models.shared.base_lightning import BasePredictorModule

log = logging.getLogger(__name__)


class ConvLSTMLightningModule(BasePredictorModule):
    """
    LightningModule specifically for ConvLSTM. 
    Inherits all training, validation, and logging logic from BasePredictorModule.
    """
    def __init__(self, t_cfg: Any, H: int, W: int, x_size: float, y_size: float):
        # Call BasePredictorModule's constructor
        super().__init__(t_cfg=t_cfg, H=H, W=W, x_size=x_size, y_size=y_size)
        
        # Initialize the specific model
        self.model = ConvLSTMModel(t_cfg.model)

    def forward(self, frames, times=None):
        return self.model(frames, times)
