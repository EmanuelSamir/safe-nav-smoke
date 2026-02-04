"""
Unified output classes for all models.
Clean, standardized structure.
"""
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from torch.distributions import Normal


@dataclass
class ModelOutput:
    """Base output class for all models."""
    pass


@dataclass
class RNPOutput(ModelOutput):
    """
    Output from RNP (Recurrent Neural Process) models.
    
    Attributes:
        state: LSTM state (h, c)
        prediction: Distribution over predictions (Normal)
    """
    state: Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
    prediction: Optional[Normal] = None


@dataclass
class PINNOutput(ModelOutput):
    """
    Output from PINN (Physics-Informed Neural Network) models.
    
    Attributes:
        smoke_dist: Distribution over smoke values
        u: Velocity field X component
        v: Velocity field Y component
        fu: Force field X (optional)
        fv: Force field Y (optional)
        coords: Query coordinates used
    """
    smoke_dist: Normal
    u: torch.Tensor
    v: torch.Tensor
    fu: Optional[torch.Tensor] = None
    fv: Optional[torch.Tensor] = None
    coords: Optional[torch.Tensor] = None
