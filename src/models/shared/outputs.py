"""
Unified output classes for all models.
Clean, standardized structure.
"""
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.distributions import Normal


@dataclass
class ModelOutput:
    """Base output class for all models."""
    pass


@dataclass
class SNPOutput(ModelOutput):
    """
    Output from SNP (Stochastic Neural Process) models.
    
    Attributes:
        state: LSTM state (h, c)
        prediction: Distribution over predictions (Normal)
        z: Sampled latent variable
        prior_mu: Prior mean
        prior_sigma: Prior std
        post_mu: Posterior mean (training only)
        post_sigma: Posterior std (training only)
    """
    state: Tuple[torch.Tensor, torch.Tensor]
    prediction: Optional[Normal] = None
    z: Optional[torch.Tensor] = None
    prior_mu: Optional[torch.Tensor] = None
    prior_sigma: Optional[torch.Tensor] = None
    post_mu: Optional[torch.Tensor] = None
    post_sigma: Optional[torch.Tensor] = None


@dataclass
class RNPOutput(ModelOutput):
    """
    Output from RNP (Recurrent Neural Process) models.
    
    Attributes:
        state: LSTM state (h, c)
        prediction: Distribution over predictions (Normal)
    """
    state: Tuple[torch.Tensor, torch.Tensor]
    prediction: Optional[Normal] = None


@dataclass
class PINNOutput(ModelOutput):
    """
    Output from PINN (Physics-Informed Neural Network) models.
    
    Attributes:
        smoke_dist: Distribution over smoke values
        u: Velocity field X component
        v: Velocity field Y component
        p: Pressure field (optional)
        q: Source term (optional)
        fu: Force field X (optional)
        fv: Force field Y (optional)
        coords: Query coordinates used
    """
    smoke_dist: Normal
    u: torch.Tensor
    v: torch.Tensor
    p: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None
    fu: Optional[torch.Tensor] = None
    fv: Optional[torch.Tensor] = None
    coords: Optional[torch.Tensor] = None
