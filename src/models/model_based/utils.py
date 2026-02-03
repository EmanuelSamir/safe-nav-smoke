import torch
from dataclasses import dataclass
from typing import Optional
from torch.distributions import Normal

@dataclass
class Output:
    """Complete model output for evaluation and physics."""
    smoke_dist: Normal          # Probability distribution for s
    u: torch.Tensor             # Velocity Field X
    v: torch.Tensor             # Velocity Field Y
    coords: Optional[torch.Tensor] = None # Coordinates [x, y, t] used
    fu: Optional[torch.Tensor] = None # Force Field X
    fv: Optional[torch.Tensor] = None # Force Field Y
