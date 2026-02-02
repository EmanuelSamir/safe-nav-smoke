import torch
from dataclasses import dataclass
from typing import Optional
from torch.distributions import Normal

@dataclass
class ObsPINN:
    """
    Standard observation bundle for PINN/Model-Based models.
    Contains spatial coordinates (xs, ys), time (ts), and measured values (values).
    """
    xs: torch.Tensor
    ys: torch.Tensor
    ts: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None # Added mask for compatibility

    def to(self, device: torch.device):
        return ObsPINN(
            xs=self.xs.to(device),
            ys=self.ys.to(device),
            ts=self.ts.to(device) if self.ts is not None else None,
            values=self.values.to(device) if self.values is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None
        )

@dataclass
class PINNOutput:
    """Complete model output for evaluation and physics."""
    smoke_dist: Normal          # Probability distribution for s
    u: torch.Tensor             # Velocity Field X
    v: torch.Tensor             # Velocity Field Y
    coords: Optional[torch.Tensor] = None # Coordinates [x, y, t] used
    fu: Optional[torch.Tensor] = None # Force Field X
    fv: Optional[torch.Tensor] = None # Force Field Y
