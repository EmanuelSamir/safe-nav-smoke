"""
Unified observation classes for all models.
Clean, standardized structure without backward compatibility.
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Obs:
    """
    Unified observation bundle for all models.
    
    Attributes:
        xs: X coordinates (B, N) or (N,)
        ys: Y coordinates (B, N) or (N,)
        values: Measured values (B, N) or (N,) - optional
        mask: Padding mask (B, N) or (N,) - True where VALID data
        ts: Time coordinates (B, N) or (N,) - optional, for model-based
    
    Usage:
        Model-Free: Obs(xs, ys, values, mask)
        Model-Based: Obs(xs, ys, values, mask, ts)
    """
    xs: torch.Tensor
    ys: torch.Tensor
    values: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    ts: Optional[torch.Tensor] = None  # For model-based (PINN)
    
    def to(self, device: torch.device) -> 'Obs':
        """Move all tensors to device."""
        return Obs(
            xs=self.xs.to(device),
            ys=self.ys.to(device),
            values=self.values.to(device) if self.values is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
            ts=self.ts.to(device) if self.ts is not None else None
        )
    
    @property
    def coords_2d(self) -> torch.Tensor:
        """Get (x, y) coordinates as single tensor."""
        return torch.stack([self.xs, self.ys], dim=-1)
    
    @property
    def coords_3d(self) -> torch.Tensor:
        """Get (x, y, t) coordinates as single tensor (for model-based)."""
        if self.ts is None:
            raise ValueError("Time coordinates (ts) not available")
        return torch.stack([self.xs, self.ys, self.ts], dim=-1)
