import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SoftplusSigma(nn.Module):
    """Helper to convert raw network output to a valid standard deviation."""
    def __init__(self, min_std: float = 0.1, scale: float = 0.9):
        super().__init__()
        self.min_std = min_std
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.softplus(x) + self.min_std

class AttentionAggregator(nn.Module):
    """
    Aggregates point-wise features using Multi-Head Attention.
    Uses the mean of input features as the query.
    """
    def __init__(self, r_dim: int, h_dim: int, num_heads: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(r_dim, h_dim)
        self.key_proj   = nn.Linear(r_dim, h_dim)
        self.value_proj = nn.Linear(r_dim, h_dim)
        self.attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(h_dim, r_dim)

    def forward(self, r_i: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # r_i: (B, N, r_dim)
        # mask: (B, N) - True where PADDING is.
        
        if mask is not None:
            # Compute a masked mean to serve as initial Query
            valid_mask = (~mask).float().unsqueeze(-1)
            q = (r_i * valid_mask).sum(dim=1, keepdim=True) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            q = r_i.mean(dim=1, keepdim=True)

        Q = self.query_proj(q)
        K = self.key_proj(r_i)
        V = self.value_proj(r_i)

        # MHA output: (B, 1, h_dim)
        attn_out, _ = self.attn(Q, K, V, key_padding_mask=mask)
        return self.out_proj(attn_out.squeeze(1))

class ResidualBlock(nn.Module):
    """Standard Residual Block for MLPs."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        return x + self.net(x)
