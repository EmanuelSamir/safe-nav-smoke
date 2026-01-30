"""
Fourier Feature Encoding for high-frequency learning.
Based on "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
"""
import torch
import torch.nn as nn
from typing import Optional


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding to overcome spectral bias.
    Maps input coordinates to high-frequency features using random Fourier features.
    
    γ(x) = [sin(2π B x), cos(2π B x)]
    
    where B is a random matrix sampled from N(0, σ²I)
    """
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 128,
        frequency_scale: float = 20.0,
        input_max: float = 100.0
    ):
        """
        Args:
            input_dim: Input dimension (e.g., 1 for t, 2 for (x,y), 3 for (x,y,t))
            num_frequencies: Number of random frequencies to sample
            frequency_scale: Scale of frequencies (σ in the paper)
            input_max: Maximum expected input value for normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.input_max = input_max
        
        # Random frequency matrix B ~ N(0, σ²I)
        # Registered as buffer (not trainable, but part of model state)
        B = torch.randn(input_dim, num_frequencies) * frequency_scale
        self.register_buffer('B', B)
        
        # Output dimension: 2 * num_frequencies (sin + cos)
        self.output_dim = num_frequencies * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinates (..., input_dim)
        
        Returns:
            Fourier features (..., 2 * num_frequencies)
        """
        # Normalize to [0, 1]
        x_norm = x / self.input_max
        
        # Project: v = x @ B
        v = torch.matmul(x_norm, self.B)  # (..., num_frequencies)
        
        # Apply sin and cos
        features = torch.cat([torch.sin(2 * torch.pi * v), 
                             torch.cos(2 * torch.pi * v)], dim=-1)
        
        return features


class ConditionalFourierFeatures(nn.Module):
    """
    Wrapper that optionally applies Fourier features based on config.
    If disabled, returns identity (optionally with a linear projection).
    """
    def __init__(
        self,
        input_dim: int,
        use_fourier: bool = False,
        num_frequencies: int = 128,
        frequency_scale: float = 20.0,
        input_max: float = 100.0,
        project_identity: bool = False
    ):
        """
        Args:
            input_dim: Input dimension
            use_fourier: Whether to use Fourier features
            num_frequencies: Number of frequencies (if use_fourier=True)
            frequency_scale: Frequency scale (if use_fourier=True)
            input_max: Input normalization max (if use_fourier=True)
            project_identity: If False and use_fourier=False, use linear projection
        """
        super().__init__()
        
        self.use_fourier = use_fourier
        self.input_dim = input_dim
        
        if use_fourier:
            self.encoder = FourierFeatures(
                input_dim=input_dim,
                num_frequencies=num_frequencies,
                frequency_scale=frequency_scale,
                input_max=input_max
            )
            self.output_dim = self.encoder.output_dim
        else:
            if project_identity:
                # Simple linear projection to match expected dimensions
                self.encoder = nn.Linear(input_dim, num_frequencies * 2)
                self.output_dim = num_frequencies * 2
            else:
                self.encoder = nn.Identity()
                self.output_dim = input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def create_coordinate_encoder(
    spatial_dim: int = 2,  # (x, y)
    temporal_dim: int = 0,  # t (0 if not used)
    use_fourier_spatial: bool = False,
    use_fourier_temporal: bool = False,
    num_frequencies: int = 128,
    frequency_scale: float = 20.0,
    spatial_max: float = 100.0,
    temporal_max: float = 10.0
) -> tuple[Optional[ConditionalFourierFeatures], Optional[ConditionalFourierFeatures], int]:
    """
    Factory function to create coordinate encoders.
    
    Returns:
        (spatial_encoder, temporal_encoder, total_output_dim)
    """
    spatial_encoder = None
    temporal_encoder = None
    total_dim = 0
    
    if spatial_dim > 0:
        spatial_encoder = ConditionalFourierFeatures(
            input_dim=spatial_dim,
            use_fourier=use_fourier_spatial,
            num_frequencies=num_frequencies,
            frequency_scale=frequency_scale,
            input_max=spatial_max
        )
        total_dim += spatial_encoder.output_dim
    
    if temporal_dim > 0:
        temporal_encoder = ConditionalFourierFeatures(
            input_dim=temporal_dim,
            use_fourier=use_fourier_temporal,
            num_frequencies=num_frequencies,
            frequency_scale=frequency_scale,
            input_max=temporal_max
        )
        total_dim += temporal_encoder.output_dim
    
    return spatial_encoder, temporal_encoder, total_dim
