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
        self.register_buffer('B', torch.randn(input_dim, num_frequencies) * frequency_scale)
        
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


if __name__ == "__main__":
    # Test FourierFeatures
    import numpy as np
    fourier_features = FourierFeatures(input_dim=2, num_frequencies=10, frequency_scale=20.0, input_max=10.0)
    x = 10*torch.tensor(np.stack([np.linspace(0, 1, 10), np.linspace(0, 1, 10)], axis=1), dtype=torch.float32)
    features = fourier_features(x)
    print(f"Input shape: {x}")
    print(f"Output shape: {features.shape}")
    print(f"Output: {features}")

    # Test ConditionalFourierFeatures
    conditional_fourier_features = ConditionalFourierFeatures(input_dim=2, use_fourier=True, num_frequencies=128, frequency_scale=20.0, input_max=100.0)
    features = conditional_fourier_features(x)
    print(f"Conditional output shape: {features.shape}")
    print(f"Conditional output: {features}")