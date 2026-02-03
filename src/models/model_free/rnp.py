"""
RNP - Recurrent Neural Process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from torch.distributions import Normal

# Removed Obs import as requested
from src.models.shared.outputs import RNPOutput
from src.models.shared.fourier_features import ConditionalFourierFeatures
from src.models.shared.layers import MLP, init_weights
from src.models.shared.observations import Obs


@dataclass
class RNPConfig:
    """Configuration for RNP model."""
    # Dimensions
    r_dim: int = 128
    h_dim: int = 128
    
    # use_actions: bool = False
    encoder_num_layers: int = 3
    decoder_num_layers: int = 3
    lstm_num_layers: int = 1
    min_std: float = 0.01
    
    # Fourier Features if enabled
    use_fourier_encoder: bool = False
    use_fourier_decoder: bool = False
    fourier_frequencies: int = 128
    fourier_scale: float = 20.0
    spatial_max: float = 30.0
    
    # Conv Encoder Options
    use_conv_encoder: bool = False
    grid_res: int = 32
    grid_range: Tuple[float, float] = (-1, 1)

# --- Copied from pinn_conv_cnp.py (Adapted) ---

class RBFSetConv(nn.Module):
    """
    Simpler RBF Discretization using Gaussian kernels.
    Maps (x, y) continuous points to a fixed grid.
    Adapted for RNP (no time feature in grid).
    """
    def __init__(self, grid_res=32, grid_range=(-1, 1), sigma=0.1):
        super().__init__()
        self.grid_res = grid_res
        self.grid_min, self.grid_max = grid_range
        self.sigma = sigma
        
        # Create grid coordinates on device buffer
        coords = torch.linspace(self.grid_min, self.grid_max, grid_res)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        
        # (1, 2, H*W)
        self.register_buffer('grid_points', torch.stack([grid_x, grid_y], dim=0).reshape(1, 2, -1))

    def forward(self, x_c, y_c, mask: Optional[torch.Tensor] = None):
        """
        x_c: (B, N, 2) normalized spatial
        y_c: (B, N, 1) smoke values
        """
        # Distances: (B, 2, N, 1) - (1, 2, 1, M)
        diff = x_c.permute(0, 2, 1).unsqueeze(-1) - self.grid_points.unsqueeze(2)
        dists_sq = (diff ** 2).sum(dim=1) # (B, N, M)
        
        # Weights (Gaussian kernel)
        weights = torch.exp(-0.5 * dists_sq / (self.sigma ** 2))

        # Apply mask if provided (True = Padding)
        if mask is not None:
             # mask: (B, N) -> (B, N, 1)
             if mask.dim() == 2:
                  mask = mask.unsqueeze(-1)
             mask_expanded = mask.float() # (B, N, 1) Ensure float for multiplication
             # Mask is True (1) for Valid, False (0) for Padding.
             # We want to keep Valid, zero out Padding.
             weights = weights * mask_expanded
        
        # Density (sum of weights per grid point)
        density = weights.sum(dim=1, keepdim=True) # (B, 1, M)
        
        # Features: [smoke]
        features = y_c.permute(0, 2, 1) # (B, 1, N)
        
        # Weighted sum: (B, 1, N) @ (B, N, M) -> (B, 1, M)
        weighted_sum = torch.bmm(features, weights)
        
        # Normalize by density (Safe division)
        # Channels: Density, Value
        out = torch.cat([density, weighted_sum / (density + 1e-5)], dim=1) # (B, 2, M)
        
        # Reshape to grid (B, 2, Res, Res)
        return out.reshape(out.shape[0], 2, self.grid_res, self.grid_res)

class ConvDeepSet(nn.Module):
    """Simple 5-layer CNN with residual connections"""
    def __init__(self, in_channels=2, hidden=64, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, latent_dim, 5, padding=2), 
            nn.BatchNorm2d(latent_dim), nn.Tanh() 
        )
    def forward(self, x):
        return self.net(x)

class ConvEncoder(nn.Module):
    """Convolutional Encoder for Set Data."""
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        seed_res = config.grid_res
        self.set_conv = RBFSetConv(grid_res=seed_res, sigma=2.0/seed_res, grid_range=config.grid_range) # Sigma relative to grid?
        
        # Channels: Density (1) + Smoke (1) = 2
        self.processor = ConvDeepSet(in_channels=2, hidden=config.h_dim, latent_dim=config.h_dim) 
        
        flat_dim = config.h_dim * seed_res * seed_res
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, config.r_dim),
            nn.ReLU()
        )

    def forward(self, obs: Obs) -> torch.Tensor:
        """
        Encode sequence of observations.
        obs.xs: (B, T, P, 1)
        """
        B, T, P, _ = obs.xs.shape
        
        # Flatten Batch and Time
        # xs: (B*T, P, 1)
        xs_flat = obs.xs.view(B*T, P, 1)
        ys_flat = obs.ys.view(B*T, P, 1)
        pts = torch.cat([xs_flat, ys_flat], dim=-1) # (BT, P, 2)
        
        vals_flat = obs.values.view(B*T, P, 1)
        
        mask_flat = None
        if obs.mask is not None:
            mask_flat = obs.mask.view(B*T, P) # (BT, P)
            
        # Normalize coordinates to [-1, 1]
        # x_norm = 2 * (x - min) / (max - min) - 1
        pts_norm = 2.0 * (pts / self.config.spatial_max) - 1.0 # Assuming min=0
        
        # Grid Encoding
        # grid: (BT, 2, Res, Res)
        grid = self.set_conv(pts_norm, vals_flat, mask=mask_flat)
        
        # CNN Processing
        # latent_grid: (BT, H, Res, Res)
        latent_grid = self.processor(grid)
        
        # Project to vector
        # r: (BT, r_dim)
        r = self.projector(latent_grid)
        
        # Reshape back to (B, T, r_dim)
        return r.view(B, T, -1)

# -----------------------------------------------

class Encoder(nn.Module):
    """Encodes observations into representation r."""
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Fourier encoding for spatial coordinates
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_encoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        # MLP: (fourier_features, value) -> r
        input_dim = self.spatial_encoder.output_dim + 1
        layer_sizes = [input_dim] + [config.h_dim] * config.encoder_num_layers
        self.net = MLP(layer_sizes, config.r_dim)
    
    def forward(self, obs: Obs) -> torch.Tensor:
        """Encode observation to representation."""
        # Encode coordinates
        # obs.xs: (B, T, P, 1) -> cat -> (B, T, P, 2)
        spatial_coords = torch.cat([obs.xs, obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(spatial_coords)
        
        # Concatenate with values: (x, y, value)
        values = obs.values
        # values shape: (B, T, P, input_dim)
        inputs = torch.cat([spatial_features, values], dim=-1)
        
        # Encode
        # embeddings shape: (B, T, P, r_dim)
        embeddings = self.net(inputs)
        
        # Aggregation step: mean with mask
        if obs.mask is not None:
            mask_expanded = obs.mask.float()
            if mask_expanded.dim() == embeddings.dim() - 1:
                mask_expanded = mask_expanded.unsqueeze(-1)
                
            # mask_expanded shape: (B, T, P, 1)
            # embeddings shape: (B, T, P, r_dim)
            sum_emb = (embeddings * mask_expanded).sum(dim=-2)
            count = mask_expanded.sum(dim=-2).clamp(min=1.0)
            R = (sum_emb / count)
            # R shape: (B, T, r_dim)
        else:
            # print("Warning: No mask provided for RNP encoder")
            R = embeddings.mean(dim=-2)
        
        return R


class Forecaster(nn.Module):
    """LSTM-based dynamics model."""
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, r_input, prev_hidden, future=0):
        """
        Args:
            r_input: (B, T, input_dim) or (B, input_dim)
            prev_hidden: (h, c) each (num_layers, B, hidden_dim)
            future: unused for now
        
        Returns:
            outputs: (B, T, hidden_dim)
            state: (h, c)
        """
        if r_input.dim() == 2:
            r_input = r_input.unsqueeze(1)  # (B, 1, input_dim)
            
        out, (h, c) = self.lstm(r_input, prev_hidden)
        # out is (B, T, hidden_dim)
        return out, (h, c)


class Decoder(nn.Module):
    """Decodes representation and query to predictions."""
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Fourier encoding for query coordinates
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_decoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        # MLP: (h, fourier_features) -> (mu, sigma)
        input_dim = config.h_dim + self.spatial_encoder.output_dim
        layer_sizes = [input_dim] + [config.h_dim] * config.decoder_num_layers
        self.net = MLP(layer_sizes, 2)  # mu, logsigma
    
    def forward(self, h: torch.Tensor, target_obs: Obs) -> Normal:
        """
        Args:
            h: Hidden state sequence (B, T, h_dim)
            target_obs: Target observation Obs with tensors (B, T, P, 1)
        
        Returns:
            Distribution over predictions
        """
        # Encode query coordinates
        # target_obs.xs: (B, T, P, 1) -> (B, T, P, 2)
        coords = torch.cat([target_obs.xs, target_obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(coords) # (B, T, P, enc_dim)
        
        # Expand h to match spatial features (P dimension)
        # h: (B, T, h_dim) -> (B, T, 1, h_dim) -> (B, T, P, h_dim)
        if h.dim() == 3: 
             h_exp = h.unsqueeze(-2).expand(-1, -1, coords.shape[-2], -1)
        else:
             raise ValueError(f"Expected h to be 3D (B, T, H), got {h.shape}")
        
        # Concatenate and decode
        decoder_in = torch.cat([h_exp, spatial_features], dim=-1)
        out = self.net(decoder_in) # (B, T, P, 2)
        
        mu = out[..., 0:1] # (B, T, P, 1)
        logsigma = out[..., 1:2] # (B, T, P, 1)
        sigma = F.softplus(logsigma) + self.config.min_std
        
        return Normal(mu, sigma)


class RNP(nn.Module):
    """
    Recurrent Neural Process for scalar field prediction.
    Deterministic model with LSTM dynamics.
    """
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Components
        if config.use_conv_encoder:
            self.encoder = ConvEncoder(config)
        else:
            self.encoder = Encoder(config)
        
        # Forecaster input: r + [optional: action]
        forecaster_input = config.r_dim

        self.forecaster = Forecaster(forecaster_input, config.h_dim, config.lstm_num_layers)
        
        self.decoder = Decoder(config)
        
        self.apply(init_weights)
    
    def forward(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        context_obs: Obs,
        target_obs: Optional[Obs] = None
    ) -> RNPOutput:
        """
        Process a sequence of observations.
        """
        h_prev, c_prev = state

        # 1. Encode sequence
        # context_obs contains tensors of shape (B, T, P, C)
        # Encoder returns (B, T, r_dim)
        r_seq = self.encoder(context_obs) 
        
        # 2. Forecaster (Sequence)
        # r_next_seq: (B, T, h_dim)
        r_next_seq, (h, c) = self.forecaster(r_seq, (h_prev, c_prev))
        
        # 3. Decode
        prediction = None
        if target_obs is not None:
            # Decode for all time steps provided in target_obs
            # Assumes target_obs aligns with r_next_seq in length
            prediction = self.decoder(r_next_seq, target_obs)
            
        return RNPOutput(state=(h, c), prediction=prediction)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM state."""
        h = torch.zeros(self.config.lstm_num_layers, batch_size, self.config.h_dim, device=device)
        c = torch.zeros(self.config.lstm_num_layers, batch_size, self.config.h_dim, device=device)
        return (h, c)

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    from src.models.shared.datasets import SequentialDataset, sequential_collate_fn
    from src.models.shared.observations import Obs
    
    # Path to data
    data_path = "/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/data/playback_data/global_source_400_100_2nd.npz"
    
    if not os.path.exists(data_path):
        print(f"Skipping test: Data not found at {data_path}")
    else:
        print("--- Testing RNP with SequentialDataset ---")
        
        # 1. Dataset
        seq_dataset = SequentialDataset(data_path, sequence_length=25, max_episodes=5)
        seq_loader = DataLoader(seq_dataset, batch_size=4, collate_fn=sequential_collate_fn)
        
        ctx_seq, trg_seq, idx = next(iter(seq_loader))
        
        print(f"Context Batch Shape: {ctx_seq.xs.shape}") # (B, T, P, 1)
        
        # 2. Model
        config = RNPConfig(
            r_dim=64,
            h_dim=64,
            encoder_num_layers=2,
            decoder_num_layers=2,
            lstm_num_layers=1,
            use_fourier_encoder=True,
            use_fourier_decoder=True,
            use_conv_encoder=True,
            grid_res=32,
            grid_range=(0, 30)
        )
        model = RNP(config)
        
        # 3. Forward
        # Init state
        state = model.init_state(batch_size=4, device=ctx_seq.xs.device)
        
        output = model(state, ctx_seq, trg_seq)
        
        print("\n--- Model Output ---")
        if output.prediction is not None:
             print(f"Prediction Mean Shape: {output.prediction.loc.shape}")
             print(f"Prediction Std Shape: {output.prediction.scale.shape}")
             # Check match
             assert output.prediction.loc.shape == trg_seq.xs.shape
        
        print(f"Next State (h) Shape: {output.state[0].shape}")
        
        print("Test Passed!")
