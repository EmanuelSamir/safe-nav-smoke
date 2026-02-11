"""
RNP - Recurrent Neural Process (Spatial / ConvLSTM Variant)
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
from src.models.shared.conv_lstm import ConvLSTM


@dataclass
class RNPConfig:
    """Configuration for RNP model."""
    # Dimensions
    r_dim: int = 128 # Used as channel dim for enc
    h_dim: int = 128 # Used as hidden channel dim for LSTM
    
    # use_actions: bool = False
    encoder_num_layers: int = 2
    decoder_num_layers: int = 2
    lstm_num_layers: int = 1
    min_std: float = 0.01
    
    # Fourier Features if enabled
    use_fourier_encoder: bool = False
    use_fourier_decoder: bool = True
    fourier_frequencies: int = 128
    fourier_scale: float = 20.0
    spatial_max: float = 30.0
    
    # Conv Encoder Options
    use_conv_encoder: bool = True
    grid_res: int = 64
    grid_range: Tuple[float, float] = (0, 30)

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
        Memory-efficient implementation using chunking over grid points.
        x_c: (B, N, 2) normalized spatial
        y_c: (B, N, 1) smoke values
        """
        B, N, _ = x_c.shape
        M = self.grid_points.shape[-1]
        
        # Output buffers
        # Channels: Density, Value
        grid_out = torch.zeros(B, 2, M, device=x_c.device)
        
        # Chunk size for grid points (M)
        # Adjust based on memory. 1024 is safe (approx 32x32 block)
        chunk_size = 1024 
        
        # Prepare mask if present
        mask_expanded = None
        if mask is not None:
             if mask.dim() == 2:
                  mask = mask.unsqueeze(-1)
             mask_expanded = mask.float() # (B, N, 1)

        # Iterate over grid points in chunks to save memory
        for i in range(0, M, chunk_size):
            end = min(i + chunk_size, M)
            # Slice grid points: (1, 2, chunk)
            grid_chunk = self.grid_points[..., i:end]
            
            # Distances: (B, 2, N, 1) - (1, 2, 1, chunk)
            # This creates (B, 2, N, chunk) which is much smaller than full (B, 2, N, M)
            diff = x_c.permute(0, 2, 1).unsqueeze(-1) - grid_chunk.unsqueeze(2)
            dists_sq = (diff ** 2).sum(dim=1) # (B, N, chunk)
            
            # Weights
            weights = torch.exp(-0.5 * dists_sq / (self.sigma ** 2))
            
            # Apply Mask
            if mask_expanded is not None:
                weights = weights * mask_expanded
            
            # Accumulate Density: sum(weights) -> (B, 1, chunk)
            density_chunk = weights.sum(dim=1, keepdim=True)
            
            # Accumulate Values: features @ weights -> (B, 1, chunk)
            # features: (B, 1, N)
            features = y_c.permute(0, 2, 1)
            value_chunk = torch.bmm(features, weights)
            
            # Store in output buffer
            grid_out[:, 0:1, i:end] = density_chunk
            grid_out[:, 1:2, i:end] = value_chunk

        # Normalize values by density
        density = grid_out[:, 0:1, :]
        weighted_sum = grid_out[:, 1:2, :]
        
        # Avoid division by zero
        normalized_values = weighted_sum / (density + 1e-5)
        
        out = torch.cat([density, normalized_values], dim=1)
        
        return out.reshape(B, 2, self.grid_res, self.grid_res)

class ConvDeepSet(nn.Module):
    """Simple 5-layer CNN with residual connections"""
    def __init__(self, in_channels=2, hidden=64, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AvgPool2d(2), # 64 -> 32
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AvgPool2d(2), # 32 -> 16
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AvgPool2d(2), # 16 -> 8
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
        
        # FIXED: RBFSetConv must handle [-1, 1] range because we normalize inputs to [-1, 1] in forward()
        self.set_conv = RBFSetConv(grid_res=seed_res, sigma=2.0/seed_res, grid_range=(-1, 1))
        
        # Channels: Density (1) + Smoke (1) = 2
        # Use simple striding to reduce resolution if needed, but current ConvDeepSet does pooling.
        self.processor = ConvDeepSet(in_channels=2, hidden=config.h_dim, latent_dim=config.h_dim) 
        
        # REMOVED PROJECTOR: We now return the grid
        self.resolution_reduction = 8 # 3 AvgPools of 2
        self.final_res = seed_res // self.resolution_reduction

    def forward(self, obs: Obs) -> torch.Tensor:
        """
        Encode sequence of observations.
        obs.xs: (B, T, P, 1)
        Returns: (B, T, Channels, H, W)
        """
        B, T, P, _ = obs.xs.shape
        
        # Flatten Batch and Time
        # xs: (B*T, P, 1)
        xs_flat = obs.xs.reshape(B*T, P, 1)
        ys_flat = obs.ys.reshape(B*T, P, 1)
        pts = torch.cat([xs_flat, ys_flat], dim=-1) # (BT, P, 2)
        
        vals_flat = obs.values.reshape(B*T, P, 1)
        
        mask_flat = None
        if obs.mask is not None:
            mask_flat = obs.mask.reshape(B*T, P) # (BT, P)
            
        # Normalize coordinates to [-1, 1]
        pts_norm = 2.0 * (pts / self.config.spatial_max) - 1.0 # Assuming min=0
        
        # Grid Encoding
        # grid: (BT, 2, Res, Res)
        grid = self.set_conv(pts_norm, vals_flat, mask=mask_flat)
        
        # CNN Processing
        # latent_grid: (BT, H, Res/8, Res/8)
        latent_grid = self.processor(grid)
        
        # Reshape back to (B, T, C, H, W)
        _, C, H, W = latent_grid.shape
        return latent_grid.view(B, T, C, H, W)

    def visualize_encoding(self, obs: Obs, save_path="encoder_debug.png"):
        """Visualization tool (Unchanged logic, just for debugging RBF)"""
        import matplotlib.pyplot as plt
        device = obs.xs.device
        self.eval()
        with torch.no_grad():
            b_idx = 0; t_idx = 0
            xs = obs.xs[b_idx, t_idx].cpu().squeeze().numpy()
            ys = obs.ys[b_idx, t_idx].cpu().squeeze().numpy()
            vals = obs.values[b_idx, t_idx].cpu().squeeze().numpy()
            
            xs_enc = obs.xs[b_idx:b_idx+1, t_idx:t_idx+1].reshape(1, -1, 1)
            ys_enc = obs.ys[b_idx:b_idx+1, t_idx:t_idx+1].reshape(1, -1, 1)
            vals_enc = obs.values[b_idx:b_idx+1, t_idx:t_idx+1].reshape(1, -1, 1)
            
            pts = torch.cat([xs_enc, ys_enc], dim=-1)
            pts_norm = 2.0 * (pts / self.config.spatial_max) - 1.0
            grid = self.set_conv(pts_norm, vals_enc)
            grid_np = grid.cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].scatter(xs, ys, c=vals, s=10)
            axes[1].imshow(grid_np[0, 0].T, origin='lower')
            axes[2].imshow(grid_np[0, 1].T, origin='lower')
            plt.savefig(save_path)
            plt.close()


# -----------------------------------------------

class Encoder(nn.Module):
    """(Deprecated) Vector Encoder."""
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_encoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        input_dim = self.spatial_encoder.output_dim + 1
        layer_sizes = [input_dim] + [config.h_dim] * config.encoder_num_layers
        self.net = MLP(layer_sizes, config.r_dim)
    
    def forward(self, obs: Obs) -> torch.Tensor:
        spatial_coords = torch.cat([obs.xs, obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(spatial_coords)
        inputs = torch.cat([spatial_features, obs.values], dim=-1)
        embeddings = self.net(inputs)
        if obs.mask is not None:
            mask_expanded = obs.mask.float().unsqueeze(-1)
            sum_emb = (embeddings * mask_expanded).sum(dim=-2)
            count = mask_expanded.sum(dim=-2).clamp(min=1.0)
            R = (sum_emb / count)
        else:
            R = embeddings.mean(dim=-2)
        return R


class Forecaster(nn.Module):
    """Contents ConvLSTM dynamics model."""
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        # ConvLSTM expects input_dim as Channels
        self.params = (input_dim, hidden_dim)
        self.conv_lstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
            num_layers=num_layers,
            bias=True,
            return_all_layers=False
        )
    
    def forward(self, r_input, prev_hidden, future=0):
        """
        Args:
            r_input: (B, T, C, H, W)
            prev_hidden: List of [(h, c)] tuples
        """
        # ConvLSTM forward
        last_layer_output, last_state_list = self.conv_lstm(r_input, prev_hidden)
        
        return last_layer_output, last_state_list

    def init_hidden(self, batch_size, spatial_size, device):
        return self.conv_lstm._init_hidden(batch_size, spatial_size, device)


class Decoder(nn.Module):
    """Spatial Decoder using Grid Sample/Interpolation."""
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_decoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        # Input to MLP is: Interpolated Feature (h_dim) + Query Pos Encoding
        input_dim = config.h_dim + self.spatial_encoder.output_dim
        layer_sizes = [input_dim] + [config.h_dim] * config.decoder_num_layers
        self.net = MLP(layer_sizes, 2)  # mu, logsigma
    
    def forward(self, h_grid: torch.Tensor, target_obs: Obs) -> Normal:
        """
        Args:
            h_grid: (B, T, C, H_grid, W_grid) Latent spatiotemporal grid
            target_obs: Target points (B, T, P, 1)
        """
        B, T, P, _ = target_obs.xs.shape
        _, _, C, Hg, Wg = h_grid.shape
         
        # 1. Flatten Batch * Time for efficient processing
        # h_grid: (BT, C, Hg, Wg)
        h_flat = h_grid.reshape(B*T, C, Hg, Wg)
        
        # 2. Prepare Query Coordinates
        # (B, T, P, 2)
        coords = torch.cat([target_obs.xs, target_obs.ys], dim=-1)
        # Flatten: (BT, P, 2)
        coords_flat = coords.reshape(B*T, P, 2)
        
        # Normalize coordinates to [-1, 1] for grid_sample
        # Obs are in [0, spatial_max]
        # grid_sample expects (x, y) in [-1, 1]
        grid_coords = 2.0 * (coords_flat / self.config.spatial_max) - 1.0
        
        # grid_sample expects input grid_coords as (N, H_out, W_out, 2)
        # Here we treat P as W_out, and H_out=1
        # (BT, 1, P, 2)
        grid_coords = grid_coords.unsqueeze(1)
        
        # 3. Interpolate Features
        # sampled: (BT, C, 1, P)
        sampled = F.grid_sample(h_flat, grid_coords, align_corners=True) # align_corners=True matches linspace(-1, 1) usually
        
        # Reshape to (BT, P, C)
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        
        # 4. Add Fourier Features (Positional Encoding)
        # Unflatten coords for FE (or use flat)
        spatial_features = self.spatial_encoder(coords_flat) # (BT, P, enc_dim)
        
        # Concatenate: (BT, P, C + enc_dim)
        decoder_in = torch.cat([sampled, spatial_features], dim=-1)
        
        # 5. Decode
        out = self.net(decoder_in) # (BT, P, 2)
        
        # Reshape back to (B, T, P, 2)
        out = out.reshape(B, T, P, 2)
        
        mu = out[..., 0:1] 
        logsigma = out[..., 1:2]
        sigma = F.softplus(logsigma) + self.config.min_std
        
        return Normal(mu, sigma)


class RNP(nn.Module):
    """
    Recurrent ConvCNP (Refactored).
    """
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Components
        if config.use_conv_encoder:
            self.encoder = ConvEncoder(config)
        else:
            raise ValueError("Refactored RNP requires use_conv_encoder=True")
        
        # Forecaster input channels = h_dim from encoder
        self.forecaster = Forecaster(config.h_dim, config.h_dim, config.lstm_num_layers)
        
        self.decoder = Decoder(config)
        
        self.apply(init_weights)
    
    def forward(
        self,
        state: List,
        context_obs: Obs,
        target_obs: Optional[Obs] = None
    ) -> RNPOutput:
        """
        Process a sequence of observations.
        """
        # state is now a List of (h, c) tuples for ConvLSTM
        
        # 1. Encode sequence
        # r_seq: (B, T, C, H, W)
        r_seq = self.encoder(context_obs) 
        
        # 2. Forecaster (Sequence)
        # r_next_seq: (B, T, C, H, W)
        r_next_seq, next_state = self.forecaster(r_seq, state)
        
        # 3. Decode
        prediction = None
        if target_obs is not None:
            prediction = self.decoder(r_next_seq, target_obs)
            
        return RNPOutput(state=next_state, prediction=prediction)

    def init_state(self, batch_size: int, device: torch.device) -> List:
        """Initialize ConvLSTM state."""
        # Calculate latent resolution
        res = self.config.grid_res // 8 # Based on ConvEncoder
        if res < 1: res = 1
        
        return self.forecaster.init_hidden(batch_size, (res, res), device)

    def forecast(
        self,
        context_obs: Obs,
        horizon: int,
        grid_res: Optional[int] = None
    ) -> List[Obs]:
        """
        Autoregressive forecasting for H steps.
        
        Args:
           context_obs: Initial context sequence (B, T, P, 1)
           horizon: Number of steps to forecast
           grid_res: Resolution for the feedback loop (defaults to config.grid_res)
           
        Returns:
           List of Obs, length H. Each Obs is (B, 1, P_grid, 1)
        """
        device = context_obs.xs.device
        B = context_obs.xs.shape[0]
        
        # 1. Warmup
        state = self.init_state(B, device)
        
        # We need the prediction at the last step of context to start the loop
        # We pass the full context.
        # Output will be predictions for t=1...T+1 (assuming T input steps)
        # We only care about the last prediction (T+1) to serve as input for next step.
        
        # To get the prediction, we need to query at some locations.
        # For the autoregressive loop, we need a grid of points to feed back in.
        if grid_res is None:
            grid_res = self.config.grid_res
            
        # Create grid coordinates
        # (1, Res, Res, 2)
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, self.config.spatial_max, grid_res, device=device),
            torch.linspace(0, self.config.spatial_max, grid_res, device=device),
            indexing='xy'
        )
        # (1, P_grid, 2)
        grid_pts = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        P_grid = grid_pts.shape[1]
        
        # Expand for Batch (B, 1, P_grid, 2)
        # grid_pts is (1, P_grid, 2)
        query_pts = grid_pts.unsqueeze(1).expand(B, 1, -1, -1)
        
        # Create query Obs (dummy values)
        query_obs = Obs(
            xs=query_pts[..., 0:1],
            ys=query_pts[..., 1:2],
            values=torch.zeros(B, 1, P_grid, 1, device=device),
            mask=None,
            ts=None # Time not strictly used in decoder spatial interpolation
        )
        
        # Run Context through Model
        # output.state is the updated state after context
        # But we also need the prediction for the FIRST forecast step (which is the output of the last context step)
        
        # The forward() method expects target_obs to forecast. 
        # But forward() aligns target_obs with the sequence. 
        # If we pass target_obs with T=1, it might mismatch if context is T > 1.
        
        # Strategy:
        # 1. Run encoder on context -> r_seq
        # 2. Run LSTM -> r_next_seq, last_state
        # 3. Take last timestep of r_next_seq -> r_last
        # 4. Decode r_last at query_pts -> pred_0
        
        # Encode
        r_seq = self.encoder(context_obs)
        r_next_seq, state = self.forecaster(r_seq, state)
        
        # Take last latent
        # r_next_seq: (B, T, C, H, W) -> (B, 1, C, H, W)
        r_last = r_next_seq[:, -1:, ...]
        
        # Decode first prediction
        dist = self.decoder(r_last, query_obs)
        pred_mean = dist.mean # (B, 1, P_grid, 1)
        
        predictions = []
        
        # Prepare first input for the loop
        # Input for step k is Prediction from step k-1
        current_input = Obs(
            xs=query_obs.xs,
            ys=query_obs.ys,
            values=pred_mean,
            mask=None,
            ts=None
        )
        
        predictions.append(current_input)
        
        # Loop for remaining H-1 steps
        for _ in range(horizon - 1):
            # Encode single step
            # current_input is (B, 1, P, 1)
            r_step = self.encoder(current_input)
            
            # LSTM Step
            r_next, state = self.forecaster(r_step, state)
            
            # Decode
            dist = self.decoder(r_next, query_obs)
            pred_mean = dist.mean
            
            current_input = Obs(
                xs=query_obs.xs,
                ys=query_obs.ys,
                values=pred_mean,
                mask=None,
                ts=None
            )
            predictions.append(current_input)
            
        return predictions

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    from src.models.shared.datasets import SequentialDataset, sequential_collate_fn
    from src.models.shared.observations import Obs
    
    # Simple test
    print("Initializing Refactored Spatial RNP...")
    cfg = RNPConfig(grid_res=64, h_dim=32) # Lower dim for test
    model = RNP(cfg).cuda()
    print("Model built.")
    
    # Dummy Data
    B, T, P = 2, 5, 100
    xs = torch.rand(B, T, P, 1).cuda() * 30
    ys = torch.rand(B, T, P, 1).cuda() * 30
    val = torch.rand(B, T, P, 1).cuda()
    obs = Obs(xs=xs, ys=ys, values=val, mask=None, ts=None)
    
    state = model.init_state(B, torch.device("cuda"))
    out = model(state, obs, obs)
    print("Forward Pass Successful")
    print("Prediction shape:", out.prediction.loc.shape)

    # Test Forecast
    print("\nTesting Forecast...")
    forecasts = model.forecast(obs, horizon=5, grid_res=32)
    print(f"Forecast returned {len(forecasts)} frames.")
    print(f"Frame 0 shape: {forecasts[0].values.shape}")

