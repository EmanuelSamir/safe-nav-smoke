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
from src.models.shared.layers import MLP, init_weights, FourierFeatureEncoder
from src.models.shared.observations import Obs
from src.models.shared.conv_lstm import ConvLSTMCell


@dataclass
class RNPConfig:
    """Configuration for RNP model."""
    # Dimensions
    h_dim: int = 128 # Used as hidden channel dim for LSTM
    
    decoder_num_layers: int = 2
    lstm_num_layers: int = 1
    min_std: float = 0.01
    
    # Fourier Features if enabled
    decoder_fourier_size: int = 128
    decoder_fourier_scale: float = 20.0
    
    # Conv Encoder Options
    grid_res: int = 64
    grid_range: Tuple[float, float] = (0, 30)

    spatial_max: float = 1.0
    spatial_min: float = 0.0

# --- Copied from pinn_conv_cnp.py (Adapted) ---

class RBFSetConv(nn.Module):
    """
    RBF Discretization using Gaussian kernels.
    Maps (x, y) continuous points to a fixed grid.
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
        B, N, _ = x_c.shape
        M = self.grid_points.shape[-1]
        
        # Output buffers
        # Channels: Density, Value
        grid_out = torch.zeros(B, 2, M, device=x_c.device)
        
        # Chunk size for grid points (M)
        # Adjust based on memory. 1024 is safe (approx 32x32 block)
        chunk_size = 1024 
        
        mask_expanded = None
        if mask is not None:
             if mask.dim() == 2:
                  mask = mask.unsqueeze(-1)
             mask_expanded = mask.float() # (B, N, 1)

        # Iterate over grid points in chunks
        for i in range(0, M, chunk_size):
            end = min(i + chunk_size, M)
            # Slice grid points: (1, 2, chunk)
            grid_chunk = self.grid_points[..., i:end]
            
            # Distances: (B, 2, N, 1) - (1, 2, 1, chunk)
            diff = x_c.permute(0, 2, 1).unsqueeze(-1) - grid_chunk.unsqueeze(2)
            dists_sq = (diff ** 2).sum(dim=1) # (B, N, chunk)
            
            weights = torch.exp(-0.5 * dists_sq / (self.sigma ** 2))
            
            # Apply Mask
            if mask_expanded is not None:
                weights = weights * mask_expanded
            
            # Accumulate Density: sum(weights) -> (B, 1, chunk)
            density_chunk = weights.sum(dim=1, keepdim=True)
            
            # Accumulate Values: features @ weights -> (B, 1, chunk)
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
        
        self.set_conv = RBFSetConv(grid_res=seed_res, sigma=2.0/seed_res, grid_range=(-1, 1))
        
        # Channels: Density (1) + Smoke (1)
        self.processor = ConvDeepSet(in_channels=2, hidden=config.h_dim, latent_dim=config.h_dim) 
        
        self.resolution_reduction = 2**3 # 3 AvgPools of 2
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
        pts_norm = 2.0 * (pts - self.config.spatial_min) / (self.config.spatial_max - self.config.spatial_min) - 1.0
        
        # Grid Encoding
        grid = self.set_conv(pts_norm, vals_flat, mask=mask_flat)
        
        # CNN Processing
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
            pts_norm = 2.0 * (pts - self.config.spatial_min) / (self.config.spatial_max - self.config.spatial_min) - 1.0
            grid = self.set_conv(pts_norm, vals_enc)
            grid_np = grid.cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].scatter(xs, ys, c=vals, s=10)
            axes[1].imshow(grid_np[0, 0].T, origin='lower')
            axes[2].imshow(grid_np[0, 1].T, origin='lower')
            plt.savefig(save_path)
            plt.close()


class Forecaster(nn.Module):
    """Contents ConvLSTM dynamics model (Single Step)."""
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim,
                kernel_size=3,
                bias=True
            ))
    
    def forward(self, r_input, prev_state):
        """
        Args:
            r_input: (B, C, H, W) - Single step input
            prev_state: List of (h, c) tuples for each layer
        Returns:
            last_layer_output: (B, C, H, W)
            next_state: List of (h, c) tuples
        """
        next_state = []
        cur_layer_input = r_input
        
        for i, cell in enumerate(self.cells):
            h, c = prev_state[i]
            h_next, c_next = cell(cur_layer_input, (h, c))
            
            next_state.append((h_next, c_next))
            cur_layer_input = h_next # Output of this layer is input to next
            
        return cur_layer_input, next_state

    def init_hidden(self, batch_size, spatial_size, device):
        init_states = []
        for cell in self.cells:
            init_states.append(cell.init_hidden(batch_size, spatial_size, device))
        return init_states


class Decoder(nn.Module):
    """Spatial Decoder using Grid Sample/Interpolation."""
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        self.spatial_encoder = FourierFeatureEncoder(
            input_dim=2,
            mapping_size=config.decoder_fourier_size,
            scale=config.decoder_fourier_scale,
        )
        
        # Input to MLP is: Interpolated Feature (h_dim) + Query Pos Encoding
        input_dim = config.h_dim + self.spatial_encoder.output_dim
        layer_sizes = [input_dim] + [config.h_dim] * config.decoder_num_layers

        self.net = MLP(layer_sizes, 2)  # mu, logsigma
    
    def forward(self, h_grid: torch.Tensor, target_obs: Obs) -> Normal:
        """
        Args:
            h_grid: (B, T, C, H_grid, W_grid) Latent spatiotemporal grid
            target_obs: Target points (B, T, P, 2)
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
        grid_coords = 2.0 * (coords_flat - self.config.spatial_min) / (self.config.spatial_max - self.config.spatial_min) - 1.0
        
        # grid_sample expects input grid_coords as (N, H_out, W_out, 2)
        # Here we treat P as W_out, and H_out=1
        # (BT, 1, P, 2)
        grid_coords = grid_coords.unsqueeze(1)
        
        # 3. Interpolate Features
        # sampled: (BT, C, 1, P). Grid coords must be in [-1, 1]
        sampled = F.grid_sample(h_flat, grid_coords, align_corners=True)

        # Reshape to (BT, P, C)
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        
        # 4. Add Fourier Features (Positional Encoding).
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
        out_dist = Normal(mu, sigma)
        
        return out_dist


class RNP(nn.Module):
    """
    Recurrent ConvCNP (Refactored).
    """
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = ConvEncoder(config)
        self.forecaster = Forecaster(config.h_dim, config.h_dim, config.lstm_num_layers)
        self.decoder = Decoder(config)
        
        self.apply(init_weights)

    def init_state(self, batch_size: int, device: torch.device) -> List:
        """Initialize ConvLSTM state."""
        # Calculate latent resolution
        res = self.config.grid_res // 8 # Based on ConvEncoder
        if res < 1: res = 1
        
        return self.forecaster.init_hidden(batch_size, (res, res), device)
    
    def forward(
        self,
        state: List,
        context_obs: Obs,
        target_obs: Optional[Obs] = None
    ) -> RNPOutput:
        """
        Process a SINGLE step of observations.
        state is a list of (h, c) tuples for ConvLSTM
        
        Args:
            state: List of (h, c) tuples
            context_obs: (B, 1, P, 1) or (B, P, 1) - Single step context
            target_obs: (B, 1, P, 1) or (B, P, 1) - Single step target
        """
        
        # 1. Encode single step
        # r_step: (B, 1, C, H, W)
        r_step = self.encoder(context_obs) 
        
        # Remove time dimension for Forecaster: (B, C, H, W)
        r_step_sq = r_step.squeeze(1) 
        
        # 2. Forecaster (Single Step)
        # r_next: (B, C, H, W)
        r_next, next_state = self.forecaster(r_step_sq, state)
        
        # Add back time dimension for Decoder: (B, 1, C, H, W)
        r_next_expanded = r_next.unsqueeze(1)
        
        # 3. Decode
        prediction = None
        if target_obs is not None:
            prediction = self.decoder(r_next_expanded, target_obs)
            
        return RNPOutput(state=next_state, prediction=[prediction])

    def forecast(
        self,
        context_obs: Obs,
        n_horizons: int,
    ) -> List[Obs]:
        """
        Autoregressive forecasting for H steps for whole map.
        
        Args:
           context_obs: Initial context sequence (B, T, P, 1)
           n_horizons: Number of steps for horizon to forecast
        Returns:
           List of Obs, length H. Each Obs is (B, 1, P_grid, 1)
        """
        device = context_obs.xs.device
        B = context_obs.xs.shape[0]
        
        # 1. Warmup
        state = self.init_state(B, device)
            
        # Create grid coordinates
        # (1, Res, Res, 2)
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(self.config.spatial_min, self.config.spatial_max, self.config.grid_res, device=device),
            torch.linspace(self.config.spatial_min, self.config.spatial_max, self.config.grid_res, device=device),
            indexing='xy'
        )

        # (1, P_grid, 2)
        grid_pts = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        P_grid = grid_pts.shape[1]
        
        # Expand for Batch (B, 1, P_grid, 2)
        query_pts = grid_pts.unsqueeze(1).expand(B, 1, -1, -1)
        
        query_obs = Obs(
            xs=query_pts[..., 0:1],
            ys=query_pts[..., 1:2],
            values=torch.zeros(B, 1, P_grid, 1, device=device),
            mask=None,
            ts=None # Time not strictly used in decoder spatial interpolation
        )
        
        # Encode
        r_step = self.encoder(context_obs)
        # Remove time dim for Forecaster
        r_step_sq = r_step.squeeze(1)
        r_next, state = self.forecaster(r_step_sq, state)
        
        # Add time dim back
        r_next_expanded = r_next.unsqueeze(1)
        
        predictions = []
        
        while len(predictions) < n_horizons:
            # Decode current latent -> List[Normal] of length forecast_horizon
            dists = self.decoder(r_next_expanded, query_obs)
            
            # Convert to Obs and append
            chunk_predictions = []
            for dist in dists:
                pred_mean = dist.mean # (B, 1, P_grid, 1)
                obs_pred = Obs(
                    xs=query_obs.xs,
                    ys=query_obs.ys,
                    values=pred_mean,
                    mask=None,
                    ts=None
                )
                chunk_predictions.append(obs_pred)
            
            # Add to total predictions
            predictions.extend(chunk_predictions)
            
            if len(predictions) >= n_horizons:
                break
            
            # If we need more steps, we must advance the state using the predictions we just made.
            # We treat the predicted sequence as the input for the next steps.
            # We must iterate through the chunk to update the LSTM state step-by-step.
            for obs_pred in chunk_predictions:
                 # Encode single step
                 r_step = self.encoder(obs_pred)
                 r_step_sq = r_step.squeeze(1)
                 
                 # LSTM Step
                 r_next, state = self.forecaster(r_step_sq, state)
                 
            # After loop, r_next corresponds to the latent at the end of the chunk.
            r_next_expanded = r_next.unsqueeze(1)
            
        return predictions[:n_horizons]

if __name__ == "__main__":
    import os
    from src.models.shared.observations import slice_obs
    
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
    
    # Test Step-by-Step
    print("Testing Step-by-Step...")
    state = model.init_state(B, torch.device("cuda"))
    
    # Context: 0..T-1. Target: 1..T
    ctx_seq = slice_obs(obs, 0, -1)
    trg_seq = slice_obs(obs, 1, None)
    
    T = ctx_seq.xs.shape[1]
    
    for t in range(T):
        ctx_t = slice_obs(ctx_seq, t, t+1)
        trg_t = slice_obs(trg_seq, t, t+1)
        
        out = model(state, ctx_t, trg_t)
        state = out.state
        if out.prediction is not None:
             # prediction is now a List[Normal]
             for i, pred in enumerate(out.prediction):
                  print(f"Step {t} (Horizon {i}): Prediction Mean Shape: {pred.mean.shape}")

    # Forecast
    print("Testing Forecast...")
    # Last Context Frame
    last_ctx = slice_obs(ctx_seq, T-1, T)
    
    # Forecast Horizon 5
    preds = model.forecast(last_ctx, n_horizons=5)
    print(f"Forecast Predictions: {len(preds)}")
    print(f"Forecast 0 Shape: {preds[0].values.shape}")

