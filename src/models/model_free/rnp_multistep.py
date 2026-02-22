"""
RNP Multistep - Recurrent Neural Process (Spatial / ConvLSTM Variant) with Multi-Step Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from torch.distributions import Normal

# Model Components
from src.models.shared.outputs import RNPOutput
from src.models.shared.layers import MLP, init_weights, FourierFeatureEncoder
from src.models.shared.observations import Obs
from src.models.shared.conv_lstm import ConvLSTMCell
from src.models.model_free.rnp import RNPConfig, Forecaster, ConvEncoder

class MultiStepDecoder(nn.Module):
    """
    Spatial Decoder using Grid Sample/Interpolation.
    Outputs H distributions (means, stds) for H future steps simultaneously.
    """
    def __init__(self, config: RNPConfig, forecast_horizon: int = 5):
        super().__init__()
        self.config = config
        self.forecast_horizon = forecast_horizon
        
        self.spatial_encoder = FourierFeatureEncoder(
            input_dim=2,
            mapping_size=config.decoder_fourier_size,
            scale=config.decoder_fourier_scale,
        )
        
        # Input to MLP is: Interpolated Feature (h_dim) + Query Pos Encoding
        input_dim = config.h_dim + self.spatial_encoder.output_dim
        layer_sizes = [input_dim] + [config.h_dim] * config.decoder_num_layers

        # Output: H * 2 (mu, logsigma for each step)
        self.net = MLP(layer_sizes, forecast_horizon * 2) 
    
    def forward(self, h_grid: torch.Tensor, target_obs: Obs) -> List[Normal]:
        """
        Args:
            h_grid: (B, T, C, H_grid, W_grid) Latent spatiotemporal grid
            target_obs: Target points (B, T, P, 2)
            
        Returns:
            List of Normal distributions [Dist_t+1, Dist_t+2, ..., Dist_t+H]
            Each dist has shape (B, T, P, 1)
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
        # (BT, P, H * 2)
        out = self.net(decoder_in) 
        
        # Reshape back to (B, T, P, H, 2)
        out = out.reshape(B, T, P, self.forecast_horizon, 2)

        dists = []
        for h in range(self.forecast_horizon):
            mu = out[..., h, 0:1] # (B, T, P, 1)
            logsigma = out[..., h, 1:2]
            sigma = F.softplus(logsigma) + self.config.min_std
            dists.append(Normal(mu, sigma))
            
        return dists


class RNPMultistep(nn.Module):
    """
    RNP with Multistep forecasting capability.
    """
    def __init__(self, config: RNPConfig, forecast_horizon: int = 5):
        super().__init__()
        self.config = config
        self.forecast_horizon = forecast_horizon
        
        # Components
        self.encoder = ConvEncoder(config)
        self.forecaster = Forecaster(config.h_dim, config.h_dim, config.lstm_num_layers)
        self.decoder = MultiStepDecoder(config, forecast_horizon=forecast_horizon)
        
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
        
        Args:
            state: List of (h, c) tuples
            context_obs: (B, 1, P, 1) - Single step context at time t
            target_obs: (B, 1, P, 1) - Targets. Note that values here might be (B, 1, P, H) but we only use XS/YS for query.
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
        # Outputs List[Normal] of length H
        prediction = None
        if target_obs is not None:
            prediction = self.decoder(r_next_expanded, target_obs)
            
        return RNPOutput(state=next_state, prediction=prediction)

    def autoregressive_forecast(
        self,
        state, 
        context_obs: Obs,
        target_obs: Obs,
        horizon: int,
        num_samples: int = 1,
    ) -> List[Dict[str, Obs]]:
        """
        Autoregressive forecasting for H steps.
        
        Args:
           state: Initial state (can be None, will be initialized)
           context_obs: Context sequence (B, T, P, 1). Includes values
           target_obs: Target points (B, 1, P, 1). Used for query coordinates
           horizon: Number of steps to forecast
           num_samples: Number of samples to draw (if > 1, expands batch size)
        Returns:
           List of Dict, length horizon. Each element contains 'sample', 'mean' and 'std' Obs.
        """
        device = context_obs.xs.device
        B = context_obs.xs.shape[0]
        
        if num_samples > 1:
            context_obs = Obs(
                xs=context_obs.xs.repeat_interleave(num_samples, dim=0),
                ys=context_obs.ys.repeat_interleave(num_samples, dim=0),
                values=context_obs.values.repeat_interleave(num_samples, dim=0),
                mask=context_obs.mask.repeat_interleave(num_samples, dim=0) if context_obs.mask is not None else None,
                ts=context_obs.ts.repeat_interleave(num_samples, dim=0) if context_obs.ts is not None else None
            )
            target_obs = Obs(
                xs=target_obs.xs.repeat_interleave(num_samples, dim=0),
                ys=target_obs.ys.repeat_interleave(num_samples, dim=0),
                values=target_obs.values.repeat_interleave(num_samples, dim=0) if target_obs.values is not None else None,
                mask=target_obs.mask.repeat_interleave(num_samples, dim=0) if target_obs.mask is not None else None,
                ts=target_obs.ts.repeat_interleave(num_samples, dim=0) if target_obs.ts is not None else None
            )
            B = B * num_samples
        
        if state is None:
            state = self.init_state(B, device)

        T_ctx = context_obs.xs.shape[1]
        
        # 1. Encode Context Autoregressively
        for t in range(T_ctx - 1):
            ctx_t = slice_obs(context_obs, t, t+1)
            out = self(state, ctx_t, None)
            state = out.state
            
        current_input = slice_obs(context_obs, T_ctx-1, T_ctx) 
        
        T_trg = target_obs.xs.shape[1]
        predictions = []
        steps_generated = 0
        
        while steps_generated < horizon:
            # Prepare Query for this step
            t_idx = min(steps_generated, T_trg - 1)
            trg_step = slice_obs(target_obs, t_idx, t_idx+1)
            
            # Forward pass
            out = self(state, current_input, trg_step)
            state = out.state
            
            dists = out.prediction
            
            chunk_samples = []
            chunk_means = []
            chunk_stds = []
            for dist in dists:
                s = dist.sample()
                m = dist.mean
                std = dist.stddev
                
                chunk_samples.append(s)
                chunk_means.append(m)
                chunk_stds.append(std)
                
            for s, m, std in zip(chunk_samples, chunk_means, chunk_stds):
                obs_sample = Obs(xs=trg_step.xs, ys=trg_step.ys, values=s, mask=trg_step.mask, ts=trg_step.ts)
                obs_mean = Obs(xs=trg_step.xs, ys=trg_step.ys, values=m, mask=trg_step.mask, ts=trg_step.ts)
                obs_std = Obs(xs=trg_step.xs, ys=trg_step.ys, values=std, mask=trg_step.mask, ts=trg_step.ts)
                
                predictions.append({'sample': obs_sample, 'mean': obs_mean, 'std': obs_std})
                
            steps_generated += len(chunk_samples)
            
            if steps_generated < horizon:
                # Feedback loop for Multistep (updates state)
                for pred_vals in chunk_samples:
                     obs_in = Obs(xs=trg_step.xs, ys=trg_step.ys, values=pred_vals)
                     out = self(state, obs_in, None)
                     state = out.state
                
                current_input = Obs(xs=trg_step.xs, ys=trg_step.ys, values=chunk_samples[-1])
            
        return predictions[:horizon]
