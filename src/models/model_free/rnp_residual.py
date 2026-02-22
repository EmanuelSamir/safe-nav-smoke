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
from src.models.model_free.rnp import RNPConfig, Forecaster, ConvEncoder, RBFSetConv, ConvDeepSet


class ResidualDecoder(nn.Module):
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

        # Residual Network: delta_t+1 = y_t+1 - y_t = g(x_t+1, C_t)
        self.residual_net = MLP(layer_sizes, 2)  # mu, logsigma
        
        # Prediction Network: y_t+1 = f(x_t+1, C_t)
        self.prediction_net = MLP(layer_sizes, 2)  # mu, logsigma
    
    def forward(self, next_h_grid: torch.Tensor, h_grid: torch.Tensor, target_obs: Obs, context_obs: Obs, mode: str = "train") -> Normal:
        """
        Args:
            next_h_grid: (B, T, C, H_grid, W_grid) Latent spatiotemporal grid at t+1
            h_grid: (B, T, C, H_grid, W_grid) Latent spatiotemporal grid at t
            context_obs: Context points (B, 1, P, 2)
            target_obs: Target points (B, 1, P, 2)
            mode: "train", "eval_delta", "eval_full"
        """

        def retrieve_input_decoder(h_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
            B, T, P, _ = coords.shape # Use shape from coords itself

            _, _, C, Hg, Wg = h_grid.shape

            # 1. Flatten Batch * Time for efficient processing
            # h_grid: (BT, C, Hg, Wg)
            h_flat = h_grid.reshape(B*T, C, Hg, Wg)
        
            # 2. Prepare Query Coordinates
            # Flatten: (BT, P, 2)
            coords_flat = coords.reshape(B*T, P, 2)
        
            # Normalize coordinates to [-1, 1] for grid_sample
            grid_coords = 2.0 * (coords_flat - self.config.spatial_min) / (self.config.spatial_max - self.config.spatial_min) - 1.0
        
            # (BT, 1, P, 2)
            grid_coords = grid_coords.unsqueeze(1)
        
            # 3. Interpolate Features
            # sampled: (BT, C, 1, P). 
            sampled = F.grid_sample(h_flat, grid_coords, align_corners=True)

            # Reshape to (BT, P, C)
            sampled = sampled.squeeze(2).permute(0, 2, 1)
        
            # 4. Add Fourier Features (Positional Encoding).
            spatial_features = self.spatial_encoder(coords_flat) # (BT, P, enc_dim)

            return sampled, spatial_features

        if mode == "train":
            # 5. Decode. Two steps decoding
            target_coords = torch.cat([target_obs.xs, target_obs.ys], dim=-1)
            context_coords = torch.cat([context_obs.xs, context_obs.ys], dim=-1)

            target_sampled, target_spatial_features = retrieve_input_decoder(next_h_grid, target_coords)
            context_sampled, context_spatial_features = retrieve_input_decoder(h_grid, context_coords)

            # Correct prediction at t
            prediction_t_in = torch.cat([context_sampled, context_spatial_features], dim=-1)
            prediction_t_out = self.prediction_net(prediction_t_in) # (BT, P, 2)
            B, T, P, _ = context_coords.shape
            prediction_t_out = prediction_t_out.reshape(B, T, P, 2)

            # Correct prediction at t+1. Notice the detach() call.
            prediction_tp1_in = torch.cat([target_sampled, target_spatial_features], dim=-1)
            prediction_tp1_out = self.prediction_net(prediction_tp1_in).detach() # (BT, P, 2)
            B, T, P, _ = target_coords.shape
            prediction_tp1_out = prediction_tp1_out.reshape(B, T, P, 2)

            # Residual Network: delta_t+1 = y_t+1 - y_t = g(x_t+1, C_t)
            residual_tp1_in = torch.cat([target_sampled, target_spatial_features], dim=-1)
            residual_tp1_out = self.residual_net(residual_tp1_in) # (BT, P, 2)
            residual_tp1_out = residual_tp1_out.reshape(B, T, P, 2)
            
            # Reshape back to (B, T, P, 2)
            out_t_mu = prediction_t_out[..., 0:1]
            out_t_logsigma = prediction_t_out[..., 1:2]
            sigma_t = F.softplus(out_t_logsigma) + self.config.min_std
            out_t_dist = Normal(out_t_mu, sigma_t)

            out_tp1_mu = prediction_tp1_out[..., 0:1] + residual_tp1_out[..., 0:1]
            sigma_tp1_base = F.softplus(prediction_tp1_out[..., 1:2]) + self.config.min_std
            sigma_residual = F.softplus(residual_tp1_out[..., 1:2]) + self.config.min_std
            out_tp1_sigma = torch.sqrt(torch.pow(sigma_tp1_base, 2) + torch.pow(sigma_residual, 2) + 1e-8)
            out_tp1_dist = Normal(out_tp1_mu, out_tp1_sigma)

            out = [out_t_dist, out_tp1_dist]
            
            # Check if we want components for analysis
            # (We attach them to the distribution object or return extra?)
            # Let's return a dict if requested? But signature is fixed in RNP?
            # Actually RNPResidual calls this. We can change RNPResidual to handle it.
            # But for minimal changes, let's attach components to the distribution object? 
            # (Dirty but works for analysis script)
            
            out_tp1_dist.components = {
                'pred_mu': prediction_tp1_out[..., 0:1],
                'delta_mu': residual_tp1_out[..., 0:1],
                'pred_sigma': sigma_tp1_base,
                'delta_sigma': sigma_residual
            }
            
        return out


class RNPResidual(nn.Module):
    """
    Recurrent ConvCNP (Refactored).
    """
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = ConvEncoder(config)
        self.forecaster = Forecaster(config.h_dim, config.h_dim, config.lstm_num_layers)
        self.decoder = ResidualDecoder(config)
        
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
            prediction = self.decoder(r_next_expanded, r_step, target_obs, context_obs)
            
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
            
            # dists is [dist_t, dist_tp1]
            dist_tp1 = out.prediction[1]
            
            s = dist_tp1.sample()
            m = dist_tp1.mean
            std = dist_tp1.stddev
                
            obs_sample = Obs(
                xs=trg_step.xs, ys=trg_step.ys, values=s, mask=trg_step.mask, ts=trg_step.ts
            )
            obs_mean = Obs(
                xs=trg_step.xs, ys=trg_step.ys, values=m, mask=trg_step.mask, ts=trg_step.ts
            )
            obs_std = Obs(
                xs=trg_step.xs, ys=trg_step.ys, values=std, mask=trg_step.mask, ts=trg_step.ts
            )
            
            predictions.append({'sample': obs_sample, 'mean': obs_mean, 'std': obs_std})
            steps_generated += 1
            
            # Autoregressive Update
            current_input = obs_sample
            
        return predictions

if __name__ == "__main__":
    import os
    from src.models.shared.observations import slice_obs
    
    # Simple test
    print("Initializing Refactored Spatial RNP Residual...")
    cfg = RNPConfig(grid_res=64, h_dim=32) # Lower dim for test
    model = RNPResidual(cfg).cuda()
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
             # prediction is now a List[Normal] (t, t+1)
             for i, pred in enumerate(out.prediction):
                  print(f"Step {t} (Prediction {i}): Prediction Mean Shape: {pred.mean.shape}")

    # Forecast
    print("Testing Forecast...")
    # Last Context Frame
    last_ctx = slice_obs(ctx_seq, T-1, T)
    
    # Forecast Horizon 5
    preds = model.forecast(last_ctx, n_horizons=5)
    print(f"Forecast Predictions: {len(preds)}")
    print(f"Forecast 0 Shape: {preds[0].values.shape}")

