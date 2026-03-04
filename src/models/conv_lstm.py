"""
conv_lstm.py — ConvLSTM Baseline Model
=========================================
Applies a generic multi-layer Convolutional LSTM to the input sequences.

Usage:
    model = ConvLSTMModel(ConvLSTMConfig(h_ctx=10, h_pred=5))
    dists = model(frames, times)
    preds = model.autoregressive_forecast(seed, horizon=15)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

@dataclass
class ConvLSTMConfig:
    # Context / prediction
    h_ctx:  int = 10          # Context frames
    h_pred: int = 5           # Future frames per forward pass

    # Architecture
    hidden_dim: int = 32      # Hidden state channels for LSTM layers
    n_layers: int = 3         # Number of ConvLSTM layers
    kernel_size: int = 3      # Kernel size for convolutions

    # Features
    use_grid: bool = True     # Append (x,y) grid after temporal aggregation
    use_time: bool = True     # Append normalised t as extra input channel per frame

    # Normalisation
    seq_len_ref: int = 25     # Used to map absolute step → t_rel ∈ [0,1]
    min_std: float = 1e-4


class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell processing 2D spatial feature maps over time.
    """
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Combined convolution for all 4 gates: Input, Forget, Cell, Output
        self.conv = nn.Conv2d(
            in_channels=in_ch + hidden_ch,
            out_channels=4 * hidden_ch,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        B, C, H, W = x.shape
        
        if state is None:
            h_cur = torch.zeros(B, self.hidden_ch, H, W, device=x.device, dtype=x.dtype)
            c_cur = torch.zeros(B, self.hidden_ch, H, W, device=x.device, dtype=x.dtype)
        else:
            h_cur, c_cur = state

        # Concatenate x and current hidden state
        combined = torch.cat([x, h_cur], dim=1)  # (B, in_ch + hidden_ch, H, W)
        
        gates = self.conv(combined)              # (B, 4 * hidden_ch, H, W)
        
        # Split into gates
        i_gate, f_gate, c_gate, o_gate = torch.split(gates, self.hidden_ch, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        c_tilde = torch.tanh(c_gate)
        
        # Update cell state and hidden state
        c_next = f * c_cur + i * c_tilde
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class ConvLSTMModel(nn.Module):
    def __init__(self, cfg: ConvLSTMConfig):
        super().__init__()
        self.cfg = cfg

        # Input channels setup
        self.c_in = 1 + (1 if cfg.use_time else 0)
        self.grid_ch = 2 if cfg.use_grid else 0
        self.c_post = cfg.hidden_dim + self.grid_ch
        
        # ConvLSTM Layers
        cells = []
        for i in range(cfg.n_layers):
            cur_in = self.c_in if i == 0 else cfg.hidden_dim
            cells.append(ConvLSTMCell(cur_in, cfg.hidden_dim, cfg.kernel_size))
        self.cells = nn.ModuleList(cells)
        
        # MLP decoder (B, H, W, C_post) → (B, H, W, 2*h_pred)
        self.fc1 = nn.Linear(self.c_post, 128)
        self.fc2 = nn.Linear(128, 2 * cfg.h_pred)

    def _build_feat_volume(self, frames: torch.Tensor,
                           times: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Build the feature seq fed to ConvLSTM.
        frames : (B, h_ctx, H, W)
        times  : (B, h_ctx) normalised to [0, 1], or None
        returns: (B, h_ctx, C_in, H, W)
        """
        # (B, h_ctx, 1, H, W)
        feat = frames.unsqueeze(2)
        
        if self.cfg.use_time:
            if times is None:
                B, T = frames.shape[:2]
                times = torch.linspace(0, 1, T, device=frames.device
                                       ).unsqueeze(0).expand(B, -1)

            B, T, H, W = frames.shape
            # t: (B, h_ctx, 1, 1, 1) → expand to (B, h_ctx, 1, H, W)
            t_feat = times.view(B, T, 1, 1, 1).expand(-1, -1, -1, H, W)
            feat = torch.cat([feat, t_feat.float()], dim=2)  # (B, h_ctx, 2, H, W)

        return feat

    def _build_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """(x, y) normalised grid: (1, 2, H, W)"""
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([gx, gy], dim=0).unsqueeze(0)

    def forward(self, frames: torch.Tensor,
                times: Optional[torch.Tensor] = None) -> List[Normal]:
        """
        frames : (B, h_ctx, H, W) context
        times  : (B, h_ctx) relative times
        returns: List[Normal] len h_pred. Each has shape (B, H, W, 1).
        """
        B, T_c, H, W = frames.shape

        # (B, T_c, C_in, H, W)
        seq = self._build_feat_volume(frames, times)
        
        # Ensure B == batch_size because states are kept across time steps
        layer_states = [None] * self.cfg.n_layers
        
        # Temporal Recurrence
        for t in range(T_c):
            # Input for this step (B, C_in, H, W)
            x_t = seq[:, t, :, :, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h_next, state_next = cell(x_t, layer_states[layer_idx])
                layer_states[layer_idx] = state_next
                x_t = h_next
        x = x_t  # (B, width, H, W)
        
        if self.cfg.use_grid:
            grid = self._build_grid(H, W, frames.device).expand(B, -1, -1, -1)
            x = torch.cat([x, grid], dim=1)     # (B, width+2, H, W)
            
        # MLP decoder
        x = x.permute(0, 2, 3, 1)               # (B, H, W, C_post)
        x = F.gelu(self.fc1(x))                 # (B, H, W, 128)
        out = self.fc2(x)                       # (B, H, W, 2*h_pred)
        
        dists = []
        for h in range(self.cfg.h_pred):
            mu    = out[..., 2*h   : 2*h+1]
            sigma = F.softplus(out[..., 2*h+1 : 2*h+2]) + self.cfg.min_std
            dists.append(Normal(mu, sigma))

        return dists

    def autoregressive_forecast(
        self,
        seed_frames:  torch.Tensor,    # (1, h_ctx, H, W) or (h_ctx, H, W)
        seed_t_start: int = 0,         # absolute time index of seed_frames[0]
        horizon:      int = 15,
        num_samples:  int = 10,
    ) -> List[dict]:
        if seed_frames.dim() == 3:
            seed_frames = seed_frames.unsqueeze(0)

        h_ctx  = self.cfg.h_ctx
        h_pred = self.cfg.h_pred
        ref    = max(self.cfg.seq_len_ref - 1, 1)
        device = seed_frames.device

        ctx = seed_frames.expand(num_samples, -1, -1, -1).clone()
        t_offset = seed_t_start
        preds    = []

        while len(preds) < horizon:
            t_abs  = torch.arange(t_offset, t_offset + h_ctx,
                                   device=device, dtype=torch.float32)
            times  = (t_abs / ref).unsqueeze(0).expand(num_samples, -1)

            with torch.no_grad():
                dists = self.forward(ctx, times)

            new_frames_for_ctx = []
            for d in dists:
                if len(preds) >= horizon:
                    break

                sampled = d.sample()
                sample_np = sampled[..., 0].cpu().to(torch.float16).numpy()
                mu_np     = d.mean[..., 0].cpu().to(torch.float16).numpy()
                std_np    = d.stddev[..., 0].cpu().to(torch.float16).numpy()

                preds.append({'sample': sample_np, 'mean': mu_np, 'std': std_np})
                new_frames_for_ctx.append(sampled)

            n_slide   = len(new_frames_for_ctx)
            new_stack = torch.cat(
                [f.permute(0, 3, 1, 2) for f in new_frames_for_ctx], dim=1
            )

            ctx      = torch.cat([ctx[:, n_slide:], new_stack], dim=1)
            t_offset += n_slide

        return preds


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ConvLSTMConfig(
        h_ctx=10, h_pred=5,
        hidden_dim=16, n_layers=2,
        use_grid=True, use_time=True
    )
    model = ConvLSTMModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"ConvLSTM params: {n:,}  (h_ctx={cfg.h_ctx}, h_pred={cfg.h_pred})")

    B, H, W = 2, 20, 30
    frames = torch.randn(B, cfg.h_ctx, H, W)
    times  = torch.linspace(0, 0.4, cfg.h_ctx).unsqueeze(0).expand(B, -1)

    dists = model(frames, times)
    assert len(dists) == cfg.h_pred
    assert dists[0].mean.shape == (B, H, W, 1)
    print(f"Forward OK  h_pred={cfg.h_pred}  mu shape={dists[0].mean.shape}")

    seed  = torch.randn(1, cfg.h_ctx, H, W)
    preds = model.autoregressive_forecast(seed, horizon=15, num_samples=5)
    assert len(preds) == 15
    assert preds[0]['sample'].shape == (5, H, W)
    print(f"Rollout OK  horizon=15  sample={preds[0]['sample'].shape}")
    print("ALL OK")
