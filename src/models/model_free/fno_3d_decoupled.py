"""
fno_3d_decoupled.py — FNO-3D with Decoupled Mean and Std Networks
===================================================================
A variant of FNO-3D where the parameterisations for the mean (mu) and 
standard deviation (std) are split into two completely decoupled networks.
This allows the std branch to be heavily regularised, use lower capacities 
(fewer Fourier modes), or undergo phased training (mean-first).
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Reuse the core SpectralConv3d block
from src.models.model_free.fno_3d import SpectralConv3d


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FNO3dDecoupledConfig:
    # Context / prediction
    h_ctx:  int = 10          
    h_pred: int = 5           

    # Mean network parameters
    modes_t_mu: int = 4
    modes_h_mu: int = 8
    modes_w_mu: int = 8
    width_mu:   int = 32
    n_layers_mu: int = 4

    # Std network parameters
    modes_t_std: int = 2
    modes_h_std: int = 4
    modes_w_std: int = 4
    width_std:   int = 16
    n_layers_std: int = 4

    # Features
    use_grid: bool = True     
    use_time: bool = True     

    # Normalisation
    seq_len_ref: int = 25     
    min_std:    float = 1e-4


# ---------------------------------------------------------------------------
# Branch Network
# ---------------------------------------------------------------------------

class FNO3dBranch(nn.Module):
    """
    A single FNO-3D pipeline outputting `h_pred` channels.
    Used internally for both the mu-branch and std-branch.
    """
    def __init__(self, c_in: int, c_post: int, h_ctx: int, h_pred: int, 
                 modes_t: int, modes_h: int, modes_w: int, 
                 width: int, n_layers: int, use_grid: bool):
        super().__init__()
        self.use_grid = use_grid
        
        # Lift
        self.lift = nn.Conv3d(c_in, width, kernel_size=1)
        
        # Spectral layers
        self.spec_convs = nn.ModuleList([
            SpectralConv3d(width, width, modes_t, modes_h, modes_w)
            for _ in range(n_layers)
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv3d(width, width, kernel_size=1)
            for _ in range(n_layers)
        ])
        
        # Temporal aggregation
        self.temporal_agg = nn.Conv3d(width, width, kernel_size=(h_ctx, 1, 1))
        
        # Decoder 
        self.fc1 = nn.Linear(c_post, 128)
        self.fc2 = nn.Linear(128, h_pred)

    def forward(self, feat: torch.Tensor, grid: torch.Tensor | None) -> torch.Tensor:
        B = feat.shape[0]
        x = self.lift(feat)   # (B, width, h_ctx, H, W)
        
        for spec, skip in zip(self.spec_convs, self.skip_convs):
            x = F.gelu(spec(x) + skip(x))
            
        x = self.temporal_agg(x).squeeze(2)  # (B, width, H, W)
        
        if self.use_grid and grid is not None:
            grid_b = grid.expand(B, -1, -1, -1)
            x = torch.cat([x, grid_b], dim=1)
            
        x = x.permute(0, 2, 3, 1)      # (B, H, W, c_post)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)             # (B, H, W, h_pred)


# ---------------------------------------------------------------------------
# Decoupled Model
# ---------------------------------------------------------------------------

class FNO3dDecoupled(nn.Module):
    """
    FNO-3D with distinct and fully decoupled networks for mean and std prediction.
    Enables specialized capacities or independent optimization phases.
    """
    def __init__(self, cfg: FNO3dDecoupledConfig):
        super().__init__()
        self.cfg = cfg
        
        self.c_in = 1 + (1 if cfg.use_time else 0)
        self.grid_ch = 2 if cfg.use_grid else 0
        
        c_post_mu = cfg.width_mu + self.grid_ch
        c_post_std = cfg.width_std + self.grid_ch
        
        # Mean branch
        self.net_mu = FNO3dBranch(
            self.c_in, c_post_mu, cfg.h_ctx, cfg.h_pred,
            cfg.modes_t_mu, cfg.modes_h_mu, cfg.modes_w_mu,
            cfg.width_mu, cfg.n_layers_mu, cfg.use_grid
        )
        
        # Std branch
        self.net_std = FNO3dBranch(
            self.c_in, c_post_std, cfg.h_ctx, cfg.h_pred,
            cfg.modes_t_std, cfg.modes_h_std, cfg.modes_w_std,
            cfg.width_std, cfg.n_layers_std, cfg.use_grid
        )

    def set_phase(self, phase: str):
        """Toggle gradients for scheduled training."""
        if phase == 'mean_only':
            for p in self.net_mu.parameters(): p.requires_grad = True
            for p in self.net_std.parameters(): p.requires_grad = False
        elif phase == 'std_only':
            for p in self.net_mu.parameters(): p.requires_grad = False
            for p in self.net_std.parameters(): p.requires_grad = True
        elif phase == 'joint':
            for p in self.net_mu.parameters(): p.requires_grad = True
            for p in self.net_std.parameters(): p.requires_grad = True
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def _build_feat_volume(self, frames: torch.Tensor, times: torch.Tensor | None) -> torch.Tensor:
        feat = frames.unsqueeze(1)
        if self.cfg.use_time:
            if times is None:
                B, T = frames.shape[:2]
                times = torch.linspace(0, 1, T, device=frames.device).unsqueeze(0).expand(B, -1)
            B, T, H, W = frames.shape
            t_feat = times.view(B, 1, T, 1, 1).expand(-1, -1, -1, H, W)
            feat = torch.cat([feat, t_feat.float()], dim=1)
        return feat

    def _build_grid(self, H: int, W: int, device) -> torch.Tensor:
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([gx, gy], dim=0).unsqueeze(0)

    def forward(self, frames: torch.Tensor, times: torch.Tensor | None = None) -> List[Normal]:
        B, T_c, H, W = frames.shape
        feat = self._build_feat_volume(frames, times)
        grid = self._build_grid(H, W, frames.device) if self.cfg.use_grid else None
        
        # Forward pass on both branches
        out_mu = self.net_mu(feat, grid)
        out_std = self.net_std(feat, grid)
        
        dists = []
        for h in range(self.cfg.h_pred):
            mu = out_mu[..., h : h+1]
            sigma = F.softplus(out_std[..., h : h+1]) + self.cfg.min_std
            dists.append(Normal(mu, sigma))
            
        return dists

    def autoregressive_forecast(self, seed_frames: torch.Tensor, seed_t_start: int = 0, 
                                horizon: int = 15, num_samples: int = 10) -> List[dict]:
        """Same autoregressive inference as base FNO3D."""
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
            t_abs  = torch.arange(t_offset, t_offset + h_ctx, device=device, dtype=torch.float32)
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
            new_stack = torch.cat([f.permute(0, 3, 1, 2) for f in new_frames_for_ctx], dim=1)
            ctx      = torch.cat([ctx[:, n_slide:], new_stack], dim=1)
            t_offset += n_slide

        return preds


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    cfg = FNO3dDecoupledConfig(h_ctx=10, h_pred=5)
    model = FNO3dDecoupled(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"FNO3dDecoupled params: {n:,}")
    
    B, H, W = 2, 20, 30
    frames = torch.randn(B, cfg.h_ctx, H, W)
    times  = torch.linspace(0, 0.4, cfg.h_ctx).unsqueeze(0).expand(B, -1)
    
    # Test 'mean_only' phase
    model.set_phase('mean_only')
    assert any(p.requires_grad for p in model.net_mu.parameters())
    assert not any(p.requires_grad for p in model.net_std.parameters())
    
    dists = model(frames, times)
    loss = -dists[0].log_prob(torch.randn_like(dists[0].mean)).mean()
    loss.backward()
    
    print("Forward + Backward OK under scheduled phase.")
    print("ALL OK")
