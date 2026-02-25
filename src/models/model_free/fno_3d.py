"""
fno_3d.py — FNO-3D with true SpectralConv3d
=============================================
Applies the 3D Fourier Neural Operator over the volume (T, H, W):

  Step A: rfftn(x, dim=(-3,-2,-1))     → spectral volume (T×H×W/2+1 complex)
  Step B: Truncate to (modes_t, modes_h, modes_w) low-frequency modes
  Step C: Multiply by learnable complex weights R  (4 quadrants in T×H space)
  Step D: irfftn → back to physical domain

This is the architecture from Li et al. (2021) §4 applied to smoke forecasting.
The key difference vs. factored conv: modes_t × modes_h cross-correlations are
explicitly learned (e.g. "this wave moves diagonally in space-time").

Usage:
    model = FNO3d(FNO3dConfig(h_ctx=10, h_pred=5))
    dists = model(frames, times)   # frames: (B, h_ctx, H, W)
    preds = model.autoregressive_forecast(seed, horizon=15)
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FNO3dConfig:
    # Context / prediction
    h_ctx:  int = 10          # Context frames (T dimension of the 3D volume)
    h_pred: int = 5           # Future frames per forward pass

    # 3D spectral modes
    modes_t: int = 4          # Temporal Fourier modes  (≤ h_ctx // 2)
    modes_h: int = 8          # Spatial H Fourier modes (≤ H // 2)
    modes_w: int = 8          # Spatial W Fourier modes (≤ W // 2)

    # Network width and depth
    width:    int = 32        # Latent channel width
    n_layers: int = 4         # Number of SpectralConv3d + skip blocks

    # Features
    use_grid: bool = True     # Append (x,y) grid after temporal aggregation
    use_time: bool = True     # Append normalised t as extra input channel per frame

    # Normalisation
    seq_len_ref: int = 25     # Used to map absolute step → t_rel ∈ [0,1]
    min_std:    float = 1e-4


# ---------------------------------------------------------------------------
# SpectralConv3d  — the core of FNO-3D
# ---------------------------------------------------------------------------

class SpectralConv3d(nn.Module):
    """
    3D spectral convolution over (T, H, W).

    Because rfftn is used (real FFT in the last dim) the weight tensor only
    covers positive W frequencies.  For T and H we need both positive and
    negative modes → 4 quadrant weight tensors:

        w1 : (+T, +H)    w2 : (+T, -H)
        w3 : (-T, +H)    w4 : (-T, -H)

    Each weight has shape (C_in, C_out, modes_t, modes_h, modes_w) complex.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 modes_t: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.modes_t = modes_t
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / (in_ch * out_ch)
        shape = (in_ch, out_ch, modes_t, modes_h, modes_w)
        self.w1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mul3(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Complex einsum: (B, C_in, m_t, m_h, m_w) × (C_in, C_out, …) → (B, C_out, …)"""
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, C_in,  T, H, W)
        out : (B, C_out, T, H, W)
        """
        B, C, T, H, W = x.shape
        mt, mh, mw = self.modes_t, self.modes_h, self.modes_w

        # ---- Step A: 3D real FFT -------------------------------------------
        # Result shape: (B, C_in, T, H, W//2+1)  — last dim is real FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")

        # ---- Step B+C: Truncate modes & multiply weights -------------------
        out_ft = torch.zeros(B, self.out_ch, T, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Quadrant (+T, +H, +W)  — low positive frequencies everywhere
        out_ft[:, :, :mt,  :mh,  :mw] += self._mul3(
            x_ft[:, :, :mt,  :mh,  :mw],  self.w1)

        # Quadrant (+T, -H, +W)  — positive temporal, high negative H
        out_ft[:, :, :mt,  -mh:, :mw] += self._mul3(
            x_ft[:, :, :mt,  -mh:, :mw],  self.w2)

        # Quadrant (-T, +H, +W)  — high negative temporal, positive H
        out_ft[:, :, -mt:, :mh,  :mw] += self._mul3(
            x_ft[:, :, -mt:, :mh,  :mw],  self.w3)

        # Quadrant (-T, -H, +W)  — high negative temporal & H
        out_ft[:, :, -mt:, -mh:, :mw] += self._mul3(
            x_ft[:, :, -mt:, -mh:, :mw],  self.w4)

        # ---- Step D: inverse 3D FFT ----------------------------------------
        return torch.fft.irfftn(out_ft, s=(T, H, W), dim=(-3, -2, -1), norm="ortho")


# ---------------------------------------------------------------------------
# FNO-3D model
# ---------------------------------------------------------------------------

class FNO3d(nn.Module):
    """
    FNO-3D: spectral convolution jointly in (time × height × width).

    Pipeline:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input (B, h_ctx, H, W)  +  optional times (B, h_ctx)     │
    │       ↓  Build feature volume  (B, C_in, h_ctx, H, W)     │
    │       ↓  Conv3d lift      →    (B, width,  h_ctx, H, W)   │
    │       ↓  [SpectralConv3d + skip] × n_layers               │
    │       ↓  Temporal aggregation  (B, width, 1, H, W)        │
    │       ↓  squeeze + optional (x,y) grid → (B, C_post, H,W)│
    │       ↓  permute → (B, H, W, C_post)                      │
    │       ↓  fc1 → fc2 → 2×h_pred outputs                    │
    │  Output: List[Normal]  length = h_pred, each (B, H, W, 1) │
    └─────────────────────────────────────────────────────────────┘

    Autoregressive inference:
        Feed h_pred predicted means back to the context window → slide → repeat.
    """

    def __init__(self, cfg: FNO3dConfig):
        super().__init__()
        self.cfg = cfg

        # Input channel count
        # smoke (1) + time (1, if use_time) → C_in
        self.c_in  = 1 + (1 if cfg.use_time else 0)
        self.grid_ch = 2 if cfg.use_grid else 0
        self.c_post  = cfg.width + self.grid_ch   # channels entering the MLP decoder

        # ---- Lift: pointwise Conv3d  (C_in → width) -------------------------
        self.lift = nn.Conv3d(self.c_in, cfg.width, kernel_size=1)

        # ---- SpectralConv3d blocks -------------------------------------------
        self.spec_convs = nn.ModuleList([
            SpectralConv3d(cfg.width, cfg.width,
                           cfg.modes_t, cfg.modes_h, cfg.modes_w)
            for _ in range(cfg.n_layers)
        ])
        # Skip connections (pointwise 3D conv)
        self.skip_convs = nn.ModuleList([
            nn.Conv3d(cfg.width, cfg.width, kernel_size=1)
            for _ in range(cfg.n_layers)
        ])

        # ---- Temporal aggregation: collapse h_ctx → 1 -----------------------
        self.temporal_agg = nn.Conv3d(cfg.width, cfg.width,
                                       kernel_size=(cfg.h_ctx, 1, 1))

        # ---- MLP decoder (B, H, W, C_post) → (B, H, W, 2*h_pred) ----------
        self.fc1 = nn.Linear(self.c_post, 128)
        self.fc2 = nn.Linear(128, 2 * cfg.h_pred)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_feat_volume(self, frames: torch.Tensor,
                            times: torch.Tensor | None) -> torch.Tensor:
        """
        Build the 3D feature volume fed to the spectral layers.

        frames : (B, h_ctx, H, W)
        times  : (B, h_ctx) normalised to [0, 1], or None
        returns: (B, C_in, h_ctx, H, W)
        """
        # smoke: (B, 1, h_ctx, H, W)
        feat = frames.unsqueeze(1)                     # (B, 1, h_ctx, H, W)

        if self.cfg.use_time:
            if times is None:
                B, T = frames.shape[:2]
                times = torch.linspace(0, 1, T, device=frames.device
                                       ).unsqueeze(0).expand(B, -1)

            # Broadcast time scalar to all pixels
            B, T, H, W = frames.shape
            # t: (B, 1, h_ctx, 1, 1) → expand to (B, 1, h_ctx, H, W)
            t_feat = times.view(B, 1, T, 1, 1).expand(-1, -1, -1, H, W)
            feat = torch.cat([feat, t_feat.float()], dim=1)  # (B, 2, h_ctx, H, W)

        return feat   # (B, C_in, h_ctx, H, W)

    def _build_grid(self, H: int, W: int, device) -> torch.Tensor:
        """(x, y) normalised grid: (1, 2, H, W)"""
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([gx, gy], dim=0).unsqueeze(0)   # (1, 2, H, W)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, frames: torch.Tensor,
                times: torch.Tensor | None = None) -> List[Normal]:
        """
        frames : (B, h_ctx, H, W)  — context smoke values in [0, 1]
        times  : (B, h_ctx)        — relative times in [0, 1], or None (auto)
        returns: List[Normal] of length h_pred
                 Each Normal: .mean, .stddev shape (B, H, W, 1)
        """
        B, T_c, H, W = frames.shape

        # ---- Build 3D feature volume  (B, C_in, h_ctx, H, W) ---------------
        feat = self._build_feat_volume(frames, times)

        # ---- Lift to latent width -------------------------------------------
        x = self.lift(feat)                             # (B, width, h_ctx, H, W)

        # ---- SpectralConv3d blocks ------------------------------------------
        for spec, skip in zip(self.spec_convs, self.skip_convs):
            # Spectral path + residual skip (pointwise)
            x = F.gelu(spec(x) + skip(x))

        # ---- Temporal aggregation → (B, width, H, W) -----------------------
        x = self.temporal_agg(x)                        # (B, width, 1, H, W)
        x = x.squeeze(2)                                # (B, width, H, W)

        # ---- Append spatial grid -------------------------------------------
        if self.cfg.use_grid:
            grid = self._build_grid(H, W, frames.device).expand(B, -1, -1, -1)
            x = torch.cat([x, grid], dim=1)            # (B, width+2, H, W)

        # ---- MLP decode  ---------------------------------------------------
        x = x.permute(0, 2, 3, 1)                      # (B, H, W, C_post)
        x = F.gelu(self.fc1(x))                        # (B, H, W, 128)
        out = self.fc2(x)                               # (B, H, W, 2*h_pred)

        dists = []
        for h in range(self.cfg.h_pred):
            mu    = out[..., 2*h   : 2*h+1]
            sigma = F.softplus(out[..., 2*h+1 : 2*h+2]) + self.cfg.min_std
            dists.append(Normal(mu, sigma))

        return dists   # List[Normal], each (B, H, W, 1)

    # ------------------------------------------------------------------
    # Autoregressive forecast
    # ------------------------------------------------------------------

    def autoregressive_forecast(
        self,
        seed_frames:  torch.Tensor,    # (1, h_ctx, H, W) or (h_ctx, H, W)
        seed_t_start: int = 0,         # absolute time index of seed_frames[0]
        horizon:      int = 15,
        num_samples:  int = 10,
    ) -> List[dict]:
        """
        Autoregressively predict `horizon` future frames.

        The h_ctx context window slides forward by h_pred at each step.
        Times are normalised by cfg.seq_len_ref.

        Returns
        -------
        preds : list[dict] length = horizon
            'sample' : (num_samples, H, W)  float16  — S sampled trajectories
            'mean'   : (H, W)               float16  — distribution mean
            'std'    : (H, W)               float16  — distribution std-dev
        """
        if seed_frames.dim() == 3:
            seed_frames = seed_frames.unsqueeze(0)   # → (1, h_ctx, H, W)

        h_ctx  = self.cfg.h_ctx
        h_pred = self.cfg.h_pred
        ref    = max(self.cfg.seq_len_ref - 1, 1)
        device = seed_frames.device

        # Expand to S sample trajectories
        ctx = seed_frames.expand(num_samples, -1, -1, -1).clone()  # (S, h_ctx, H, W)
        t_offset = seed_t_start   # absolute step of the first context frame
        preds    = []

        while len(preds) < horizon:
            # Relative times for this context window
            t_abs  = torch.arange(t_offset, t_offset + h_ctx,
                                   device=device, dtype=torch.float32)
            times  = (t_abs / ref).unsqueeze(0).expand(num_samples, -1)  # (S, h_ctx)

            with torch.no_grad():
                dists = self.forward(ctx, times)   # List[Normal], each (S,H,W,1)

            new_frames_for_ctx = []
            for d in dists:
                if len(preds) >= horizon:
                    break

                sample_np = (d.sample()[:, :, :, 0]
                              .cpu().to(torch.float16).numpy())  # (S, H, W)
                mu_np     = d.mean   [0, :, :, 0].cpu().to(torch.float16).numpy()  # scalar
                std_np    = d.stddev [0, :, :, 0].cpu().to(torch.float16).numpy()

                preds.append({'sample': sample_np, 'mean': mu_np, 'std': std_np})
                new_frames_for_ctx.append(d.mean)   # (S, H, W, 1)

            # Slide context window by len(new_frames_for_ctx)
            n_slide   = len(new_frames_for_ctx)
            new_stack = torch.cat(
                [f.permute(0, 3, 1, 2) for f in new_frames_for_ctx], dim=1
            )  # (S, n_slide, H, W)

            ctx      = torch.cat([ctx[:, n_slide:], new_stack], dim=1)
            t_offset += n_slide

        return preds


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg = FNO3dConfig(
        h_ctx=10, h_pred=5,
        modes_t=4, modes_h=6, modes_w=6,
        width=16, n_layers=4,
        use_grid=True, use_time=True,
    )
    model = FNO3d(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"FNO3d params: {n:,}  (h_ctx={cfg.h_ctx}, h_pred={cfg.h_pred})")

    B, H, W = 2, 20, 30
    frames = torch.randn(B, cfg.h_ctx, H, W)
    times  = torch.linspace(0, 0.4, cfg.h_ctx).unsqueeze(0).expand(B, -1)

    dists = model(frames, times)
    assert len(dists) == cfg.h_pred
    assert dists[0].mean.shape == (B, H, W, 1)
    print(f"Forward OK  h_pred={cfg.h_pred}  mu shape={dists[0].mean.shape}")

    # Rollout
    seed  = torch.randn(1, cfg.h_ctx, H, W)
    preds = model.autoregressive_forecast(seed, horizon=15, num_samples=5)
    assert len(preds) == 15
    assert preds[0]['sample'].shape == (5, H, W)
    print(f"Rollout OK  horizon=15  sample={preds[0]['sample'].shape}")
    print("ALL OK")
