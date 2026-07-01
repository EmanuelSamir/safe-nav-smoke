from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import model_validator
from torch.distributions import Normal

from src.utils.config_utils import StrictBaseModel
from src.models.shared.schemas import FNOConfig




class SpectralConv3d(nn.Module):
    """3D spectral convolution over (T, H, W).

    Because rfftn is used (real FFT in the last dim) the weight tensor only
    covers positive W frequencies.  For T and H we need both positive and
    negative modes → 4 quadrant weight tensors:

        w1 : (+T, +H)    w2 : (+T, -H)
        w3 : (-T, +H)    w4 : (-T, -H)

    Each weight has shape (C_in, C_out, modes_t, modes_h, modes_w) complex.
    """

    def __init__(self, in_ch: int, out_ch: int, modes_t: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
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
        """X   : (B, C_in,  T, H, W)
        out : (B, C_out, T, H, W)
        """
        B, C, T, H, W = x.shape
        mt, mh, mw = self.modes_t, self.modes_h, self.modes_w

        # Result shape: (B, C_in, T, H, W//2+1)  — last dim is real FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")

        out_ft = torch.zeros(B, self.out_ch, T, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        # Quadrant (+T, +H, +W) - low positive frequencies everywhere
        out_ft[:, :, :mt, :mh, :mw] += self._mul3(x_ft[:, :, :mt, :mh, :mw], self.w1)

        # Quadrant (+T, -H, +W) - positive temporal, high negative H
        out_ft[:, :, :mt, -mh:, :mw] += self._mul3(x_ft[:, :, :mt, -mh:, :mw], self.w2)

        # Quadrant (-T, +H, +W) - high negative temporal, positive H
        out_ft[:, :, -mt:, :mh, :mw] += self._mul3(x_ft[:, :, -mt:, :mh, :mw], self.w3)

        # Quadrant (-T, -H, +W) - high negative temporal & H
        out_ft[:, :, -mt:, -mh:, :mw] += self._mul3(x_ft[:, :, -mt:, -mh:, :mw], self.w4)

        return torch.fft.irfftn(out_ft, s=(T, H, W), dim=(-3, -2, -1), norm="ortho")


class FNO(nn.Module):
    """FNO: spectral convolution jointly in (time × height × width).

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
    """

    def __init__(self, cfg: FNOConfig):
        super().__init__()
        self.cfg = cfg

        # Input channel count
        # 1 channel (greyscale) + time (1, if use_time) + spatial grid (2, if use_grid) → C_in
        self.c_in = 1 + (1 if cfg.use_time else 0) + (2 if cfg.use_grid else 0)
        self.c_post = cfg.width

        self.lift = nn.Conv3d(self.c_in, cfg.width, kernel_size=1)

        self.spec_convs = nn.ModuleList(
            [
                SpectralConv3d(cfg.width, cfg.width, cfg.modes_t, cfg.modes_h, cfg.modes_w)
                for _ in range(cfg.n_layers)
            ]
        )

        self.skip_convs = nn.ModuleList(
            [nn.Conv3d(cfg.width, cfg.width, kernel_size=1) for _ in range(cfg.n_layers)]
        )

        # Temporal aggregation
        self.temporal_agg = nn.Conv3d(cfg.width, cfg.width, kernel_size=(cfg.h_ctx, 1, 1))

        # (B, H, W, C_post) → (B, H, W, 2*h_pred) or (B, H, W, h_pred)
        self.fc1 = nn.Linear(self.c_post, 128)
        out_features = 2 * cfg.h_pred if cfg.is_probabilistic else cfg.h_pred
        self.fc2 = nn.Linear(128, out_features)

    def _build_feat_volume(self, frames: torch.Tensor, times: torch.Tensor | None) -> torch.Tensor:
        """Build the 3D feature volume fed to the spectral layers.

        frames : (B, C, h_ctx, H, W) or (B, h_ctx, H, W)
        times  : (B, h_ctx) normalised to [0, 1], or None
        returns: (B, C_in, h_ctx, H, W)
        """
        B, T, H, W = frames.shape
        # smoke: (B, 1, h_ctx, H, W)
        feat = frames.unsqueeze(1)

        if self.cfg.use_time:
            if times is None:
                times = torch.linspace(0, 1, T, device=frames.device).unsqueeze(0).expand(B, -1)

            # t: (B, 1, h_ctx, 1, 1) → expand to (B, 1, h_ctx, H, W)
            t_feat = times.view(B, 1, T, 1, 1).expand(-1, -1, -1, H, W)
            feat = torch.cat([feat, t_feat.float()], dim=1)  # (B, 2, h_ctx, H, W)

        if self.cfg.use_grid:
            # grid2d: (1, 2, H, W)
            grid2d = self._build_grid(H, W, frames.device)
            # grid3d: (B, 2, T, H, W)
            grid3d = grid2d.unsqueeze(2).expand(B, -1, T, -1, -1)
            feat = torch.cat([feat, grid3d.float()], dim=1)

        return feat

    def _build_grid(self, H: int, W: int, device) -> torch.Tensor:
        """(x, y) normalised grid: (1, 2, H, W)"""
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([gx, gy], dim=0).unsqueeze(0)

    def forward(self, frames: torch.Tensor, times: torch.Tensor | None = None) -> list:
        """Frames : (B, h_ctx, H, W)  — context smoke values in [0, 1].

        times  : (B, h_ctx)        — relative times in [0, 1], or None (auto)
        returns: List[Normal] or List[torch.Tensor] of length h_pred
        """
        B, T_c, H, W = frames.shape

        # Build 3D feature volume  (B, C_in, h_ctx, H, W)
        feat = self._build_feat_volume(frames, times)

        x = self.lift(feat)

        for spec, skip in zip(self.spec_convs, self.skip_convs):
            # Spectral path + residual skip (pointwise)
            x = F.gelu(spec(x) + skip(x))

        # Temporal aggregation -> (B, width, H, W)
        x = self.temporal_agg(x)  # (B, width, 1, H, W)
        x = x.squeeze(2)  # (B, width, H, W)

        # MLP decode
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C_post)
        x = F.gelu(self.fc1(x))  # (B, H, W, 128)
        out = self.fc2(x)

        if not self.cfg.is_probabilistic:
            return out

        dists = []

        for h in range(self.cfg.h_pred):
            mu = out[..., 2 * h : 2 * h + 1]
            sigma = F.softplus(out[..., 2 * h + 1 : 2 * h + 2]) + self.cfg.min_std
            dists.append(Normal(mu, sigma))

        return dists  # List[Normal], each (B, H, W, 1)

    def autoregressive_forecast(
        self,
        seed_frames: torch.Tensor,  # (1, h_ctx, H, W) or (h_ctx, H, W)
        seed_t_start: int = 0,  # absolute time index of seed_frames[0]
        horizon: int = 15,
        num_samples: int = 10,
        mode: str = "mean",  # 'mean', 'sample'
    ) -> List[dict]:

        # Standardize seed_frames to always have batch dim
        if seed_frames.dim() == 3:  # (h_ctx, H, W) -> standardize to 4D
            seed_frames = seed_frames.unsqueeze(0)

        h_ctx = self.cfg.h_ctx
        h_pred = self.cfg.h_pred
        ref = max(self.cfg.sequence_length - 1, 1)
        device = seed_frames.device

        # Expand to S sample trajectories
        ctx = seed_frames.expand(num_samples, -1, -1, -1).clone()  # (S, h_ctx, H, W)
        t_offset = seed_t_start  # absolute step of the first context frame
        preds = []

        while len(preds) < horizon:
            t_abs = torch.arange(t_offset, t_offset + h_ctx, device=device, dtype=torch.float32)
            times = (t_abs / ref).unsqueeze(0).expand(num_samples, -1)  # (S, h_ctx)

            with torch.no_grad():
                out_forward = self.forward(ctx, times)

            new_frames_for_ctx = []
            
            if not self.cfg.is_probabilistic:
                for h in range(h_pred):
                    if len(preds) >= horizon:
                        break
                    sampled = out_forward[..., h:h+1]
                    sample_np = sampled[..., 0].cpu().to(torch.float16).numpy()
                    preds.append({"sample": sample_np, "mean": sample_np})
                    new_frames_for_ctx.append(sampled)
            else:
                eps = torch.randn(num_samples, 1, 1, 1, device=device)
                for d in out_forward:
                    if len(preds) >= horizon:
                        break

                    if mode == "mean":
                        sampled = d.mean
                    elif mode == "sample":
                        sampled = d.mean + d.stddev * eps
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    sample_np = sampled[..., 0].cpu().to(torch.float16).numpy()
                    mu_np = d.mean[..., 0].cpu().to(torch.float16).numpy()
                    std_np = d.stddev[..., 0].cpu().to(torch.float16).numpy()

                    preds.append({"sample": sample_np, "mean": mu_np, "std": std_np})
                    new_frames_for_ctx.append(sampled)

            # Slide context window by len(new_frames_for_ctx)
            n_slide = len(new_frames_for_ctx)
            new_stack = torch.cat(
                [f.permute(0, 3, 1, 2) for f in new_frames_for_ctx], dim=1
            )  # Output shape: (S, n_slide, H, W)

            # Safely append new_stack and keep exactly the last h_ctx elements
            ctx = torch.cat([ctx, new_stack], dim=1)[:, -h_ctx:]
            t_offset += n_slide

        return preds


def main():
    print("Starting FNO sanity check...")
    fno_cfg = FNOConfig(
        h_ctx=10,
        h_pred=5,
        modes_t=4,
        modes_h=8,
        modes_w=8,
        width=32,
        n_layers=4,
        use_grid=True,
        use_time=True,
        min_std=1e-4,
        sequence_length=25
    )

    # 2. Instantiate FNO model
    model = FNO(fno_cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"FNO params: {n:,}  (h_ctx={fno_cfg.h_ctx}, h_pred={fno_cfg.h_pred})")

    B, H, W = 2, 20, 30
    frames = torch.randn(B, fno_cfg.h_ctx, H, W)
    times = torch.linspace(0, 0.4, fno_cfg.h_ctx).unsqueeze(0).expand(B, -1)

    dists = model(frames, times)
    assert len(dists) == fno_cfg.h_pred
    assert dists[0].mean.shape == (B, H, W, 1)
    print(f"Forward OK  h_pred={fno_cfg.h_pred}  mu shape={dists[0].mean.shape}")

    # Rollout
    seed = torch.randn(1, fno_cfg.h_ctx, H, W)
    preds = model.autoregressive_forecast(seed, horizon=15, num_samples=5)
    assert len(preds) == 15
    assert preds[0]["sample"].shape == (5, H, W)
    print(f"Rollout OK  horizon=15  sample={preds[0]['sample'].shape}")

    # 3. Test validation error raising
    print("Testing FNOConfig validation constraints...")
    try:
        FNOConfig(
            h_ctx=10,
            h_pred=5,
            modes_t=6,
            modes_h=8,
            modes_w=8,
            width=32,
            n_layers=4,
            use_grid=True,
            use_time=True,
            sequence_length=25,
            min_std=1e-4,
        )
        raise RuntimeError("Validation failed to raise error for modes_t > h_ctx // 2")
    except ValueError:
        print("Validation error raised correctly for invalid modes_t")

    print("ALL OK")


if __name__ == "__main__":
    main()
