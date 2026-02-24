"""
Fourier Neural Operator (FNO) for dense smoke-field next-step prediction.

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs" (2021).

Input shape:  (B, H, W, 1)   – current smoke density at time t
Output shape: (B, H, W, 1)   – predicted smoke density at time t+1

The model can optionally concatenate a spatial grid to the input channel
to help the network locate itself in the domain (see `use_grid`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class FNOConfig:
    modes1: int = 12          # Fourier modes retained along dim-1
    modes2: int = 12          # Fourier modes retained along dim-2
    width: int = 32           # Latent channel width
    use_grid: bool = True     # Concatenate (x, y) coordinate grid to input
    n_layers: int = 4         # Number of Fourier + skip-connection blocks
    min_std: float = 1e-4     # Minimum predicted std-dev (numerical stability)
    forecast_horizon: int = 1 # Steps predicted per forward pass (H)
    use_uncertainty_input: bool = False
    # When True, forward() expects (B, H, W, 2): [smoke_mean, smoke_std].
    # During teacher-forcing   std = 0.
    # During scheduled sampling std = predicted σ from previous step.
    # Lets the model learn to propagate its own uncertainty.


# ---------------------------------------------------------------------------
# Spectral convolution layer
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """
    2-D spectral convolution: multiplies truncated Fourier coefficients by
    learnable complex weights.  Only the lowest `modes1 × modes2` frequency
    modes (+ their conjugate-symmetric counterparts) are kept.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               modes1, modes2, dtype=torch.cfloat))

    # (B, C_in, X, Y/2+1), (C_in, C_out, X, Y/2+1) -> (B, C_out, X, Y/2+1)
    @staticmethod
    def _cmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Hx, Wy = x.shape

        x_ft = torch.fft.rfft2(x)  # (B, C, Hx, Wy//2+1)

        out_ft = torch.zeros(B, self.out_channels, Hx, Wy // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Low-frequency block (top-left corner)
        out_ft[:, :, :self.modes1, :self.modes2] = self._cmul(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Conjugate-symmetric block (bottom-left corner)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._cmul(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(Hx, Wy))


# ---------------------------------------------------------------------------
# FNO2d model
# ---------------------------------------------------------------------------

class FNO2d(nn.Module):
    """
    Fourier Neural Operator for autoregressive smoke-field prediction.

    Forward call signature:
        x_in : (B, H, W, 1)  – current dense smoke frame
    Returns:
        x_out: (B, H, W, 1)  – predicted next dense smoke frame

    The model is deliberately kept stateless so that autoregressive rollouts
    are handled externally (feeding predictions back as the next input).
    """

    def __init__(self, cfg: FNOConfig):
        super().__init__()
        self.modes1 = cfg.modes1
        self.modes2 = cfg.modes2
        self.width = cfg.width
        self.use_grid = cfg.use_grid
        self.n_layers = cfg.n_layers
        self.min_std = cfg.min_std
        self.forecast_horizon = cfg.forecast_horizon
        self.use_uncertainty_input = cfg.use_uncertainty_input

        # Input channels:
        #   smoke channels: 1 (mean only) or 2 (mean + std)
        #   + 2 if use_grid (x, y coordinates)
        smoke_ch = 2 if cfg.use_uncertainty_input else 1
        in_ch = smoke_ch + (2 if cfg.use_grid else 0)

        # Lift to latent space
        self.fc0 = nn.Linear(in_ch, self.width)

        # Spectral + skip-connection layers
        self.convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(cfg.n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(cfg.n_layers)
        ])

        # Probabilistic decoder: 2 params (mu, log-sigma) x forecast_horizon steps
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2 * self.forecast_horizon)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grid(shape, device) -> torch.Tensor:
        """Returns a (B, H, W, 2) tensor with (x, y) in [0, 1]."""
        B, H, W = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, W, device=device)  # (W,)
        gy = torch.linspace(0, 1, H, device=device)  # (H,)
        # Expand to (B, H, W, 1) each
        gx = gx.view(1, 1, W, 1).expand(B, H, W, 1)
        gy = gy.view(1, H, 1, 1).expand(B, H, W, 1)
        return torch.cat([gx, gy], dim=-1)           # (B, H, W, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # API symmetry with RNP (stateless — state is always None)
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int = 1, device=None):
        """No-op: FNO is stateless. Returns None so rollout scripts work."""
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> list:
        """
        x : (B, H, W, C_in)
            If use_uncertainty_input=False  → C_in = 1       (smoke mean)
            If use_uncertainty_input=True   → C_in = 2       (smoke mean, smoke std)
            (+2 channels if use_grid is enabled for spatial coordinates)
        returns : List[Normal]  length = forecast_horizon
            Each Normal has .mean and .stddev of shape (B, H, W, 1).
            Index 0 = t+1, Index 1 = t+2, ...
        """
        if self.use_grid:
            grid = self._build_grid(x.shape, x.device)  # (B, H, W, 2)
            x = torch.cat([x, grid], dim=-1)             # (B, H, W, 3)

        # Lift  (B, H, W, C_in) -> (B, H, W, width)
        x = self.fc0(x)
        # (B, H, W, width) -> (B, width, H, W)  for conv layers
        x = x.permute(0, 3, 1, 2)

        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)

        # (B, width, H, W) -> (B, H, W, width) for MLP
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        out = self.fc2(x)   # (B, H, W, 2 * forecast_horizon)

        # Split into forecast_horizon (mu, sigma) pairs
        dists = []
        for h in range(self.forecast_horizon):
            mu    = out[..., 2*h  : 2*h+1]                       # (B, H, W, 1)
            sigma = F.softplus(out[..., 2*h+1: 2*h+2]) + self.min_std
            dists.append(Normal(mu, sigma))
        return dists

    # ------------------------------------------------------------------
    # Autoregressive rollout  (mirrors RNP.autoregressive_forecast API)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def autoregressive_forecast(
        self,
        seed_frame: torch.Tensor,
        horizon: int,
        num_samples: int = 1,
        H: int = None,
        W: int = None,
    ):
        """
        Roll out the FNO autoregressively for `horizon` total steps.

        Parameters
        ----------
        seed_frame : Tensor
            Shape (1, H, W, 1) — smoke frame at time t.
            Or (1, H, W, 2) if use_uncertainty_input=True (std will default to 0).
        horizon : int
            Total number of future steps to predict.
        num_samples : int
            - use_uncertainty_input=False: independent sample trajectories (Monte-Carlo).
            - use_uncertainty_input=True : kept for API compat; rollout is deterministic
              (μ, σ propagation), samples are drawn from the output Normal at each step.
        H, W : int, optional
            Grid dimensions. Inferred from seed_frame if not provided.

        Returns
        -------
        preds : list[dict]  length = horizon
            Each dict contains:
              'sample' : np.ndarray  (num_samples, H, W)  float16  (or mean tiled)
              'mean'   : np.ndarray  (H, W)               float16
              'std'    : np.ndarray  (H, W)               float16

        Notes
        -----
        With forecast_horizon=H, only ceil(horizon/H) forward passes are needed.
        Each pass produces H predicted frames; the last frame seeds the next pass.
        """
        if seed_frame.dim() == 3:
            seed_frame = seed_frame.unsqueeze(0)

        if H is None:
            H = seed_frame.shape[1]
        if W is None:
            W = seed_frame.shape[2]

        if self.use_uncertainty_input:
            # Seed is (1, H, W, 1) or (1, H, W, 2)
            if seed_frame.shape[-1] == 1:
                # Pad with zero std
                seed_2ch = torch.cat(
                    [seed_frame, torch.zeros_like(seed_frame)], dim=-1)  # (1, H, W, 2)
            else:
                seed_2ch = seed_frame
            # In uncertainty mode no sample expansion; rollout propagates (mu, sigma) pairs
            x_t = seed_2ch          # (1, H, W, 2)
        else:
            # Expand seed to (num_samples, H, W, 1) for independent rollouts
            x_t = seed_frame.expand(num_samples, -1, -1, -1).clone()  # (S, H, W, 1)

        preds = []

        while len(preds) < horizon:
            dists = self.forward(x_t)   # List[Normal] length forecast_horizon
            last_d = dists[0]

            for dist in dists:
                if len(preds) >= horizon:
                    break
                last_d = dist

                if self.use_uncertainty_input:
                    # Deterministic: record (mu, sigma) from the distribution directly
                    mu_np  = dist.mean[0, :, :, 0].detach().cpu().to(torch.float16).numpy()
                    std_np = dist.stddev[0, :, :, 0].detach().cpu().to(torch.float16).numpy()
                    # Also draw samples for API compat (adds stochastic variation on top)
                    samples = dist.sample().expand(num_samples, -1, -1, -1) # (S, H, W, 1)
                    sample_np = (samples[:, :, :, 0]
                                 .detach().cpu().to(torch.float16).numpy())  # (S, H, W)
                else:
                    sample = dist.sample()  # (S, H, W, 1)
                    sample_np = (sample[:, :, :, 0]
                                 .detach().cpu().to(torch.float16).numpy())
                    mu_np  = sample_np.astype('float32').mean(0).astype('float16')
                    std_np = sample_np.astype('float32').std(0).astype('float16')

                preds.append({
                    'sample': sample_np,
                    'mean':   mu_np,
                    'std':    std_np,
                })

            # Advance to next chunk
            if len(preds) < horizon:
                if self.use_uncertainty_input:
                    # Propagate (mu, sigma) from last distribution
                    x_t = torch.cat(
                        [last_d.mean, last_d.stddev], dim=-1).detach()  # (1, H, W, 2)
                else:
                    x_t = dist.sample().detach()  # (S, H, W, 1)

        return preds


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    # --- H=1 (single-step, backward compat) ---
    cfg1 = FNOConfig(modes1=8, modes2=8, width=16, n_layers=2, forecast_horizon=1)
    m1 = FNO2d(cfg1)
    x = torch.randn(4, 32, 32, 1)
    dists = m1(x)
    assert len(dists) == 1 and dists[0].mean.shape == (4, 32, 32, 1)
    preds = m1.autoregressive_forecast(x[:1], horizon=5, num_samples=3)
    assert len(preds) == 5 and preds[0]['sample'].shape == (3, 32, 32)
    print(f"H=1 OK — params: {sum(p.numel() for p in m1.parameters()):,}")

    # --- H=3 (multistep) ---
    cfg3 = FNOConfig(modes1=8, modes2=8, width=16, n_layers=2, forecast_horizon=3)
    m3 = FNO2d(cfg3)
    dists3 = m3(x)
    assert len(dists3) == 3
    for d in dists3:
        assert d.mean.shape == (4, 32, 32, 1)
    # horizon=15 → ceil(15/3)=5 forward passes
    preds3 = m3.autoregressive_forecast(x[:1], horizon=15, num_samples=6)
    assert len(preds3) == 15 and preds3[0]['sample'].shape == (6, 32, 32)
    print(f"H=3 OK — params: {sum(p.numel() for p in m3.parameters()):,}")
    print("ALL OK")

