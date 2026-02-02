import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Dict

from src.models.shared.layers import SoftplusSigma
from src.models.model_based.utils import ObsPINN, PINNOutput

class RBFSetConv(nn.Module):
    """
    Simpler RBF Discretization using Gaussian kernels.
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

    def forward(self, x_c, y_c, t_c, mask: Optional[torch.Tensor] = None):
        """
        x_c: (B, N, 2) normalized spatial
        y_c: (B, N, 1) smoke
        t_c: (B, N, 1) time
        """
        # Distances: (B, 2, N, 1) - (1, 2, 1, M)
        diff = x_c.permute(0, 2, 1).unsqueeze(-1) - self.grid_points.unsqueeze(2)
        dists_sq = (diff ** 2).sum(dim=1) # (B, N, M)
        
        # Weights (Gaussian kernel)
        weights = torch.exp(-0.5 * dists_sq / (self.sigma ** 2))

        # Apply mask if provided (True = Padding)
        if mask is not None:
             # mask: (B, N) -> (B, N, 1) to broadcast over grid dimension M
             mask_expanded = mask.unsqueeze(-1)
             weights = weights.masked_fill(mask_expanded, 0.0)
        
        # Density (sum of weights per grid point)
        density = weights.sum(dim=1, keepdim=True) # (B, 1, M)
        
        # Features: [smoke, time]
        features = torch.cat([y_c, t_c], dim=-1).permute(0, 2, 1) # (B, 2, N)
        
        # Weighted sum: (B, 2, N) @ (B, N, M) -> (B, 2, M)
        weighted_sum = torch.bmm(features, weights)
        
        # Normalize by density (Safe division)
        # Add basic feature channels
        out = torch.cat([density, weighted_sum / (density + 1e-5)], dim=1) # (B, 3, M)
        
        # Reshape to grid
        return out.reshape(out.shape[0], 3, self.grid_res, self.grid_res)

class ConvDeepSet(nn.Module):
    """Simple 5-layer CNN with residual connections"""
    def __init__(self, in_channels=3, hidden=64, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, latent_dim, 5, padding=2), # Final layer to latent dim
            # No Tanh on final latent? Original had Tanh. User block had Tanh.
            # Usually latent should be unbounded or normalized. Tanh bounds it [-1, 1].
            # User block: nn.Conv2d(..., latent_dim, ...), nn.Tanh()
            # I will keep Tanh if user had it, but usually BatchNorm before activation.
            # Logic: Conv -> BN -> Act.
            # Last layer: Conv -> BN -> Tanh (if desired).
            nn.BatchNorm2d(latent_dim), nn.Tanh() 
        )
    def forward(self, x):
        return self.net(x)

from src.models.shared.fourier_features import ConditionalFourierFeatures

class PINN_Conv_CNP(nn.Module):
    """
    Simplified ConvCNP for CUDA usage.
    Structure:
    1. SetConv: Discretize context (x,y,t,s) -> Grid
    2. ConvDeepSet: Process Grid -> Latent Grid
    3. Decoder: Continuous Query (x,y,t) + Latent Grid -> Physics Output
    """
    def __init__(self, 
                 grid_res=32, 
                 hidden_dim=64, 
                 out_mode="full",
                 spatial_min=0.0, spatial_max=100.0,
                 temporal_max=10.0,
                 latent_dim=16,
                 use_fourier_features=True):
        super().__init__()
        
        self.spatial_min = spatial_min
        self.spatial_max = spatial_max
        self.len = spatial_max - spatial_min
        self.use_fourier = use_fourier_features
        
        # 1. Discretization
        self.set_conv = RBFSetConv(grid_res, sigma=0.1)
        
        # 2. Processor
        self.processor = ConvDeepSet(in_channels=3, hidden=hidden_dim, latent_dim=latent_dim)
        
        # 3. Time Encoding (Optional Fourier)
        if self.use_fourier:
            self.time_encoder = ConditionalFourierFeatures(
                input_dim=1, use_fourier=True, 
                num_frequencies=64, frequency_scale=10.0, input_max=temporal_max
            )
            t_dim = self.time_encoder.output_dim
        else:
            self.time_encoder = None
            t_dim = 1
            
        # 3. Decoder MLP
        # Input: Latent (from grid) + Time (Encoded/Raw) + Raw X,Y
        input_dim = latent_dim + t_dim + 2 
        
        self.decoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh()
        )
        
        # Heads
        self.head_u = nn.Linear(hidden_dim, 2)
        self.head_f = nn.Linear(hidden_dim, 2)
        self.head_s_mu = nn.Linear(hidden_dim, 1)
        self.head_s_std = nn.Sequential(nn.Linear(hidden_dim, 1), SoftplusSigma())
        
        self.out_mode = out_mode
        if out_mode == "full":
            self.head_p = nn.Linear(hidden_dim, 2)

    def forward(self, context: ObsPINN, query: ObsPINN) -> PINNOutput:
        # 1. Normalize Context spatial
        ctx_xy = (torch.stack([context.xs, context.ys], -1) - self.spatial_min) / self.len 
        ctx_xy = 2.0 * ctx_xy - 1.0 # [-1, 1]
        
        # 2. Grid Encoding
        grid = self.set_conv(ctx_xy, context.values.unsqueeze(-1), context.ts.unsqueeze(-1), mask=context.mask)
        
        # 3. CNN Processing
        latent_grid = self.processor(grid) # (B, H, Res, Res)
        
        # 4. Decoder Query
        # Normalize query spatial for interpolation
        q_raw = torch.stack([query.xs, query.ys, query.ts], -1)
        if self.training: q_raw.requires_grad_(True)
        
        q_xy = q_raw[..., :2]
        q_xy_norm = 2.0 * ((q_xy - self.spatial_min) / self.len) - 1.0
        
        # Interpolate from grid
        # grid_sample expects (B, H, W, 2) in (x, y) order
        sample_coords = q_xy_norm.unsqueeze(2) 
        
        # (B, C, N, 1)
        latents = F.grid_sample(latent_grid, sample_coords, align_corners=True)
        latents = latents.squeeze(-1).permute(0, 2, 1) # (B, N, C)
        
        # Time processing
        if self.use_fourier:
            t_embed = self.time_encoder(query.ts.unsqueeze(-1))
        else:
            t_embed = query.ts.unsqueeze(-1)
        print("- latent mean:", latents.mean().item())
        print("- latent std:", latents.std().item())
        print("- latent max:", latents.max().item())
        print("- latent min:", latents.min().item())
            
        # Concatenate: [Latent, TimeEmbed, RawXY]
        decoder_in = torch.cat([latents, t_embed, q_xy], dim=-1)
        
        # Predict
        feat = self.decoder_net(decoder_in)
        
        u_v = self.head_u(feat)
        f_uv = self.head_f(feat)
        s_mu = self.head_s_mu(feat)
        s_std = self.head_s_std(feat)
        
        return PINNOutput(
            smoke_dist=Normal(s_mu, s_std),
            u=u_v[..., 0:1], v=u_v[..., 1:2],
            fu=f_uv[..., 0:1], fv=f_uv[..., 1:2],
            coords=q_raw # Pass gradients back
        )
