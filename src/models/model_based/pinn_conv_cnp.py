import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Dict, Tuple

from src.models.shared.layers import SoftplusSigma
from src.models.model_based.utils import ObsPINN, PINNOutput
from src.models.shared.fourier_features import ConditionalFourierFeatures

class RBFSetConv(nn.Module):
    """
    Maps off-grid points (x, y) with features (t, s) to a fixed grid.
    """
    def __init__(self, grid_res=32, grid_range=(-1, 1), sigma=0.1):
        super().__init__()
        self.grid_res = grid_res
        self.grid_min, self.grid_max = grid_range
        self.sigma = sigma
        
        # Precompute grid mesh
        x = torch.linspace(self.grid_min, self.grid_max, grid_res)
        y = torch.linspace(self.grid_min, self.grid_max, grid_res)
        # indexing='ij' -> y varies along dim 0, x along dim 1
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') 
        
        # Shape: (1, 2, H, W) -> flattened to (1, 2, H*W)
        self.register_buffer('grid_points', torch.stack([grid_x, grid_y], dim=0).reshape(1, 2, -1))

    def forward(self, x_c: torch.Tensor, y_c: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:
        """
        x_c: (B, N, 2) spatial coordinates
        y_c: (B, N, 1) smoke values
        t_c: (B, N, 1) time values
        
        Returns: (B, Channels, H, W)
        Channels: [density, mean_s, mean_t, ...]
        """
        B, N, _ = x_c.shape
        H, W = self.grid_res, self.grid_res
        M = H * W
        
        # Grid points: (1, 2, M)
        # x_c: (B, N, 2) -> (B, 2, N)
        x_c_t = x_c.permute(0, 2, 1)
        
        # Simple RBF attention/weighting isn't efficient with pair-wise broadcast for large N*M
        # But for N=300, M=32*32=1024, it's 3e5 interactions per batch. Doable.
        
        # dist^2 = (x - x')^2 + (y - y')^2
        # (B, 2, N, 1) - (1, 2, 1, M)
        diff = x_c_t.unsqueeze(-1) - self.grid_points.unsqueeze(2) # (B, 2, N, M)
        dists_sq = (diff ** 2).sum(dim=1) # (B, N, M)
        
        weights = torch.exp(-0.5 * dists_sq / (self.sigma ** 2)) # (B, N, M)
        
        # Density
        density = weights.sum(dim=1, keepdim=True) # (B, 1, M)
        
        # Features: s (smoke), t (time)
        # values: (B, N, 2)
        feats = torch.cat([y_c, t_c], dim=-1).permute(0, 2, 1) # (B, 2, N)
        
        # Weighted sum of features
        # (B, 2, N) @ (B, N, M) -> (B, 2, M)
        weighted_feats = torch.bmm(feats, weights) 
        
        # Detect where density is near zero to avoid division
        # Simple safe division:
        density_safe = torch.clamp(density, min=1e-5)
        normalized_feats = weighted_feats / density_safe
        
        # Zero out low density areas for cleanliness
        mask = (density > 1e-5).float()
        normalized_feats = normalized_feats * mask
        
        # Concatenate density and features
        # Out: (B, 3, M) -> (B, 3, H, W)
        out = torch.cat([density, normalized_feats], dim=1)
        out = out.reshape(B, 3, H, W)
        
        return out

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, padding=2),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class PINNConvDecoder(nn.Module):
    """
    Decodes from Spatially-interpolated latent grid + Query Time -> Physics
    """
    def __init__(self, latent_dim=64, hidden_dim=128, out_mode="full",
                 use_fourier_temporal=True, temporal_max=3.0):
        super().__init__()
        self.out_mode = out_mode
        
        self.temporal_encoder = ConditionalFourierFeatures(
            input_dim=1, use_fourier=use_fourier_temporal,
            num_frequencies=64, frequency_scale=10.0, input_max=temporal_max)
            
        # Input to MLP: Latent (from grid) + Time Embedding + (Optional X,Y embeddings?)
        # Since Latent is spatially localized, it should contain X,Y info contextually?
        # Explicit X,Y is usually helpful for physics to know boundary conditions etc.
        
        input_dim = latent_dim + self.temporal_encoder.output_dim + 2 # +2 for raw x,y
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        
        self.vel_head = nn.Linear(hidden_dim, 2)
        self.f_head = nn.Linear(hidden_dim, 2)
        if out_mode == "full":
            self.phys_head = nn.Linear(hidden_dim, 2)
            
        self.smoke_mu = nn.Linear(hidden_dim, 1)
        self.smoke_std = nn.Sequential(nn.Linear(hidden_dim, 1), SoftplusSigma(min_std=0.01))

    def _bilinear_sample(self, img, grid):
        """
        Custom bilinear sampling to avoid MPS fallback for grid_sample.
        img: (B, C, H, W) feature map
        grid: (B, N, 1, 2) coordinates in [-1, 1]
        Returns: (B, C, N, 1) sampled values
        """
        B, C, H, W = img.shape
        _, N, _, _ = grid.shape
        
        # Extract x, y from grid (assuming grid has last dim 2: x, y)
        x = grid[..., 0] # (B, N, 1)
        y = grid[..., 1] # (B, N, 1)
        
        # Map [-1, 1] to [0, W-1] and [0, H-1]
        x = ((x + 1) / 2) * (W - 1)
        y = ((y + 1) / 2) * (H - 1)
        
        # Get corner coordinates
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        
        # Clip coordinates to image bounds
        x0 = torch.clamp(x0, 0, W - 1)
        x1 = torch.clamp(x1, 0, W - 1)
        y0 = torch.clamp(y0, 0, H - 1)
        y1 = torch.clamp(y1, 0, H - 1)
        
        # Get interpolation weights
        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x1.float() - x) * (y - y0.float())
        wc = (x - x0.float()) * (y1.float() - y)
        wd = (x - x0.float()) * (y - y0.float())
        
        # Gather pixel values
        # We need to use gather or advanced indexing. 
        # img is (B, C, H, W). We want key (b, c, y, x).
        # Flattening H, W is easier.
        img_flat = img.view(B, C, H * W)
        
        # Helper to compute flattened indices
        def get_vals(x_coord, y_coord):
            # x_coord: (B, N, 1), y_coord: (B, N, 1)
            indices = y_coord * W + x_coord 
            # Expand indices for channels: (B, C, N, 1) ??
            # img_flat: (B, C, HW). gather expects index same dim?
            # Let's align dims. 
            # We want output (B, C, N*1).
            # indices is (B, N*1).
            idx_expanded = indices.view(B, 1, N).expand(-1, C, -1)
            vals = torch.gather(img_flat, 2, idx_expanded)
            return vals.view(B, C, N, 1)
            
        Ia = get_vals(x0, y0)
        Ib = get_vals(x0, y1)
        Ic = get_vals(x1, y0)
        Id = get_vals(x1, y1)
        
        # Interpolate
        # Weights (B, N, 1) -> (B, 1, N, 1)
        wa = wa.permute(0, 2, 1).unsqueeze(-1)
        wb = wb.permute(0, 2, 1).unsqueeze(-1)
        wc = wc.permute(0, 2, 1).unsqueeze(-1)
        wd = wd.permute(0, 2, 1).unsqueeze(-1)
        
        # (B, C, N, 1)
        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def forward(self, coords: torch.Tensor, z_grid: torch.Tensor) -> PINNOutput:
        """
        coords: (B, N, 3) -> (x, y, t)
        z_grid: (B, C, H, W)
        """
        B, N, _ = coords.shape
        
        # 1. Sample Grid at query (x, y)
        # grid_sample expects coordinates in [-1, 1]
        # Our coords should be normalized to [-1, 1] before calling model? 
        # Dataset assumes [0, 50] meters. Neural Process usually normalizes.
        # We will assume coords are pre-normalized or we normalize here.
        # Let's assume input is in metric space and we map to [-1, 1]
        # BUT for RBFSetConv to work, x_c and grid must match.
        # Assuming we normalize inputs to [-1, 1] outside or consistently.
        
        xy = coords[..., :2] # (B, N, 2)
        t = coords[..., 2:3] # (B, N, 1)
        
        # (B, N, 1, 2) for grid_sample
        # grid_sample expects (x, y). 
        # CAUTION: Check coordinate systems. 
        # If we use x in range [-1, 1], we are good.
        grid_sample_coords = xy.unsqueeze(2) 
        
        # Sample: (B, C, N, 1) -> (B, N, C)
        # Using custom bilinear sample to avoid MPS fallback
        # z_local = F.grid_sample(z_grid, grid_sample_coords, align_corners=True).squeeze(-1).permute(0, 2, 1)
        z_local = self._bilinear_sample(z_grid, grid_sample_coords).squeeze(-1).permute(0, 2, 1)
        
        # 2. Time Encoding
        t_embed = self.temporal_encoder(t)
        
        # 3. Concatenate
        # Vector: [z_local, t_embed, raw_xy]
        inp = torch.cat([z_local, t_embed, xy], dim=-1)
        
        feat = self.net(inp)
        
        # Heads
        u_v = self.vel_head(feat)
        u, v = u_v[..., 0:1], u_v[..., 1:2]
        
        f_uv = self.f_head(feat)
        fu, fv = f_uv[..., 0:1], f_uv[..., 1:2]
        
        p, q = None, None
        if self.out_mode == "full":
            p_q = self.phys_head(feat)
            p, q = p_q[..., 0:1], p_q[..., 1:2]
            
        s_mu = self.smoke_mu(feat)
        s_std = self.smoke_std(feat)
        
        return PINNOutput(
            smoke_dist=Normal(s_mu, s_std),
            u=u, v=v, p=p, q=q, fu=fu, fv=fv
        )

class PINN_Conv_CNP(nn.Module):
    def __init__(self, 
                 grid_res=32, 
                 hidden_dim=64, 
                 out_mode="full",
                 spatial_min=0.0, spatial_max=100.0, # Metric bounds
                 temporal_max=10.0):
        super().__init__()
        
        self.spatial_min = spatial_min
        self.spatial_max = spatial_max
        self.range_len = spatial_max - spatial_min
        
        # Components
        self.set_conv = RBFSetConv(grid_res=grid_res, grid_range=(-1, 1), sigma=0.1) # Operates in normalized [-1, 1]
        
        self.cnn = CNNEncoder(in_channels=3, hidden=hidden_dim)
        
        self.decoder = PINNConvDecoder(
            latent_dim=hidden_dim, 
            hidden_dim=hidden_dim,
            out_mode=out_mode,
            temporal_max=temporal_max
        )

    def _normalize(self, x):
        """Map [min, max] to [-1, 1]"""
        return 2.0 * (x - self.spatial_min) / self.range_len - 1.0

    def forward(self, context: ObsPINN, query: ObsPINN) -> PINNOutput:
        # 1. Normalize Context Spatial Coords to [-1, 1]
        ctx_xy = torch.stack([context.xs, context.ys], dim=-1)
        ctx_xy_norm = self._normalize(ctx_xy)
        
        # 2. SetConv -> Grid (B, 3, H, W)
        grid_feats = self.set_conv(ctx_xy_norm, context.values.unsqueeze(-1), context.ts.unsqueeze(-1))
        
        # 3. CNN -> Latent Grid (B, C, H, W)
        z_grid = self.cnn(grid_feats)
        
        # 4. Normalize Query Spatial Coords
        # We need gradients for query coords for physics loss
        # Use query.xs directly if possible, but we need to normalize for grid sampling
        # Normalization is linear, so gradients flow fine.
        
        # Construct query tensor (B, N, 3)
        query_tensor = torch.stack([query.xs, query.ys, query.ts], dim=-1)
        
        if self.training:
             query_tensor.requires_grad_(True)
             
        q_xy = query_tensor[..., :2]
        
        # Use simple manual normalization to preserve gradients
        q_xy_norm = 2.0 * (q_xy - self.spatial_min) / self.range_len - 1.0
        
        # Reconstruct normalized query tensor for decoder structure
        # (Pass normalized XY for sampling, but maybe Raw XY/T for MLP embeddings?)
        # Decoder uses `coords` arg. Let's pass (NormX, NormY, T) via a custom struct or just tensor?
        # Decoder logic:
        # xy = coords[..., :2] -> used for grid_sample and embedding
        # t = coords[..., 2:3]
        
        # We pass [NormX, NormY, T] to decoder
        decoder_input = torch.cat([q_xy_norm, query_tensor[..., 2:3]], dim=-1)
        
        out = self.decoder(decoder_input, z_grid)
        
        # Attach Original Coords (with requires_grad) for Physics Loss computation
        # The physics loss needs d(output)/d(x_metric).
        # Our output depends on decoder_input which depends on query_tensor (metric).
        # So d(out)/d(query_tensor) should be correct via chain rule.
        out.coords = query_tensor # This is the metric coordinate tensor
        
        return out
