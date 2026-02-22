import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from torch.distributions import Normal
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from src.models.shared.layers import AttentionAggregator, MeanAggregator, SoftplusSigma, FourierFeatureEncoder
from src.models.model_based.utils import PINNOutput
from src.models.shared.observations import Obs


class ContextEncoder(nn.Module):
    """Codifica puntos (x, y, t, s) en una representación determinista C."""
    def __init__(self, input_dim=4, latent_dim=128, hidden_dim=128,
                 fourier_mapping_size=32, 
                 aggregator_type="attention"):
        super().__init__()

        self.point_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        if fourier_mapping_size and fourier_mapping_size > 0:
            self.fourier_encoder = FourierFeatureEncoder(input_dim=hidden_dim, mapping_size=fourier_mapping_size)
            self.out_proj = nn.Linear(2*fourier_mapping_size, latent_dim)
        else:
            self.fourier_encoder = None
            self.out_proj = nn.Linear(hidden_dim, latent_dim)

        if aggregator_type == "mean":
            self.aggregator = MeanAggregator(latent_dim)
        else:
            self.aggregator = AttentionAggregator(latent_dim, latent_dim)

    def forward(self, obs: Obs) -> torch.Tensor:
        # Encode spatial and temporal separately
        coords = torch.cat([obs.xs, obs.ys, obs.ts, obs.values], dim=-1)
        point_embeddings = self.point_net(coords)
        
        if self.fourier_encoder is not None:
            point_embeddings = self.fourier_encoder(point_embeddings)

        point_embeddings = self.out_proj(point_embeddings)

        mask = obs.mask
        if mask is not None and mask.dim() == 3:
             mask = mask.squeeze(-1)
        C = self.aggregator(point_embeddings, mask=mask)
        return C

class PINNDecoder(nn.Module):
    """
    Decoder tipo Neural Process con restricciones físicas.
    Input: (x, y, t, C) -> Output: [u, v, p, s, q]
    """
    def __init__(self, context_dim=128, hidden_dim=128, fourier_mapping_size=None):
        super().__init__()

        if fourier_mapping_size and fourier_mapping_size > 0:
            self.fourier_encoder = FourierFeatureEncoder(input_dim=3, mapping_size=fourier_mapping_size)
            input_dim = 2*fourier_mapping_size
        else:
            self.fourier_encoder = None
            input_dim = 3
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + context_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        
        # Cabezales de salida
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        ) # [u, v]

        self.f_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        ) # [f_u, f_v]
        
        # Humo s como distribución Normal
        self.smoke_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.smoke_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            SoftplusSigma(min_std=0.01)
        )

    def forward(self, coords: torch.Tensor, C: torch.Tensor) -> PINNOutput:
        # coords: (B, N, 3) -> [x, y, t]
        # C: (B, context_dim)
        B, N, _ = coords.shape
        
        # Encode coordinates with Fourier
        if self.fourier_encoder is not None:
            coords = self.fourier_encoder(coords)
        
        # Expandir contexto para cada punto de consulta
        C_exp = C.unsqueeze(1).expand(-1, N, -1)
        decoder_in = torch.cat([coords, C_exp], dim=-1)
        
        feat = self.backbone(decoder_in)
        
        u_v = self.vel_head(feat)
        u_v = u_v.view(B, N, 2)
        u, v = u_v[..., 0:1], u_v[..., 1:2]

        f_uv = self.f_head(feat).view(B, N, 2)
        fu, fv = f_uv[..., 0:1], f_uv[..., 1:2]
            
        s_mu = self.smoke_mu(feat)
        s_mu = s_mu.view(B, N, 1)
        s_std = self.smoke_std(feat)
        s_std = s_std.view(B, N, 1)
        
        return PINNOutput(
            smoke_dist=Normal(s_mu, s_std),
            u=u, v=v, fu=fu, fv=fv
        )

class PINN_CNP(nn.Module):
    """
    PINN + CNP
    """
    def __init__(self, latent_dim=128, hidden_dim=128,
                 fourier_mapping_size=None,
                 aggregator_type="attention"):
        super().__init__()
        
        self.encoder = ContextEncoder(
            latent_dim=latent_dim,
            #fourier_mapping_size=fourier_mapping_size,
            hidden_dim=hidden_dim,
            aggregator_type=aggregator_type
        )
        self.decoder = PINNDecoder(
            context_dim=latent_dim,
            hidden_dim=hidden_dim,
            fourier_mapping_size=fourier_mapping_size
        )

    def forward(self, context: Obs, query: Obs) -> PINNOutput:
        C = self.encoder(context)
        
        if query.ts is None:
             raise ValueError("Time coordinates (ts) required for PINN query")

        B, _ = C.shape

        # std_C_in_B = C.std(dim=0)
        # if B > 1:
        #     print("C", C)
        #     print("std_C_in_B: ", std_C_in_B)
        
        query_coords = torch.cat([query.xs, query.ys, query.ts], dim=-1)
        
        # if self.training:
        #     query_coords = query_coords.detach().requires_grad_(True)
        
        out = self.decoder(query_coords, C)
        out.coords = query_coords
        
        return out
