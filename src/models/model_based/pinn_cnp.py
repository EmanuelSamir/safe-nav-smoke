import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from src.models.shared.layers import AttentionAggregator, SoftplusSigma
from src.models.model_based.utils import ObsPINN, PINNOutput
from src.models.shared.fourier_features import ConditionalFourierFeatures


class ContextEncoder(nn.Module):
    """Codifica puntos (x, y, t, s) en una representación determinista C."""
    def __init__(self, input_dim=4, latent_dim=128, 
                 use_fourier_spatial=False, use_fourier_temporal=False,
                 fourier_frequencies=128, fourier_scale=20.0,
                 spatial_max=100.0, temporal_max=10.0):
        super().__init__()
        
        # Fourier encoders
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2, use_fourier=use_fourier_spatial,
            num_frequencies=fourier_frequencies,
            frequency_scale=fourier_scale, input_max=spatial_max)
        
        self.temporal_encoder = ConditionalFourierFeatures(
            input_dim=1, use_fourier=use_fourier_temporal,
            num_frequencies=fourier_frequencies,
            frequency_scale=fourier_scale, input_max=temporal_max)
        
        # Calculate actual input dim
        actual_input_dim = self.spatial_encoder.output_dim + self.temporal_encoder.output_dim + 1
        
        self.point_net = nn.Sequential(
            nn.Linear(actual_input_dim, latent_dim), nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
        self.aggregator = AttentionAggregator(latent_dim, latent_dim)

    def forward(self, obs: ObsPINN) -> torch.Tensor:
        # Encode spatial and temporal separately
        spatial_coords = torch.stack([obs.xs, obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(spatial_coords)
        temporal_features = self.temporal_encoder(obs.ts.unsqueeze(-1))
        
        # Stack context: [spatial_features, temporal_features, s]
        ctx_input = torch.cat([spatial_features, temporal_features, obs.values.unsqueeze(-1)], dim=-1)
        point_embeddings = self.point_net(ctx_input)
        # C: representation vector (B, latent_dim)
        C = self.aggregator(point_embeddings, mask=obs.mask)
        return C

class PINNDecoder(nn.Module):
    """
    Decoder tipo Neural Process con restricciones físicas.
    Input: (x, y, t, C) -> Output: [u, v, p, s, q]
    """
    def __init__(self, context_dim=128, hidden_dim=128, out_mode="full",
                 use_fourier_spatial=False, use_fourier_temporal=False,
                 fourier_frequencies=128, fourier_scale=20.0,
                 spatial_max=100.0, temporal_max=10.0):
        super().__init__()
        self.out_mode = out_mode
        
        # Fourier encoders
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2, use_fourier=use_fourier_spatial,
            num_frequencies=fourier_frequencies,
            frequency_scale=fourier_scale, input_max=spatial_max)
        
        self.temporal_encoder = ConditionalFourierFeatures(
            input_dim=1, use_fourier=use_fourier_temporal,
            num_frequencies=fourier_frequencies,
            frequency_scale=fourier_scale, input_max=temporal_max)
        
        # Input: fourier_features + context
        input_dim = self.spatial_encoder.output_dim + self.temporal_encoder.output_dim + context_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        
        # Cabezales de salida
        self.vel_head = nn.Linear(hidden_dim, 2) # [u, v]
        self.f_head = nn.Linear(hidden_dim, 2) # [f_u, f_v]
        
        if out_mode == "full":
            self.phys_head = nn.Linear(hidden_dim, 2) # [p, q]
        
        # Humo s como distribución Normal
        self.smoke_mu = nn.Linear(hidden_dim, 1)
        self.smoke_std = nn.Sequential(nn.Linear(hidden_dim, 1), SoftplusSigma(min_std=0.01))

    def forward(self, coords: torch.Tensor, C: torch.Tensor) -> PINNOutput:
        # coords: (B, N, 3) -> [x, y, t]
        # C: (B, context_dim)
        B, N, _ = coords.shape
        
        # Encode coordinates with Fourier
        spatial_coords = coords[..., :2]  # (B, N, 2)
        temporal_coords = coords[..., 2:3]  # (B, N, 1)
        spatial_features = self.spatial_encoder(spatial_coords)
        temporal_features = self.temporal_encoder(temporal_coords)
        
        # Expandir contexto para cada punto de consulta
        C_exp = C.unsqueeze(1).expand(-1, N, -1)
        decoder_in = torch.cat([spatial_features, temporal_features, C_exp], dim=-1)
        
        feat = self.net(decoder_in.view(B*N, -1))
        
        # Extraer predicciones
        u_v = self.vel_head(feat).view(B, N, 2)
        u, v = u_v[..., 0:1], u_v[..., 1:2]

        f_uv = self.f_head(feat).view(B, N, 2)
        fu, fv = f_uv[..., 0:1], f_uv[..., 1:2]
        
        p, q = None, None
        if self.out_mode == "full":
            p_q = self.phys_head(feat).view(B, N, 2)
            p, q = p_q[..., 0:1], p_q[..., 1:2]
            
        s_mu = self.smoke_mu(feat).view(B, N, 1)
        s_std = self.smoke_std(feat).view(B, N, 1)
        
        return PINNOutput(
            smoke_dist=Normal(s_mu, s_std),
            u=u, v=v, p=p, q=q, fu=fu, fv=fv
        )

class PINN_CNP(nn.Module):
    """
    Modelo Integrado PINN + Conditional Neural Process.
    Diseñado para descubrir física y predecir campos de humo.
    """
    def __init__(self, latent_dim=128, hidden_dim=128, out_mode="full",
                 use_fourier_spatial=False, use_fourier_temporal=False,
                 fourier_frequencies=128, fourier_scale=20.0,
                 spatial_max=100.0, temporal_max=10.0):
        super().__init__()
        self.encoder = ContextEncoder(
            latent_dim=latent_dim,
            use_fourier_spatial=use_fourier_spatial,
            use_fourier_temporal=use_fourier_temporal,
            fourier_frequencies=fourier_frequencies,
            fourier_scale=fourier_scale,
            spatial_max=spatial_max,
            temporal_max=temporal_max
        )
        self.decoder = PINNDecoder(
            context_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_mode=out_mode,
            use_fourier_spatial=use_fourier_spatial,
            use_fourier_temporal=use_fourier_temporal,
            fourier_frequencies=fourier_frequencies,
            fourier_scale=fourier_scale,
            spatial_max=spatial_max,
            temporal_max=temporal_max
        )

    def forward(self, context: ObsPINN, query: ObsPINN) -> PINNOutput:
        """
        1. Comprime el contexto en un vector C.
        2. Evalúa los puntos de consulta (query) usando el decoder PINN.
        """
        # Obtenemos representación del contexto
        C = self.encoder(context)
        
        # Construimos el tensor de coordenadas de consulta [x, y, t]
        query_coords = torch.stack([query.xs, query.ys, query.ts], dim=-1)
        
        if self.training:
            query_coords = query_coords.detach().requires_grad_(True)
        
        out = self.decoder(query_coords, C)
        out.coords = query_coords
        
        return out
