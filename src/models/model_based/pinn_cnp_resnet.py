import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Optional, List

from src.models.model_based.utils import ObsPINN, PINNOutput
from src.models.shared.layers import AttentionAggregator, SoftplusSigma
from src.models.shared.fourier_features import ConditionalFourierFeatures

class ResidualBlock(nn.Module):
    """Bloque residual estándar para MLPs."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(), # Tanh es preferible para derivados PINN continuos
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        return x + self.net(x)

class ResidualPointNet(nn.Module):
    """Encoder de contexto con conexiones residenciales."""
    def __init__(self, input_dim=4, latent_dim=128, num_blocks=2,
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
            
        actual_input_dim = self.spatial_encoder.output_dim + self.temporal_encoder.output_dim + 1
        
        self.input_layer = nn.Linear(actual_input_dim, latent_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(latent_dim) for _ in range(num_blocks)
        ])
        self.aggregator = AttentionAggregator(latent_dim, latent_dim)

    def forward(self, obs: ObsPINN) -> torch.Tensor:
        # Encode spatial and temporal separately
        spatial_coords = torch.stack([obs.xs, obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(spatial_coords)
        temporal_features = self.temporal_encoder(obs.ts.unsqueeze(-1))
        
        # Stack context
        ctx_input = torch.cat([spatial_features, temporal_features, obs.values.unsqueeze(-1)], dim=-1)
        
        x = F.tanh(self.input_layer(ctx_input))
        for block in self.blocks:
            x = block(x)
            
        C = self.aggregator(x, mask=obs.mask)
        return C

class ResidualPINNDecoder(nn.Module):
    """
    Decoder con Inyección de Coordenadas y Capas Residuales.
    Implementa skip connections que re-inyectan (x, y, t, C) en cada etapa.
    """
    def __init__(self, context_dim=128, hidden_dim=128, num_blocks=3, out_mode="full",
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
            
        self.input_dim = self.spatial_encoder.output_dim + self.temporal_encoder.output_dim + context_dim
        
        # Proyectamos la entrada inicial
        self.first_layer = nn.Linear(self.input_dim, hidden_dim)
        
        # Bloques residuales con inyección de la entrada original
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'res': ResidualBlock(hidden_dim),
                'skip': nn.Linear(self.input_dim, hidden_dim) # Para re-inyectar coords+C
            }))
            
        # Cabezales de salida
        self.vel_head = nn.Linear(hidden_dim, 2)
        self.f_head = nn.Linear(hidden_dim, 2)
        if out_mode == "full":
            self.phys_head = nn.Linear(hidden_dim, 2)
        
        self.smoke_mu = nn.Linear(hidden_dim, 1)
        self.smoke_std = nn.Sequential(nn.Linear(hidden_dim, 1), SoftplusSigma(min_std=0.01))

    def forward(self, coords: torch.Tensor, C: torch.Tensor) -> PINNOutput:
        B, N, _ = coords.shape
        C_exp = C.unsqueeze(1).expand(-1, N, -1)
        
        # Encode coordinates with Fourier
        spatial_coords = coords[..., :2]
        temporal_coords = coords[..., 2:3]
        spatial_features = self.spatial_encoder(spatial_coords)
        temporal_features = self.temporal_encoder(temporal_coords)
        
        # Input base: (fourier_x, fourier_t, C)
        raw_input = torch.cat([spatial_features, temporal_features, C_exp], dim=-1).view(B*N, -1)
        
        # Capa inicial
        h = F.tanh(self.first_layer(raw_input))
        
        # Procesamiento residual con inyección
        for block in self.blocks:
            # H_next = h + residual(h) + linear(raw_input)
            h = block['res'](h) + block['skip'](raw_input)
            
        # Salidas
        u_v = self.vel_head(h).view(B, N, 2)
        u, v = u_v[..., 0:1], u_v[..., 1:2]

        f_uv = self.f_head(h).view(B, N, 2)
        fu, fv = f_uv[..., 0:1], f_uv[..., 1:2]
        
        p, q = None, None
        if self.out_mode == "full":
            p_q = self.phys_head(h).view(B, N, 2)
            p, q = p_q[..., 0:1], p_q[..., 1:2]
            
        s_mu = self.smoke_mu(h).view(B, N, 1)
        s_std = self.smoke_std(h).view(B, N, 1)
        
        # Guardamos coords para el cálculo de gradientes en el Loss
        return PINNOutput(
            smoke_dist=Normal(s_mu, s_std),
            u=u, v=v, p=p, q=q, coords=coords, fu=fu, fv=fv
        )

class PINN_CNP_ResNet(nn.Module):
    """
    Versión Avanzada de PINN-CNP con Conexiones Residuales e Inyección de Coordenadas.
    """
    def __init__(self, latent_dim=128, hidden_dim=256, num_blocks=3, out_mode="full",
                 use_fourier_spatial=False, use_fourier_temporal=False,
                 fourier_frequencies=128, fourier_scale=20.0,
                 spatial_max=100.0, temporal_max=10.0):
        super().__init__()
        self.encoder = ResidualPointNet(
            latent_dim=latent_dim, num_blocks=num_blocks,
            use_fourier_spatial=use_fourier_spatial,
            use_fourier_temporal=use_fourier_temporal,
            fourier_frequencies=fourier_frequencies,
            fourier_scale=fourier_scale,
            spatial_max=spatial_max,
            temporal_max=temporal_max
        )
        self.decoder = ResidualPINNDecoder(
            context_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            num_blocks=num_blocks, 
            out_mode=out_mode,
            use_fourier_spatial=use_fourier_spatial,
            use_fourier_temporal=use_fourier_temporal,
            fourier_frequencies=fourier_frequencies,
            fourier_scale=fourier_scale,
            spatial_max=spatial_max,
            temporal_max=temporal_max
        )

    def forward(self, context: ObsPINN, query: ObsPINN) -> PINNOutput:
        C = self.encoder(context)
        query_coords = torch.stack([query.xs, query.ys, query.ts], dim=-1)
        
        if self.training:
            query_coords = query_coords.detach().requires_grad_(True)
            
        return self.decoder(query_coords, C)
