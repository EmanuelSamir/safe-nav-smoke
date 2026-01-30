import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from dataclasses import dataclass
from typing import Optional, Tuple

from src.models.model_based.utils import ObsPINN, PINNOutput
from src.models.shared.layers import AttentionAggregator, SoftplusSigma
from src.models.shared.fourier_features import ConditionalFourierFeatures

@dataclass
class LNPOutput(PINNOutput):
    """Extiende PINNOutput para incluir la distribución latente (para el Loss ELBO)."""
    latent_mu: Optional[torch.Tensor] = None
    latent_sigma: Optional[torch.Tensor] = None

class StochasticEncoder(nn.Module):
    """
    Codifica el contexto en una distribución normal multivariante.
    s_context -> q(z | s_context) = N(mu_z, sigma_z)
    """
    def __init__(self, input_dim=4, latent_dim=128, 
                 use_fourier_spatial=False, use_fourier_temporal=False,
                 fourier_frequencies=128, fourier_scale=20.0,
                 spatial_max=100.0, temporal_max=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        
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
        
        # Point-wise MLP
        self.point_net = nn.Sequential(
            nn.Linear(actual_input_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )
        
        self.aggregator = AttentionAggregator(128, 128)
        
        # Cabezales para la distribución latente
        self.mu_head = nn.Linear(128, latent_dim)
        self.sigma_head = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.Sigmoid() # Usamos sigmoid para escala controlada (0-1)
        )

    def forward(self, obs: ObsPINN) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode spatial and temporal separately
        spatial_coords = torch.stack([obs.xs, obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(spatial_coords)
        temporal_features = self.temporal_encoder(obs.ts.unsqueeze(-1))
        
        # Stack context: [spatial_features, temporal_features, values]
        ctx_input = torch.cat([spatial_features, temporal_features, obs.values.unsqueeze(-1)], dim=-1)
        
        h = self.point_net(ctx_input)
        r = self.aggregator(h, mask=obs.mask)
        
        mu = self.mu_head(r)
        # Sigma con un pequeño offset para estabilidad numérica
        sigma = 0.1 + 0.9 * self.sigma_head(r) 
        
        return mu, sigma

class PINN_LNP(nn.Module):
    """
    Latent Neural Process con Restricciones Físicas (PINN).
    El contexto define una distribución de "mundos posibles" z.
    """
    def __init__(self, latent_dim=128, hidden_dim=256, out_mode="full",
                 use_fourier_spatial=False, use_fourier_temporal=False,
                 fourier_frequencies=128, fourier_scale=20.0,
                 spatial_max=100.0, temporal_max=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 1. Stochastic Encoder
        self.encoder = StochasticEncoder(
            latent_dim=latent_dim,
            use_fourier_spatial=use_fourier_spatial,
            use_fourier_temporal=use_fourier_temporal,
            fourier_frequencies=fourier_frequencies,
            fourier_scale=fourier_scale,
            spatial_max=spatial_max,
            temporal_max=temporal_max
        )
        
        # Fourier encoders also needed for Decoder 
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2, use_fourier=use_fourier_spatial,
            num_frequencies=fourier_frequencies,
            frequency_scale=fourier_scale, input_max=spatial_max)
        
        self.temporal_encoder = ConditionalFourierFeatures(
            input_dim=1, use_fourier=use_fourier_temporal,
            num_frequencies=fourier_frequencies,
            frequency_scale=fourier_scale, input_max=temporal_max)
            
        decoder_input_dim = self.spatial_encoder.output_dim + self.temporal_encoder.output_dim + latent_dim
        
        # 2. PINN Decoder (Conditioned on sampled z)
        # Usamos Tanh para suavidad en las derivadas PINN
        self.decoder_net = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        
        self.vel_head = nn.Linear(hidden_dim, 2)
        self.f_head = nn.Linear(hidden_dim, 2)
        self.smoke_mu = nn.Linear(hidden_dim, 1)
        self.smoke_std = nn.Sequential(nn.Linear(hidden_dim, 1), SoftplusSigma(min_std=0.01))
        
        self.out_mode = out_mode
        if out_mode == "full":
            self.phys_head = nn.Linear(hidden_dim, 2) # [p, q]

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Muestreo estocástico z = mu + sigma * epsilon."""
        if self.training:
            std = sigma
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu # En evaluación usamos la media

    def encode(self, obs: ObsPINN) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(obs)

    def forward(self, context: ObsPINN, query: ObsPINN, z: Optional[torch.Tensor] = None) -> LNPOutput:
        # 1. Obtener parámetros de la distribución latente (Prior)
        mu_z, sigma_z = self.encode(context)
        
        # 2. Muestrear z si no se provee uno externo
        if z is None:
            z = self.reparameterize(mu_z, sigma_z)
        
        # 3. Preparar entrada para el decoder [coords, z]
        query_coords = torch.stack([query.xs, query.ys, query.ts], dim=-1)
        if self.training:
            query_coords = query_coords.detach().requires_grad_(True)
            
        B, N, _ = query_coords.shape
        z_exp = z.unsqueeze(1).expand(-1, N, -1)
        
        # Encode coordinates with Fourier
        spatial_coords = query_coords[..., :2]
        temporal_coords = query_coords[..., 2:3]
        spatial_features = self.spatial_encoder(spatial_coords)
        temporal_features = self.temporal_encoder(temporal_coords)
        
        decoder_in = torch.cat([spatial_features, temporal_features, z_exp], dim=-1).view(B*N, -1)
        feat = self.decoder_net(decoder_in)
        
        # 4. Generar campos
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
        
        return LNPOutput(
            smoke_dist=Normal(s_mu, s_std),
            u=u, v=v, p=p, q=q, fu=fu, fv=fv,
            coords=query_coords,
            latent_mu=mu_z,
            latent_sigma=sigma_z
        )
