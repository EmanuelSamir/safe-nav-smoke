import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Optional, Tuple

from src.models.model_based.utils import ObsPINN, PINNOutput
from src.models.model_based.pinn_lnp import LNPOutput
from src.models.model_based.pinn_cnp_resnet import ResidualBlock, ResidualPINNDecoder
from src.models.shared.layers import AttentionAggregator, SoftplusSigma
from src.models.shared.fourier_features import ConditionalFourierFeatures

class StochasticResidualEncoder(nn.Module):
    """
    Encoder estocástico con bloques residuales.
    s_context -> q(z | s_context) = N(mu_z, sigma_z)
    """
    def __init__(self, input_dim=4, latent_dim=128, num_blocks=2,
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
        
        self.input_layer = nn.Linear(actual_input_dim, 128)
        self.blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(num_blocks)
        ])
        
        self.aggregator = AttentionAggregator(128, 128)
        
        # Cabezales estocásticos
        self.mu_head = nn.Linear(128, latent_dim)
        self.sigma_head = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, obs: ObsPINN) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode spatial and temporal separately
        spatial_coords = torch.stack([obs.xs, obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(spatial_coords)
        temporal_features = self.temporal_encoder(obs.ts.unsqueeze(-1))
        
        # Stack context
        ctx_input = torch.cat([spatial_features, temporal_features, obs.values.unsqueeze(-1)], dim=-1)
        
        x = F.tanh(self.input_layer(ctx_input))
        for block in self.blocks:
            x = block(x)
            
        r = self.aggregator(x, mask=obs.mask)
        
        mu = self.mu_head(r)
        sigma = 0.1 + 0.9 * self.sigma_head(r)
        
        return mu, sigma

class PINN_LNP_ResNet(nn.Module):
    """
    Latent Neural Process con Skip Connections e Inyección de Coordenadas.
    Combina la naturaleza estocástica del LNP con la estabilidad de ResNet.
    """
    def __init__(self, latent_dim=128, hidden_dim=256, num_blocks=3, out_mode="full",
                 use_fourier_spatial=False, use_fourier_temporal=False,
                 fourier_frequencies=128, fourier_scale=20.0,
                 spatial_max=100.0, temporal_max=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = StochasticResidualEncoder(
            latent_dim=latent_dim, num_blocks=2,
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

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(sigma) * sigma
        else:
            return mu

    def encode(self, obs: ObsPINN) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(obs)

    def forward(self, context: ObsPINN, query: ObsPINN, z: Optional[torch.Tensor] = None) -> LNPOutput:
        # 1. Encoding (Prior)
        mu_z, sigma_z = self.encode(context)
        
        # 2. Latent Sample
        if z is None:
            z = self.reparameterize(mu_z, sigma_z)
            
        # 3. Decoding using ResNet with coord injection
        query_coords = torch.stack([query.xs, query.ys, query.ts], dim=-1)
        if self.training:
            query_coords = query_coords.detach().requires_grad_(True)
            
        # El ResidualPINNDecoder de pinn_cnp_resnet ya maneja la expansión de z e inyección
        out = self.decoder(query_coords, z)
        
        return LNPOutput(
            smoke_dist=out.smoke_dist,
            u=out.u, v=out.v, p=out.p, q=out.q, fu=out.fu, fv=out.fv,
            coords=query_coords,
            latent_mu=mu_z,
            latent_sigma=sigma_z
        )
