"""
SNP - Stochastic Neural Process
Clean implementation for scalar field prediction with temporal dynamics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
from torch.distributions import Normal

from src.models.shared.observations import Obs
from src.models.shared.outputs import SNPOutput
from src.models.shared.fourier_features import ConditionalFourierFeatures


@dataclass
class SNPConfig:
    """Configuration for SNP model."""
    # Dimensions
    z_dim: int = 128        # Latent stochastic dimension
    h_dim: int = 128        # Hidden deterministic dimension (LSTM)
    r_dim: int = 128        # Representation/embedding dimension
    
    # Network sizes
    encoder_hidden: int = 128
    decoder_hidden: int = 128
    prior_hidden: int = 128
    posterior_hidden: int = 128
    
    # Model options
    use_actions: bool = True
    action_dim: int = 2
    aggregator: str = "mean"  # "mean" or "attention"
    min_std: float = 0.01
    
    # Fourier Features (for high-frequency learning)
    use_fourier_encoder: bool = False
    use_fourier_decoder: bool = False
    fourier_frequencies: int = 128
    fourier_scale: float = 20.0
    spatial_max: float = 100.0  # For normalization


def init_weights(m):
    """Xavier initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Encoder(nn.Module):
    """Encodes observations into representation r."""
    def __init__(self, config: SNPConfig):
        super().__init__()
        self.config = config
        
        # Optional Fourier encoding for spatial coordinates
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_encoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        # Point-wise MLP: (fourier_features, value) -> embedding
        input_dim = self.spatial_encoder.output_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.encoder_hidden),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden, config.r_dim),
            nn.ReLU()
        )
    
    def forward(self, obs: Obs) -> torch.Tensor:
        """
        Args:
            obs: Observation with xs, ys, values, mask
        
        Returns:
            r: Representation (B, r_dim)
        """
        # Encode spatial coordinates
        coords = torch.stack([obs.xs, obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(coords)
        
        # Concatenate with values
        values = obs.values if obs.values.dim() == 2 else obs.values.squeeze(-1)
        inputs = torch.cat([spatial_features, values.unsqueeze(-1)], dim=-1)
        
        # Encode each point
        embeddings = self.net(inputs)
        
        # Aggregate (mean pooling with mask)
        if obs.mask is not None:
            mask_expanded = obs.mask.unsqueeze(-1).float()
            sum_emb = (embeddings * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            r = sum_emb / count
        else:
            r = embeddings.mean(dim=1)
        
        return r


class PriorNet(nn.Module):
    """Prior distribution p(z_t | h_t)."""
    def __init__(self, h_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.sigma = nn.Linear(hidden, z_dim)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(h)
        mu = self.mu(feat)
        sigma = F.softplus(self.sigma(feat)) + 1e-4
        return mu, sigma


class PosteriorNet(nn.Module):
    """Posterior distribution q(z_t | h_t, r_target)."""
    def __init__(self, h_dim: int, r_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + r_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.sigma = nn.Linear(hidden, z_dim)
    
    def forward(self, h: torch.Tensor, r_tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(torch.cat([h, r_tgt], dim=-1))
        mu = self.mu(feat)
        sigma = F.softplus(self.sigma(feat)) + 1e-4
        return mu, sigma


class Decoder(nn.Module):
    """Decodes latent z and query locations into predictions."""
    def __init__(self, config: SNPConfig):
        super().__init__()
        self.config = config
        
        # Optional Fourier encoding for query coordinates
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_decoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        # Input: z + fourier_features
        input_dim = config.z_dim + self.spatial_encoder.output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.decoder_hidden),
            nn.ReLU(),
            nn.Linear(config.decoder_hidden, config.decoder_hidden),
            nn.ReLU(),
            nn.Linear(config.decoder_hidden, config.decoder_hidden),
            nn.ReLU()
        )
        
        self.mu_head = nn.Linear(config.decoder_hidden, 1)
        self.sigma_head = nn.Linear(config.decoder_hidden, 1)
    
    def forward(self, z: torch.Tensor, query: Obs) -> Normal:
        """
        Args:
            z: Latent variable (B, z_dim)
            query: Query locations
        
        Returns:
            Distribution over predictions
        """
        B, N = query.xs.shape
        
        # Expand z for each query point
        z_exp = z.unsqueeze(1).expand(-1, N, -1)
        
        # Encode query coordinates
        coords = torch.stack([query.xs, query.ys], dim=-1)
        spatial_features = self.spatial_encoder(coords)
        
        # Concatenate and decode
        decoder_in = torch.cat([z_exp, spatial_features], dim=-1)
        feat = self.net(decoder_in.view(B*N, -1))
        
        mu = self.mu_head(feat).view(B, N, 1)
        sigma = F.softplus(self.sigma_head(feat)).view(B, N, 1) + self.config.min_std
        
        return Normal(mu, sigma)


class SNP_v1(nn.Module):
    """
    Stochastic Neural Process for scalar field prediction.
    Implements latent path with prior/posterior and posterior dropout.
    """
    def __init__(self, config: SNPConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = Encoder(config)
        self.prior = PriorNet(config.h_dim, config.z_dim, config.prior_hidden)
        self.posterior = PosteriorNet(config.h_dim, config.r_dim, config.z_dim, config.posterior_hidden)
        self.decoder = Decoder(config)
        
        # LSTM for temporal dynamics
        lstm_input_dim = config.r_dim
        if config.use_actions:
            lstm_input_dim += config.action_dim
        self.lstm = nn.LSTMCell(lstm_input_dim, config.h_dim)
        
        self.apply(init_weights)
    
    def forward(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        obs: Optional[Obs] = None,
        query: Optional[Obs] = None
    ) -> SNPOutput:
        """
        One step forward.
        
        Args:
            state: LSTM state (h, c)
            action: Action vector (optional if use_actions=False)
            done: Done flag (unused, for interface compatibility)
            obs: Context observation
            query: Query observation (with values for training)
        
        Returns:
            SNPOutput with predictions and latent variables
        """
        h_prev, c_prev = state
        B = h_prev.shape[0]
        device = h_prev.device
        
        # 1. Encode context
        if obs is not None:
            r_ctx = self.encoder(obs)
        else:
            r_ctx = torch.zeros(B, self.config.r_dim, device=device)
        
        # 2. Encode target (for posterior, training only)
        r_tgt = None
        if query is not None and query.values is not None and self.training:
            r_tgt = self.encoder(query)
        
        # 3. Update LSTM state
        if self.config.use_actions and action is not None:
            lstm_in = torch.cat([r_ctx, action], dim=-1)
        else:
            lstm_in = r_ctx
        
        h, c = self.lstm(lstm_in, (h_prev, c_prev))
        
        # 4. Latent path
        prior_mu, prior_sigma = self.prior(h)
        dist_prior = Normal(prior_mu, prior_sigma)
        
        post_mu, post_sigma = None, None
        
        if self.training:
            if r_tgt is not None:
                # Posterior available
                post_mu, post_sigma = self.posterior(h, r_tgt)
                dist_post = Normal(post_mu, post_sigma)
                
                # Posterior dropout: 50% chance to use prior
                if torch.rand(1).item() < 0.5:
                    z = dist_prior.rsample()
                else:
                    z = dist_post.rsample()
            else:
                # No target, use prior
                z = dist_prior.rsample()
        else:
            # Inference: use prior mean
            z = prior_mu
        
        # 5. Decode
        prediction = None
        if query is not None:
            prediction = self.decoder(z, query)
        
        return SNPOutput(
            state=(h, c),
            prediction=prediction,
            z=z,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            post_mu=post_mu,
            post_sigma=post_sigma
        )
    
    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM state."""
        h = torch.zeros(batch_size, self.config.h_dim, device=device)
        c = torch.zeros(batch_size, self.config.h_dim, device=device)
        return (h, c)
