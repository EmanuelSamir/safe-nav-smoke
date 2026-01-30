"""
SNP v2 - Improved Stochastic Neural Process
With VRNN and DRNN separation + Fourier Features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Optional, Tuple, List

from src.models.shared.observations import Obs
from src.models.shared.layers import AttentionAggregator, SoftplusSigma
from src.models.shared.fourier_features import ConditionalFourierFeatures

@dataclass
class SNP_v2_Config:
    """Configuration for SNP v2 model."""
    # Dimensions
    x_dim: int = 2
    y_dim: int = 1
    action_dim: int = 2
    use_actions: bool = True
    
    # Latent and RNN dimensions
    r_dim: int = 100       # Dimension of aggregated observation
    embed_dim: int = 100   # Dimension of the embedding before RNN
    h_dim: int = 100       # RNN hidden state dimension (deter_dim)
    z_dim: int = 100       # Latent z dimension (stoch_dim)
    
    # Hidden layers dimensions
    encoder_hidden_dim: int = 100
    prior_hidden_dim: int = 100
    posterior_hidden_dim: int = 100
    decoder_hidden_dim: int = 100
    
    # Configuration
    min_std: float = 0.1
    max_std_scale: float = 0.9
    
    # Fourier Features
    use_fourier_encoder: bool = False
    use_fourier_decoder: bool = False
    fourier_frequencies: int = 128
    fourier_scale: float = 20.0
    spatial_max: float = 100.0


@dataclass
class SNP_v2_State:
    """
    Bundles the hidden states of the VRNN and DRNN, plus the latest latent sample.
    """
    vrnn_h: torch.Tensor
    vrnn_c: torch.Tensor
    drnn_h: torch.Tensor
    drnn_c: torch.Tensor
    z: torch.Tensor

    def reset_at(self, dones: torch.Tensor):
        """Resets states where dones is True."""
        mask = 1.0 - dones.float().view(-1, 1)
        return SNP_v2_State(
            vrnn_h=self.vrnn_h * mask,
            vrnn_c=self.vrnn_c * mask,
            drnn_h=self.drnn_h * mask,
            drnn_c=self.drnn_c * mask,
            z=self.z * mask
        )


@dataclass
class SNP_v2_Output:
    """Output of a single forward step of SNP v2."""
    state: SNP_v2_State
    prior: Normal
    posterior: Optional[Normal]
    decoded: Optional[torch.Tensor]


class Encoder(nn.Module):
    """Encodes a set of (x, y, value) into a latent distribution."""
    def __init__(self, config: SNP_v2_Config):
        super().__init__()
        self.config = config
        
        # Fourier encoder
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_encoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        input_dim = self.spatial_encoder.output_dim + config.y_dim
        
        self.point_net = nn.Sequential(
            nn.Linear(input_dim, config.encoder_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(config.encoder_hidden_dim, config.r_dim)
        )
        self.aggregator = AttentionAggregator(config.r_dim, config.encoder_hidden_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(config.r_dim, config.encoder_hidden_dim), nn.ReLU(inplace=True)
        )
        self.mu_head = nn.Linear(config.encoder_hidden_dim, config.embed_dim)
        self.sigma_layer = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, config.embed_dim),
            SoftplusSigma(min_std=config.min_std, scale=config.max_std_scale)
        )

    def forward(self, obs: Obs) -> Normal:
        # Fourier encoding
        coords = torch.stack([obs.xs, obs.ys], dim=-1)
        spatial_feat = self.spatial_encoder(coords)
        
        # Stack pos and value: (B, N, feat_dim + 1)
        values = obs.values
        if values.dim() == 3:
            values = values.squeeze(-1)
        x = torch.cat([spatial_feat, values.unsqueeze(-1)], dim=-1)
        
        rs = self.point_net(x)
        r = self.aggregator(rs, mask=obs.mask)
        
        hidden = self.projection(r)
        # Stronger clamping to prevent initial explosion
        hidden = torch.clamp(hidden, min=-10.0, max=10.0) 
        mu = self.mu_head(hidden)
        # Add clamping to avoid extreme values before Softplus
        sigma = self.sigma_layer(hidden)
        return Normal(mu, sigma)


class LatentDistributor(nn.Module):
    """Manages Prior and Posterior distributions for the latent space z."""
    def __init__(self, config: SNP_v2_Config):
        super().__init__()
        self.config = config
        
        # Prior conditions on VRNN state
        self.prior_net = nn.Sequential(
            nn.Linear(config.h_dim, config.prior_hidden_dim), nn.ELU(),
            nn.Linear(config.prior_hidden_dim, config.prior_hidden_dim), nn.ELU()
        )
        self.prior_mu = nn.Linear(config.prior_hidden_dim, config.z_dim)
        self.prior_mu = nn.Linear(config.prior_hidden_dim, config.z_dim)
        self.prior_sigma = nn.Sequential(
            nn.Linear(config.prior_hidden_dim, config.z_dim), 
            SoftplusSigma(min_std=config.min_std)
        )

        # Posterior conditions on VRNN state AND current observation embedding
        self.posterior_net = nn.Sequential(
            nn.Linear(config.h_dim + config.embed_dim, config.posterior_hidden_dim), nn.ELU(),
            nn.Linear(config.posterior_hidden_dim, config.posterior_hidden_dim), nn.ELU()
        )
        self.posterior_mu = nn.Linear(config.posterior_hidden_dim, config.z_dim)
        self.posterior_mu = nn.Linear(config.posterior_hidden_dim, config.z_dim)
        self.posterior_sigma = nn.Sequential(
            nn.Linear(config.posterior_hidden_dim, config.z_dim), 
            SoftplusSigma(min_std=config.min_std)
        )

    def get_prior(self, vrnn_h: torch.Tensor) -> Normal:
        out = self.prior_net(vrnn_h)
        return Normal(self.prior_mu(out), self.prior_sigma(out))

    def get_posterior(self, vrnn_h: torch.Tensor, obs_embed: torch.Tensor) -> Normal:
        out = self.posterior_net(torch.cat([vrnn_h, obs_embed], dim=-1))
        return Normal(self.posterior_mu(out), self.posterior_sigma(out))


class Decoder(nn.Module):
    """Predicts measurement values at query locations given the latent state."""
    def __init__(self, config: SNP_v2_Config):
        super().__init__()
        self.config = config
        
        # Fourier encoder
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_decoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        # Input: h_d (DRNN) + z (latent) + spatial_feat
        input_dim = config.h_dim + config.z_dim + self.spatial_encoder.output_dim
        
        self.input_proj = nn.Linear(input_dim, config.decoder_hidden_dim)
        
        # Residual blocks for better field modeling
        self.res1 = self._make_res_block(config.decoder_hidden_dim)
        self.res2 = self._make_res_block(config.decoder_hidden_dim)
        
        self.output = nn.Linear(config.decoder_hidden_dim, config.y_dim)

    def _make_res_block(self, dim: int):
        return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, drnn_h: torch.Tensor, z: torch.Tensor, query: Obs) -> torch.Tensor:
        # drnn_h: (B, h_dim), z: (B, z_dim)
        B, N = query.xs.shape
        
        # Expand latents to match query points
        h_exp = drnn_h.unsqueeze(1).expand(-1, N, -1)
        z_exp = z.unsqueeze(1).expand(-1, N, -1)
        
        # Fourier encode query
        q_pos = torch.stack([query.xs, query.ys], dim=-1)
        spatial_feat = self.spatial_encoder(q_pos)
        
        # Flatten for efficient processing
        flat_input = torch.cat([h_exp, z_exp, spatial_feat], dim=-1).view(B * N, -1)
        
        feat = F.relu(self.input_proj(flat_input))
        feat = feat + F.relu(self.res1(feat))
        feat = feat + F.relu(self.res2(feat))
        
        return self.output(feat).view(B, N, self.config.y_dim)


class SNP_v2(nn.Module):
    """
    Improved Variational Auto-Encoding State Space Model (SNP V2).
    With VRNN and DRNN separation.
    """
    def __init__(self, config: SNP_v2_Config):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(config)
        self.latent_dist = LatentDistributor(config)
        self.decoder = Decoder(config)
        
        # RNN Dynamics
        self.vrnn_cell = nn.LSTMCell(config.embed_dim, config.h_dim)
        self.drnn_cell = nn.LSTMCell(config.action_dim, config.h_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTMCell):
            nn.init.orthogonal_(m.weight_ih, gain=0.01)
            nn.init.orthogonal_(m.weight_hh, gain=0.01)
            nn.init.constant_(m.bias_ih, 0)
            nn.init.constant_(m.bias_hh, 0)

    def init_state(self, batch_size: int, device: torch.device) -> SNP_v2_State:
        """Helper to create an empty initial state."""
        kwargs = {"device": device, "dtype": torch.float32}
        return SNP_v2_State(
            vrnn_h=torch.zeros(batch_size, self.config.h_dim, **kwargs),
            vrnn_c=torch.zeros(batch_size, self.config.h_dim, **kwargs),
            drnn_h=torch.zeros(batch_size, self.config.h_dim, **kwargs),
            drnn_c=torch.zeros(batch_size, self.config.h_dim, **kwargs),
            z=torch.zeros(batch_size, self.config.z_dim, **kwargs)
        )

    def forward(self, 
                state: SNP_v2_State, 
                action: torch.Tensor, 
                dones: torch.Tensor, 
                obs: Optional[Obs] = None, 
                query: Optional[Obs] = None) -> SNP_v2_Output:
        """
        Single step transition of the model.
        """
        # 1. Reset states if sequences ended
        state = state.reset_at(dones)
        
        # 2. Update Deterministic Path (Action)
        if hasattr(self.config, 'use_actions') and not self.config.use_actions:
             action = torch.zeros_like(action)
        drnn_h, drnn_c = self.drnn_cell(action, (state.drnn_h, state.drnn_c))
        
        # 3. Process Observation and Update Latent Path (VRNN)
        if obs is not None:
            obs_dist = self.encoder(obs)
            obs_embed = obs_dist.rsample() # Sample embedding for stochasticity
            
            vrnn_h, vrnn_c = self.vrnn_cell(obs_embed, (state.vrnn_h, state.vrnn_c))
            
            # Clamp cell state to prevent instability
            vrnn_c = torch.clamp(vrnn_c, -100., 100.)
            
            prior = self.latent_dist.get_prior(vrnn_h)
            posterior = self.latent_dist.get_posterior(vrnn_h, obs_dist.loc) # Use mean for posterior conditioning consistency
            z = posterior.rsample()
        else:
            # Blind transition
            blind_input = torch.zeros(action.shape[0], self.config.embed_dim, device=action.device)
            vrnn_h, vrnn_c = self.vrnn_cell(blind_input, (state.vrnn_h, state.vrnn_c))
            
            prior = self.latent_dist.get_prior(vrnn_h)
            posterior = None
            z = prior.rsample()
            
        # 4. Decode if query points provided
        decoded = self.decoder(drnn_h, z, query) if query is not None else None
        
        next_state = SNP_v2_State(vrnn_h, vrnn_c, drnn_h, drnn_c, z)
        return SNP_v2_Output(next_state, prior, posterior, decoded)

    def forward_sequence(self, initial_state: SNP_v2_State, 
                         sequence: List[Tuple[Obs, torch.Tensor, torch.Tensor]]) -> List[SNP_v2_Output]:
        """Runs the model over an entire sequence of (obs, action, done)."""
        outputs = []
        current_state = initial_state
        for obs, action, done in sequence:
            out = self.forward(current_state, action, done, obs, query=obs)
            outputs.append(out)
            current_state = out.state
        return outputs

    @staticmethod
    def load_from_checkpoint(path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        params = ckpt.get('config', ckpt.get('hyper_parameters', SNP_v2_Config()))
        model = SNP_v2(params).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model
