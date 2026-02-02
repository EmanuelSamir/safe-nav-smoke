"""
RNP - Recurrent Neural Process
Clean implementation with Fourier Features support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
from torch.distributions import Normal

from src.models.shared.observations import Obs
from src.models.shared.outputs import RNPOutput
from src.models.shared.fourier_features import ConditionalFourierFeatures


@dataclass
class RNPConfig:
    """Configuration for RNP model."""
    # Dimensions
    r_dim: int = 128
    h_dim: int = 128
    
    # Model options
    action_dim: int = 2
    use_actions: bool = False
    num_layers: int = 3
    lstm_layers: int = 1
    min_std: float = 0.01
    
    # Fourier Features
    use_fourier_encoder: bool = False
    use_fourier_decoder: bool = False
    fourier_frequencies: int = 128
    fourier_scale: float = 20.0
    spatial_max: float = 100.0


def init_weights(m):
    """Xavier initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLP(nn.Module):
    """Simple MLP."""
    def __init__(self, layer_sizes, output_dim):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """Encodes observations into representation r."""
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Fourier encoding for spatial coordinates
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_encoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        # MLP: (fourier_features, value) -> r
        input_dim = self.spatial_encoder.output_dim + 1
        layer_sizes = [input_dim] + [config.h_dim] * config.num_layers
        self.net = MLP(layer_sizes, config.r_dim)
    
    def forward(self, obs: Obs) -> torch.Tensor:
        """Encode observation to representation."""
        # Encode coordinates
        coords = torch.stack([obs.xs, obs.ys], dim=-1)
        spatial_features = self.spatial_encoder(coords)
        
        # Concatenate with values
        values = obs.values if obs.values.dim() == 2 else obs.values.squeeze(-1)
        inputs = torch.cat([spatial_features, values.unsqueeze(-1)], dim=-1)
        
        # Encode
        embeddings = self.net(inputs)
        
        # Aggregate with mask
        if obs.mask is not None:
            mask_expanded = obs.mask.unsqueeze(-1).float()
            sum_emb = (embeddings * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            r = sum_emb / count
        else:
            r = embeddings.mean(dim=1)
        
        return r


class Forecaster(nn.Module):
    """LSTM-based dynamics model."""
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, r_input, prev_hidden):
        """
        Args:
            r_input: (B, input_dim)
            prev_hidden: (h, c) each (num_layers, B, hidden_dim)
        
        Returns:
            r_next: (B, hidden_dim)
            state: (h, c)
        """
        # Add time dimension
        r_input = r_input.unsqueeze(1)  # (B, 1, input_dim)
        out, (h, c) = self.lstm(r_input, prev_hidden)
        r_next = out.squeeze(1)  # (B, hidden_dim)
        return r_next, (h, c)


class Decoder(nn.Module):
    """Decodes representation and query to predictions."""
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Fourier encoding for query coordinates
        self.spatial_encoder = ConditionalFourierFeatures(
            input_dim=2,
            use_fourier=config.use_fourier_decoder,
            num_frequencies=config.fourier_frequencies,
            frequency_scale=config.fourier_scale,
            input_max=config.spatial_max
        )
        
        # MLP: (h, fourier_features) -> (mu, sigma)
        input_dim = config.h_dim + self.spatial_encoder.output_dim
        layer_sizes = [input_dim] + [config.h_dim] * config.num_layers
        self.net = MLP(layer_sizes, 2)  # mu, logsigma
    
    def forward(self, h: torch.Tensor, query: Obs) -> Normal:
        """
        Args:
            h: Hidden state (B, h_dim)
            query: Query locations
        
        Returns:
            Distribution over predictions
        """
        B, N = query.xs.shape
        
        # Expand h
        h_exp = h.unsqueeze(1).expand(-1, N, -1)
        
        # Encode query coordinates
        coords = torch.stack([query.xs, query.ys], dim=-1)
        spatial_features = self.spatial_encoder(coords)
        
        # Concatenate and decode
        decoder_in = torch.cat([h_exp, spatial_features], dim=-1)
        out = self.net(decoder_in.view(B*N, -1))
        
        mu = out[:, 0].view(B, N, 1)
        logsigma = out[:, 1].view(B, N, 1)
        sigma = F.softplus(logsigma) + self.config.min_std
        
        return Normal(mu, sigma)


class RNP(nn.Module):
    """
    Recurrent Neural Process for scalar field prediction.
    Deterministic model with LSTM dynamics.
    """
    def __init__(self, config: RNPConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = Encoder(config)
        
        # Forecaster input: r + [optional: action]
        forecaster_input = config.r_dim
        if config.use_actions:
            forecaster_input += config.action_dim
        self.forecaster = Forecaster(forecaster_input, config.h_dim, config.lstm_layers)
        
        self.decoder = Decoder(config)
        
        self.apply(init_weights)
    
    def forward(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        obs: Optional[Obs] = None,
        query: Optional[Obs] = None
    ) -> RNPOutput:
        """
        One step forward.
        
        Args:
            state: LSTM state (h, c)
            action: Action vector (optional)
            done: Done flag (unused)
            obs: Context observation
            query: Query observation
        
        Returns:
            RNPOutput with predictions
        """
        h_prev, c_prev = state
        B = h_prev.shape[1] if h_prev.dim() > 1 else 1
        device = h_prev.device
        
        # 1. Encode context
        if obs is not None:
            r = self.encoder(obs)
        else:
            r = torch.zeros(B, self.config.r_dim, device=device)
        
        # 2. Dynamics
        if self.config.use_actions and action is not None:
            forecaster_in = torch.cat([r, action], dim=-1)
        else:
            forecaster_in = r
        
        r_next, (h, c) = self.forecaster(forecaster_in, (h_prev, c_prev))
        
        # 3. Decode
        prediction = None
        if query is not None:
            prediction = self.decoder(r_next, query)
        
        return RNPOutput(
            state=(h, c),
            prediction=prediction
        )
    
    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM state."""
        h = torch.zeros(self.config.lstm_layers, batch_size, self.config.h_dim, device=device)
        c = torch.zeros(self.config.lstm_layers, batch_size, self.config.h_dim, device=device)
        return (h, c)
