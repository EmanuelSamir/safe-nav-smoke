import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, activation=nn.ReLU()):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, hidden_dim=64):
        super().__init__()
        # Encodes (x, y) pairs into representation r
        self.mlp = MLP(x_dim + y_dim, hidden_dim, r_dim)

    def forward(self, x_context, y_context):
        # x_context: (B, N_ctx, x_dim)
        # y_context: (B, N_ctx, y_dim)
        xy = torch.cat([x_context, y_context], dim=-1) # (B, N_ctx, x_dim + y_dim)
        r_i = self.mlp(xy) # (B, N_ctx, r_dim)
        r = torch.mean(r_i, dim=1) # (B, r_dim) - Aggregation
        return r

class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dim=64):
        super().__init__()
        # Encodes (x, y) pairs into (mu, sigma) of latent z
        self.mlp = MLP(x_dim + y_dim, hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, z_dim)
        self.sigma_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, x_context, y_context):
        # x_context: (B, N_ctx, x_dim)
        # y_context: (B, N_ctx, y_dim)
        xy = torch.cat([x_context, y_context], dim=-1)
        h = self.mlp(xy) # (B, N_ctx, hidden_dim)
        h_agg = torch.mean(h, dim=1) # (B, hidden_dim)
        
        mu = self.mean_layer(h_agg) # (B, z_dim)
        log_sigma = self.sigma_layer(h_agg) # (B, z_dim)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma) # Restrict sigma range often helps stability
        # Or standard exp: sigma = torch.exp(log_sigma)
        
        return mu, sigma

class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_dim=1, mapping_size=32, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        # B is sampled from Gaussian distribution N(0, scale^2) = scale * N(0, 1)
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: (..., input_dim)
        # B: (input_dim, mapping_size)
        # proj: (..., mapping_size)
        proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class Decoder(nn.Module):
    def __init__(self, x_dim, r_dim, y_dim, hidden_dim=64):
        super().__init__()
        # Decodes (x_target, r) into y_target mean (and optionally variance)
        self.mlp = MLP(x_dim + r_dim, hidden_dim, y_dim, activation=nn.Tanh()) # Output mean only for basic MSE

    def forward(self, x_target, r):
        # x_target: (B, N_tar, x_dim)
        # r: (B, r_dim)
        # Repeat r for each target point
        N_tar = x_target.shape[1]
        r_rep = r.unsqueeze(1).repeat(1, N_tar, 1) # (B, N_tar, r_dim)
        
        xr = torch.cat([x_target, r_rep], dim=-1) # (B, N_tar, x_dim + r_dim)
        y_pred = self.mlp(xr) # (B, N_tar, y_dim)
        return y_pred

def get_beta(schedule, current_epoch):
    """
    Computes beta value based on schedule configuration.
    schedule: dict with 'start', 'end', 'warmup', and optional 'start_epoch'
    current_epoch: current training epoch
    """
    start = schedule.get('start', 0.0)
    end = schedule.get('end', 0.0)
    warmup = schedule.get('warmup', 0)
    start_epoch = schedule.get('start_epoch', 0) # Delay before warmup starts
    
    if current_epoch < start_epoch:
        return start
        
    # Effective epoch relative to start_epoch
    relative_epoch = current_epoch - start_epoch
    
    if warmup <= 0:
        return end
        
    if relative_epoch >= warmup:
        return end
        
    # Linear interpolation
    alpha = relative_epoch / float(warmup)
    return start + alpha * (end - start)
