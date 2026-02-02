
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.model_based.utils import ObsPINN

# ==========================================
# FNP Architecture (Autonomous / No Flow Time Input)
# ==========================================

def comp_posenc(dim_posenc, pos):
    """
    Computes Sinusoidal Positional Encoding.
    pos: (B, N, D)
    Returns: (B, N, D*dim_posenc)
    """
    shp = pos.shape
    omega = torch.arange(dim_posenc//2, dtype=torch.float, device=pos.device)
    omega = torch.pi * (2.0**(omega - 2.0))
    
    # pos: (B, N, D) -> (B, N, D, 1)
    # omega: (K) -> (1, 1, 1, K)
    out = pos.unsqueeze(-1) * omega.view(1, 1, 1, -1)
    
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    
    emb = torch.cat([emb_sin, emb_cos], dim=-1) # (B, N, D, 2*K) = (..., D*dim_p)
    return emb.reshape(shp[0], shp[1], -1)

def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU()]
    for _ in range(depth - 1):
        modules.extend([nn.Linear(dim_hid, dim_hid), nn.ReLU()])
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)

class FlowNP(nn.Module):
    """
    Flow Matching Neural Process (FNP)
    Autonomous Flow: Velocity v(y_t, Context) depends only on state, not flow time t.
    """
    def __init__(self, 
                 dim_x=3,       # (x, y, t_phys)
                 dim_y=1,       # Smoke value
                 dim_posenc=10, 
                 d_model=128, 
                 emb_depth=2, 
                 dim_feedforward=256, 
                 nhead=4, 
                 dropout=0.0, 
                 num_layers=3, 
                 timesteps=20):
        super().__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_posenc = dim_posenc
        self.timesteps = timesteps
        
        # Predictor Head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y)
        )
        
        # Embedder
        # Input: dim_x * dim_posenc + dim_y
        # No extra channels for time/ones (Autonomous)
        input_dim = dim_x * dim_posenc + dim_y
        
        self.embedder = build_mlp(input_dim, d_model, d_model, emb_depth)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def encode(self, xc, yc, xt, yt):
        # xc: (B, M, F_x) ; yc: (B, M, F_y)
        x_y_ctx = torch.cat((xc, yc), dim=-1)
        x_y_tar = torch.cat((xt, yt), dim=-1)
        
        # Concatenate full sequence
        inp = torch.cat((x_y_ctx, x_y_tar), dim=1) # (B, M+N, F_total)
        
        embeddings = self.embedder(inp)
        encoded = self.encoder(embeddings)
        
        # Extract Target Output
        num_tar = xt.shape[1]
        encoded_tar = encoded[:, -num_tar:]
        
        return self.predictor(encoded_tar)

    def predict_velocity(self, y_t, t, context_obs, query_coords):
        """
        Predict vector field v(y_t) (Autonomous).
        t is IGNORED.
        """
        B, N, _ = y_t.shape
        
        # Prepare Context
        xc_raw = torch.stack([context_obs.xs, context_obs.ys], dim=-1)
        yc = context_obs.values.unsqueeze(-1)
        
        # Prepare Target X
        xt_raw = query_coords
        
        # Apply Positional Encoding to Raw Coords
        # No extra 'ones' or 't' channels (Autonomous)
        xc_enc = comp_posenc(self.dim_posenc, xc_raw)
        xt_enc = comp_posenc(self.dim_posenc, xt_raw)
        
        # Predict
        return self.encode(xc_enc, yc, xt_enc, y_t)

    def forward(self, context: ObsPINN, query: ObsPINN) -> torch.Tensor:
        """
        Calculates Flow Matching Loss.
        """
        y1 = query.values.unsqueeze(-1)
        y0 = torch.randn_like(y1)
        
        B, N, _ = y1.shape
        t = torch.rand(B, device=y1.device).view(B, 1, 1)
        
        # Interpolate
        yt = t * y1 + (1 - t) * y0
        
        x_tgt_coords = torch.stack([query.xs, query.ys], dim=-1)
        
        # Predict Velocity (Autonomous: t ignored inside, but passed for API)
        pred_v = self.predict_velocity(yt, t, context, x_tgt_coords)
        
        # Target Velocity
        target_v = y1 - y0
        
        loss = (pred_v - target_v) ** 2
        
        if query.mask is not None:
             valid = (~query.mask).float().unsqueeze(-1)
             return (loss * valid).sum() / (valid.sum() + 1e-5)
        else:
             return loss.mean()

    def sample(self, context: ObsPINN, query: ObsPINN, steps=20) -> torch.Tensor:
        """
        Sampling using Euler (Autonomous Flow).
        """
        B, N = query.xs.shape
        x_tgt_coords = torch.stack([query.xs, query.ys], dim=-1)
        
        yt = torch.randn(B, N, 1, device=query.xs.device)
        dt = 1.0 / steps
        
        for i in range(steps):
             # Autonomous: v depends on yt, context, coords. Not t.
             v = self.predict_velocity(yt, None, context, x_tgt_coords)
             yt = yt + v * dt
             
        return yt
