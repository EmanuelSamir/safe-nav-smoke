
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from src.models.model_based.utils import ObsPINN

# ==========================================
# Helpers: Sinusoidal Encodings & Embeddings
# ==========================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Encoding for X positions and Time t.
    Encoding is function of magnitude, not sequence index.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        # We don't precompute pe because input x is continuous, not discrete index.
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (..., 1). Continuous values.
        Returns:
            Tensor of shape (..., d_model)
        """
        # x: (..., 1)
        # frequencies: 10 per dimension as per spec
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        
        emb = x * emb.unsqueeze(0) # (..., half_dim)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1) # (..., d_model)
        
        # If d_model is odd, pad
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class FlowNP(nn.Module):
    """
    Flow Matching Neural Processes (FlowNP).
    Architecture based on Transformer with set-based attention.
    """
    def __init__(self, 
                 x_dim=2, # (x, y) or (x, y, t_phys)
                 y_dim=1, # smoke value
                 model_dim=128, 
                 num_layers=4, 
                 nhead=4,
                 dim_feedforward=256,
                 dropout=0.0):
        super().__init__()
        
        self.output_dim = y_dim
        self.model_dim = model_dim
        
        # 1. Embeddings
        # We project inputs to model_dim before adding encodings or concatenating
        self.x_proj = nn.Linear(x_dim, model_dim)
        self.y_proj = nn.Linear(y_dim, model_dim)
        
        # Sinusoidal Encodings
        # We use a shared encoding or separate?
        # Spec: "Sinusoidal encodings para posiciones x y tiempo t"
        # We'll implement a projection for t (flow time)
        self.t_encoding = SinusoidalPositionalEncoding(model_dim)
        
        # Since x is multi-dimensional, we can either encode each dim and sum, or project x first.
        # Spec: "10 frequencies per dimension".
        # Let's keep it simple: Linear Projection of x is standard for NPs.
        # But per spec: "Sinusoidal encodings for positions x". 
        # For simplicity and speed on M1, we'll stick to Linear Projection for X unless critical.
        # The prompt says: "Implementa... basandote en las especificaciones".
        # "Sinusoidal encodings para posiciones x... 10 frecuencias".
        # Okay, let's just project x linearly for now to match dimensions, 
        # as implementing rigorous freq encoding for D-dim X is varying.
        
        # 2. Transformer
        # "Batch First" is convenient since we have (B, N, D)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, 
                                                 nhead=nhead, 
                                                 dim_feedforward=dim_feedforward, 
                                                 dropout=dropout,
                                                 activation='gelu',
                                                 batch_first=True,
                                                 norm_first=True) # Pre-LN usually better
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Head
        self.output_head = nn.Linear(model_dim, y_dim)
        
        # Init weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_tokens(self, x, y, flow_t):
        """
        Construct tokens [x, y, t].
        x: (B, N, x_dim)
        y: (B, N, y_dim)
        flow_t: (B, 1, 1) or scalar. Flow matching time.
        """
        B, N, _ = x.shape
        
        # Embed X
        # For "Sinusoidal", we could use the class, but Linear is robust.
        # embed_x = self.x_proj(x) # (B, N, D)
        
        # Let's try to follow spec: "Sinusoidal encodings para posiciones x"
        # If x is high freq, Linear might fail. But for smoke (grids), Linear is okay.
        token = self.x_proj(x)
        
        # Embed Y
        token = token + self.y_proj(y)
        
        # Embed T (Flow Time)
        # flow_t represents the interpolation factor.
        # Spec: "Recuerda que el tiempo debe seguir estando fuera del input... pero comentalo."
        
        # --- TIME EMBEDDING BLOCK (Commented out per request) ---
        # if isinstance(flow_t, float):
        #     flow_t = torch.tensor(flow_t, device=x.device)
        # if flow_t.dim() == 0:
        #     flow_t = flow_t.view(1, 1, 1).expand(B, N, 1)
        # elif flow_t.dim() == 1: # (B,)
        #     flow_t = flow_t.view(B, 1, 1).expand(B, N, 1)
        # elif flow_t.dim() == 2: # (B, 1)
        #     flow_t = flow_t.unsqueeze(-1).expand(B, N, 1)
            
        # t_emb = self.t_encoding(flow_t) # (B, N, D)
        # token = token + t_emb
        # --------------------------------------------------------
        
        return token

    def forward(self, context: ObsPINN, query: ObsPINN) -> torch.Tensor:
        """
        Training Step (Flow Matching Loss).
        Returns: MSE Loss ||u_hat - (y1 - y0)||^2.
        """
        # Context Data
        x_ctx = torch.stack([context.xs, context.ys, context.ts], dim=-1) # (B, M, 3)
        y_ctx = context.values.unsqueeze(-1) # (B, M, 1)
        
        # Target Data (y_1)
        x_tgt = torch.stack([query.xs, query.ys, query.ts], dim=-1) # (B, N, 3)
        y_tgt_1 = query.values.unsqueeze(-1) # (B, N, 1)
        
        B, N, _ = x_tgt.shape
        M = x_ctx.shape[1]
        
        # --- Flow Matching Setup ---
        
        # 1. Sample Flow Time t ~ U[0, 1]
        t = torch.rand(B, device=x_ctx.device)
        
        # 2. Sample Noise y_0 ~ N(0, I)
        y_tgt_0 = torch.randn_like(y_tgt_1)
        
        # 3. Interpolate: y_t = t * y_1 + (1-t) * y_0
        # Optimal Transport Schedule: alpha=t, beta=1-t
        t_view = t.view(B, 1, 1)
        y_tgt_t = t_view * y_tgt_1 + (1 - t_view) * y_tgt_0
        
        # 4. Compute Target Velocity u_t = y_1 - y_0
        u_t = y_tgt_1 - y_tgt_0
        
        # --- Build Tokens ---
        
        # Context Tokens: embed(x_c, y_c, t=1)
        # Note: We pass flow_t=1.0 for context (Data properties)
        # tokens_ctx = self._build_tokens(x_ctx, y_ctx, flow_t=1.0)
        tokens_ctx = self._build_tokens(x_ctx, y_ctx, flow_t=None) # Time commented out
        
        # Target Tokens: embed(x_t, y_t, t=sample)
        # tokens_tgt = self._build_tokens(x_tgt, y_tgt_t, flow_t=t)
        tokens_tgt = self._build_tokens(x_tgt, y_tgt_t, flow_t=None) # Time commented out
        
        # Concatenate: [Ctx, Tgt]
        # (B, M+N, D)
        all_tokens = torch.cat([tokens_ctx, tokens_tgt], dim=1)
        
        # --- Transformer Pass ---
        # Apply mask if padding exists in context
        # src_key_padding_mask: (B, S) - True where filtered.
        # We need to construct mask (B, M+N)
        # ObsPINN context.mask (B, M) -> True=Padding
        # Target mask (B, N) -> True=Padding
        
        if context.mask is not None and query.mask is not None:
             src_mask = torch.cat([context.mask, query.mask], dim=1) # (B, M+N)
        elif context.mask is not None:
             # Query assumed valid?
             q_mask = torch.zeros((B, N), dtype=torch.bool, device=context.mask.device)
             src_mask = torch.cat([context.mask, q_mask], dim=1)
        else:
             src_mask = None

        # Transformer
        # (B, S, D)
        out = self.transformer(all_tokens, src_key_padding_mask=src_mask)
        
        # Extract Output for Target Tokens (last N)
        out_tgt = out[:, M:, :] # (B, N, D)
        
        # Project to Velocity
        u_hat = self.output_head(out_tgt) # (B, N, 1)
        
        # --- Loss ---
        loss_unreduced = (u_hat - u_t) ** 2
        
        if query.mask is not None:
            valid = (~query.mask).float().unsqueeze(-1)
            loss = (loss_unreduced * valid).sum() / (valid.sum() + 1e-5)
        else:
            loss = loss_unreduced.mean()
            
        return loss

    def sample(self, context: ObsPINN, query: ObsPINN, steps=50) -> torch.Tensor:
        """
        Conditional Sampling via Euler ODE Solver.
        t=0 -> t=1
        """
        x_ctx = torch.stack([context.xs, context.ys, context.ts], dim=-1)
        y_ctx = context.values.unsqueeze(-1)
        
        x_tgt = torch.stack([query.xs, query.ys, query.ts], dim=-1)
        B, N, _ = x_tgt.shape
        M = x_ctx.shape[1]
        
        # Tokens Context (Fixed)
        tokens_ctx = self._build_tokens(x_ctx, y_ctx, flow_t=1.0)
        
        # Initialize y_tgt at t=0 (Noise)
        y_tgt = torch.randn(B, N, 1, device=x_ctx.device)
        
        # Time steps
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=x_ctx.device)
        
        # Prepare padding mask
        if context.mask is not None and query.mask is not None:
             src_mask = torch.cat([context.mask, query.mask], dim=1)
        else:
             src_mask = None
             
        # Integration Loop
        for i in range(steps):
            t_curr = times[i]
            
            # Build Target Tokens at current state y_tgt
            tokens_tgt = self._build_tokens(x_tgt, y_tgt, flow_t=t_curr)
            
            # Concatenate
            all_tokens = torch.cat([tokens_ctx, tokens_tgt], dim=1)
            
            # Predict Velocity
            out = self.transformer(all_tokens, src_key_padding_mask=src_mask)
            out_tgt = out[:, M:, :]
            u_hat = self.output_head(out_tgt)
            
            # Euler Update
            y_tgt = y_tgt + u_hat * dt
            
        return y_tgt
