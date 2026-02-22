import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from src.models.shared.observations import Obs
from src.models.shared.layers import MLP as BaseMLP
from src.models.shared.layers import FourierFeatureEncoder

# Helper to adapt BaseMLP to user expectations (dim_in, dim_hid, dim_out, num_layers)
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, num_layers=3):
        super().__init__()
        layer_sizes = [dim_in] + [dim_hid] * (num_layers - 1)
        self.net = BaseMLP(layer_sizes, dim_out)

    def forward(self, x):
        return self.net(x)

def get_beta(pde_beta_schedule, epoch):
    # Placeholder for potential schedule logic
    if pde_beta_schedule is None: return 0.0
    return pde_beta_schedule.get(epoch, 0.0)

class FlowNP(nn.Module):
    def __init__(self, 
                 dim_x=3,       # (x, y, t_phys)
                 dim_y=1,       # Smoke value
                 fourier_mapping_size=32, 
                 fourier_scale=10.0,
                 d_model=128, 
                 emb_depth=3, 
                 dim_feedforward=256, 
                 nhead=8, 
                 dropout=0.1, 
                 num_layers=4, 
                 timesteps=20, # kept for compatibility with config
                 pde_solver=None,
                 pde_beta_schedule=None):
        super().__init__()
        
        self.pde_solver = pde_solver
        self.pde_beta_schedule = pde_beta_schedule
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.timesteps = timesteps # Sampling steps from config
        
        # 1. Fourier Feature Encoder (Optional)
        # We encode the augmented input: x + time.
        # So input dimension is dim_x + 1 (flow time or mask).
        self.fourier_encoder = None
        input_enc_dim = dim_x + 1
        
        if fourier_mapping_size > 0:
            self.fourier_encoder = FourierFeatureEncoder(input_enc_dim, mapping_size=fourier_mapping_size, scale=fourier_scale)
            # Size increases by 2 * mapping_size
            feat_dim = 2 * fourier_mapping_size # FourierFeatureEncoder output is sin/cos cat
        else:
            feat_dim = input_enc_dim
            
        # 2. Embedder: Projects [encoded_x, y] -> d_model
        # Input: feat_dim + dim_y
        self.embedder = MLP(feat_dim + dim_y, d_model, d_model, num_layers=emb_depth)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 4. Predictor: d_model -> dim_y (Velocity)
        self.predictor = MLP(d_model, dim_feedforward, dim_y, num_layers=3) 


    def _encode_input(self, coords, extra_val):
        # coords: (B, N, dim_x)
        # extra_val: (B, N, 1) -> can be time t or mask 1.
        aug = torch.cat([coords, extra_val], dim=-1) # (B, N, dim_x + 1)
        
        if self.fourier_encoder is not None:
             # FourierFeatureEncoder output already projects input_dim -> 2*mapping_size
             # Using shared.layers implementation which does matmul then sin/cos
             return self.fourier_encoder(aug)
             
        return aug

    def forward(self, context: Obs, query: Obs) -> torch.Tensor:
        """
        Calculates Flow Matching Loss.
        Input: context (Obs), query (Obs) including target values.
        Returns: strict MSE Loss on velocity
        """
        # Unpack Context
        # coords: (B, Nc, 3) -> [x, y, t_phys]
        # values: (B, Nc, 1) -> [smoke]
        
        # Handle potential (B, N) vs (B, N, 1) inputs
        c_xs = context.xs.unsqueeze(-1) if context.xs.ndim == 2 else context.xs
        c_ys = context.ys.unsqueeze(-1) if context.ys.ndim == 2 else context.ys
        c_ts = context.ts.unsqueeze(-1) if context.ts.ndim == 2 else context.ts
        
        x_ctx = torch.cat([c_xs, c_ys, c_ts], dim=-1) # (B, Nc, 3)
        y_ctx = context.values
        
        mask_ctx = context.mask
        if mask_ctx is not None and mask_ctx.ndim == 3:
            mask_ctx = mask_ctx.squeeze(-1) # Ensure (B, Nc)
            
        # Unpack Query (Target)
        q_xs = query.xs.unsqueeze(-1) if query.xs.ndim == 2 else query.xs
        q_ys = query.ys.unsqueeze(-1) if query.ys.ndim == 2 else query.ys
        q_ts = query.ts.unsqueeze(-1) if query.ts.ndim == 2 else query.ts
        
        x_tar = torch.cat([q_xs, q_ys, q_ts], dim=-1) # (B, Nt, 3)
        y_tar = query.values
        
        mask_tar = query.mask 
        if mask_tar is not None and mask_tar.ndim == 3:
            mask_tar = mask_tar.squeeze(-1) # Ensure (B, Nt)
        
        B = x_tar.shape[0]
        # Sample Flow Time t ~ U[0, 1]
        t = torch.rand(B, 1, 1, device=x_tar.device)
        
        # Sample Noise path
        y_1 = y_tar
        y_0 = torch.randn_like(y_1)
        
        # Interpolate
        # t broadcasted to (B, Nt, 1) via elementwise mul
        y_t = t * y_1 + (1 - t) * y_0
        v_target = y_1 - y_0
        
        # Predict Velocity
        v_pred = self.predict_velocity(t, x_ctx, y_ctx, x_tar, y_t, mask_ctx=mask_ctx, mask_tar=mask_tar)
            
        # Calculate Loss
        loss = F.mse_loss(v_pred, v_target, reduction='none') # (B, Nt, dim_y)
        
        if mask_tar is not None:
             # Expand mask to match dim_y
             # mask_tar is (B, Nt). loss is (B, Nt, dim_y)
             mask_broad = mask_tar.unsqueeze(-1).expand_as(loss)
             # Convert to float for multiplication if needed, but bool works for indexing or mul
             loss = loss * mask_broad.float()
             # Normalize by sum of mask
             return loss.sum() / (mask_broad.sum() + 1e-8)
             
        return loss.mean()

    def predict_velocity(self, t, x_ctx, y_ctx, x_tar, y_tar_noisy, mask_ctx=None, mask_tar=None):
        """
        Internal velocity prediction.
        """
        # 1. Prepare Context Tokens: [x_c, 1, y_c]
        ones = torch.ones(x_ctx.shape[:-1] + (1,), device=x_ctx.device)
        xc_enc = self._encode_input(x_ctx, ones)
        
        ctx_tokens = torch.cat([xc_enc, y_ctx], dim=-1)
        
        # 2. Prepare Target Tokens: [x_t, t, y_t_noisy]
        # t might be (B, 1, 1). Expand to (B, Nt, 1)
        if t.ndim == 3 and t.shape[1] == 1:
             t = t.expand(-1, x_tar.shape[1], -1)
             
        xt_enc = self._encode_input(x_tar, t)
        tar_tokens = torch.cat([xt_enc, y_tar_noisy], dim=-1)
        
        # 3. Concatenate Sequence
        inp = torch.cat([ctx_tokens, tar_tokens], dim=1) # (B, Nc+Nt, dim)
        embeddings = self.embedder(inp)
        
        # Prepare padding mask for Transformer
        # Transformer src_key_padding_mask: True wherever the position is padding (should trigger ignore)
        # User mask: True wherever the position is valid
        # So we need ~mask (logical NOT)
        
        src_key_padding_mask = None
        if mask_ctx is not None or mask_tar is not None:
             B, Nc = x_ctx.shape[:2]
             Nt = x_tar.shape[1]
             
             # If one mask is None, assume all valid (True)
             if mask_ctx is None:
                 mask_ctx = torch.ones((B, Nc), dtype=torch.bool, device=x_ctx.device)
             if mask_tar is None:
                 mask_tar = torch.ones((B, Nt), dtype=torch.bool, device=x_tar.device)
                 
             # Concatenate masks
             combined_mask = torch.cat([mask_ctx, mask_tar], dim=1) # (B, Nc+Nt)
             
             # Create padding mask: True where mask is False (invalid)
             src_key_padding_mask = ~combined_mask.bool()
        
        # 4. Transformer
        encoded = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # 5. Predict from target tokens only
        num_tar = x_tar.shape[1]
        encoded_tar = encoded[:, -num_tar:]
        return self.predictor(encoded_tar)

    def sample(self, context: Obs, query: Obs, steps=None) -> torch.Tensor:
        """
        Sampling using Euler Integration (Observation Interface).
        """
        if steps is None: steps = self.timesteps
        
        # Unpack Obs
        x_ctx = torch.cat([context.xs, context.ys, context.ts], dim=-1)
        y_ctx = context.values
        
        mask_ctx = context.mask
        if mask_ctx is not None and mask_ctx.ndim == 3:
            mask_ctx = mask_ctx.squeeze(-1)

        # query.coords_3d gives target locations
        x_tar = torch.cat([query.xs, query.ys, query.ts], dim=-1)
        
        mask_tar = query.mask # Usually None for sampling grid but handled if present
        if mask_tar is not None and mask_tar.ndim == 3:
            mask_tar = mask_tar.squeeze(-1)
        
        B, N, _ = x_tar.shape
        y_t = torch.randn(B, N, self.dim_y, device=x_ctx.device)
        dt = 1.0 / steps
        
        for i in range(steps):
            t_curr = i * dt
            # Broadcast t_curr to (B, 1, 1) then let helper expand
            t_tensor = torch.full((B, 1, 1), t_curr, device=x_ctx.device)
            
            v_pred = self.predict_velocity(t_tensor, x_ctx, y_ctx, x_tar, y_t, mask_ctx=mask_ctx, mask_tar=mask_tar)
            y_t = y_t + v_pred * dt
            
        return y_t
