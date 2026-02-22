import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from .common import MLP, FourierFeatureEncoder, get_beta

class FlowNP(pl.LightningModule):
    def __init__(self, 
                 dim_x=1, 
                 dim_y=1, 
                 fourier_mapping_size=32, 
                 fourier_scale=10.0,
                 d_model=128, 
                 emb_depth=3, 
                 dim_feedforward=256, 
                 nhead=8, 
                 dropout=0.1, 
                 num_layers=4, 
                 lr=1e-3,
                 pde_solver=None,
                 pde_beta_schedule=None):
        super().__init__()
        self.save_hyperparameters(ignore=['pde_solver'])
        
        self.pde_solver = pde_solver
        self.pde_beta_schedule = pde_beta_schedule
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        # 1. Fourier Feature Encoder (Optional)
        # We encode the augmented input: x + time.
        # So input dimension is dim_x + 1.
        self.fourier_encoder = None
        input_enc_dim = dim_x + 1
        
        if fourier_mapping_size > 0:
            self.fourier_encoder = FourierFeatureEncoder(input_enc_dim, fourier_mapping_size, fourier_scale)
            # Size increases by 2 * mapping_size
            feat_dim = input_enc_dim + 2 * fourier_mapping_size
        else:
            feat_dim = input_enc_dim
            
        # 2. Embedder: Projects [encoded_x, y] -> d_model
        # Input: feat_dim + dim_y
        self.embedder = MLP(feat_dim + dim_y, d_model, d_model, num_layers=emb_depth)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 4. Predictor: d_model -> dim_y (Velocity)
        self.predictor = MLP(d_model, dim_feedforward, dim_y, num_layers=3) # Depth 3 like in user code

    def _encode_input(self, coords, extra_val):
        # coords: (B, N, dim_x)
        # extra_val: (B, N, 1) -> can be time t or mask 1.
        aug = torch.cat([coords, extra_val], dim=-1) # (B, N, dim_x + 1)
        
        if self.fourier_encoder is not None:
            f = self.fourier_encoder(aug)
            return torch.cat([aug, f], dim=-1)
        return aug

    def forward(self, x_ctx, y_ctx, x_tar, y_tar_noisy=None, t=None):
        # 1. Prepare Context Tokens: [x_c, 1, y_c]
        ones = torch.ones(x_ctx.shape[:-1] + (1,), device=self.device)
        xc_enc = self._encode_input(x_ctx, ones)
        ctx_tokens = torch.cat([xc_enc, y_ctx], dim=-1)
        
        # 2. Prepare Target Tokens: [x_t, t, y_t_noisy]
        # Broadcast t if needed
        if t is None: t = torch.zeros(x_tar.shape[0], x_tar.shape[1], 1, device=self.device)
        if t.ndim == 2: t = t.unsqueeze(1).expand(-1, x_tar.shape[1], -1)
        
        xt_enc = self._encode_input(x_tar, t)
        tar_tokens = torch.cat([xt_enc, y_tar_noisy], dim=-1)
        
        # 3. Concatenate Sequence
        inp = torch.cat([ctx_tokens, tar_tokens], dim=1) # (B, Nc+Nt, dim)
        embeddings = self.embedder(inp)
        
        # 4. Transformer
        # We need to disable efficient attention for double backward (PINN loss)
        # as of PyTorch 2.x, efficient attention doesn't support higher order derivatives.
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            encoded = self.encoder(embeddings)
        
        # 5. Predict from target tokens only
        num_tar = x_tar.shape[1]
        encoded_tar = encoded[:, -num_tar:]
        return self.predictor(encoded_tar)

    def training_step(self, batch, batch_idx):
        x_ctx, y_ctx = batch['x_context'], batch['y_context']
        x_tar, y_tar = batch['x_target'], batch['y_target']
        
        # 1. Flow Matching Loss
        t = torch.rand(x_tar.shape[0], 1, device=self.device)
        y_1 = y_tar
        y_0 = torch.randn_like(y_1)
        
        t_b = t.unsqueeze(1)
        y_t = t_b * y_1 + (1 - t_b) * y_0 
        v_target = y_1 - y_0
        
        v_pred = self.forward(x_ctx, y_ctx, x_tar, y_tar_noisy=y_t, t=t)
        loss_mse = nn.functional.mse_loss(v_pred, v_target)
        
        # 2. PINN Loss (Optional)
        loss_pde = 0.0
        pde_beta = 0.0

        pde_beta = get_beta(self.pde_beta_schedule, self.current_epoch)
             
        if self.pde_solver is not None and pde_beta > 0:
            # Enable grad for x_tar to compute spatial derivatives
            x_tar_grad = x_tar.clone().requires_grad_(True)
            
            # Sample using Euler integration
            train_steps = 2 # Reduced from 10 to avoid OOM with standard attention
            y_pred_pde = self.sample(x_ctx, y_ctx, x_tar_grad, steps=train_steps)
            
            residual = self.pde_solver.compute_residual(y_pred_pde, x_tar_grad, **batch)
            loss_pde_val = torch.mean(residual ** 2)
            loss_pde = pde_beta * loss_pde_val
            self.log('train_loss_pde', loss_pde_val, prog_bar=True)
        else:
            self.log('train_loss_pde', 0.0, prog_bar=True)

        loss = loss_mse + loss_pde

        self.log('pde_beta', pde_beta, prog_bar=False)
        self.log('train_loss_mse', loss_mse, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_ctx, y_ctx = batch['x_context'], batch['y_context']
        x_tar, y_tar = batch['x_target'], batch['y_target']
        
        t = torch.rand(x_tar.shape[0], 1, device=self.device)
        y_1 = y_tar
        y_0 = torch.randn_like(y_1)
        t_b = t.unsqueeze(1)
        y_t = t_b * y_1 + (1 - t_b) * y_0
        v_target = y_1 - y_0
        
        v_pred = self.forward(x_ctx, y_ctx, x_tar, y_tar_noisy=y_t, t=t)
        loss = nn.functional.mse_loss(v_pred, v_target)
        self.log('val_loss', loss, prog_bar=True)
        return loss
        
    def sample(self, x_ctx, y_ctx, x_tar, steps=20):
        # Euler Integration
        B, N, _ = x_tar.shape
        y_t = torch.randn(B, N, self.dim_y, device=self.device)
        dt = 1.0 / steps
        
        for i in range(steps):
            t_curr = i * dt
            t_tensor = torch.full((B, 1), t_curr, device=self.device)
            
            v_pred = self.forward(x_ctx, y_ctx, x_tar, y_tar_noisy=y_t, t=t_tensor)
            y_t = y_t + v_pred * dt
            
        return y_t

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
