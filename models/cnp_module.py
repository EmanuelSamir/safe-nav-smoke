import torch
import torch.nn as nn
import pytorch_lightning as pl
from .common import DeterministicEncoder, Decoder, FourierFeatureEncoder, get_beta

class CNP(pl.LightningModule):
    def __init__(self, x_dim=1, y_dim=1, r_dim=128, hidden_dim=128, lr=1e-3,
                 pde_solver=None, pde_beta_schedule=None,
                 fourier_mapping_size=0, fourier_scale=10.0):
        super().__init__()
        self.save_hyperparameters(ignore=['pde_solver']) # Don't save solver object to hparams
        
        self.fourier_encoder = None
        enc_x_dim = x_dim
        
        if fourier_mapping_size > 0:
            self.fourier_encoder = FourierFeatureEncoder(x_dim, fourier_mapping_size, fourier_scale)
            enc_x_dim = x_dim + 2 * fourier_mapping_size
        
        self.encoder = DeterministicEncoder(enc_x_dim, y_dim, r_dim, hidden_dim)
        self.decoder = Decoder(enc_x_dim, r_dim, y_dim, hidden_dim)
        self.loss_fn = nn.MSELoss()
        
        self.pde_solver = pde_solver
        self.pde_beta_schedule = pde_beta_schedule or {'start': 0.0, 'end': 0.0, 'warmup': 0}
        
    def _encode_x(self, x):
        if self.fourier_encoder is not None:
            f = self.fourier_encoder(x)
            return torch.cat([x, f], dim=-1)
        return x
        
    def forward(self, x_context, y_context, x_target):
        x_ctx_enc = self._encode_x(x_context)
        x_tar_enc = self._encode_x(x_target)
        
        r = self.encoder(x_ctx_enc, y_context)
        y_pred = self.decoder(x_tar_enc, r)
        return y_pred
    
    def training_step(self, batch, batch_idx):
        x_ctx, y_ctx = batch['x_context'], batch['y_context']
        x_tar, y_tar = batch['x_target'], batch['y_target']
        
        beta_pde = get_beta(self.pde_beta_schedule, self.current_epoch)
            
        # Enable gradients on target coordinates for PINN loss (check dynamic weight too)
        # Check if schedule *ever* puts weight > 0 (e.g. end > 0 or start > 0) to enable grad
        # Or just check beta_pde > 0
        schedule_active = (self.pde_beta_schedule.get('end', 0) > 0) or (self.pde_beta_schedule.get('start', 0) > 0)
        
        if schedule_active and self.pde_solver is not None:
            x_tar.requires_grad_(True)
            
        y_pred = self(x_ctx, y_ctx, x_tar)
        
        mse_loss = self.loss_fn(y_pred, y_tar)
        loss = mse_loss
        
        if beta_pde > 0 and self.pde_solver is not None:
            # Flatten params or pass batch directly
            res = self.pde_solver.compute_residual(y_pred, x_tar, **batch)
            pde_loss = torch.mean(res**2)
            loss = loss + beta_pde * pde_loss
            self.log('train_pde_loss', pde_loss, prog_bar=True)
        else:
            self.log('train_pde_loss', 0.0, prog_bar=True)
        
        self.log('beta_pde', beta_pde, prog_bar=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse', mse_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_ctx, y_ctx = batch['x_context'], batch['y_context']
        x_tar, y_tar = batch['x_target'], batch['y_target']
            
        y_pred = self(x_ctx, y_ctx, x_tar)
        loss = self.loss_fn(y_pred, y_tar)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
