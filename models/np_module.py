import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.distributions import Normal, kl_divergence
from .common import LatentEncoder, Decoder, FourierFeatureEncoder, get_beta

class NP(pl.LightningModule):
    def __init__(self, x_dim=1, y_dim=1, z_dim=128, hidden_dim=128, lr=1e-3,
                 pde_solver=None,
                 kl_beta_schedule=None, pde_beta_schedule=None,
                 fourier_mapping_size=0, fourier_scale=10.0):
        super().__init__()
        self.save_hyperparameters(ignore=['pde_solver'])
        
        self.fourier_encoder = None
        enc_x_dim = x_dim
        
        if fourier_mapping_size > 0:
            self.fourier_encoder = FourierFeatureEncoder(x_dim, fourier_mapping_size, fourier_scale)
            # Input to networks becomes x (original) + fourier features (2 * mapping_size)
            enc_x_dim = x_dim + 2 * fourier_mapping_size
            
        self.latent_encoder = LatentEncoder(enc_x_dim, y_dim, z_dim, hidden_dim)
        # Decoder inputs: x (enc_x_dim) + z (z_dim) -> y_dim
        self.decoder = Decoder(enc_x_dim, z_dim, y_dim, hidden_dim)
        
        self.pde_solver = pde_solver
        
        # Default schedules if None (constant)
        self.kl_beta_schedule = kl_beta_schedule or {'start': 0.01, 'end': 0.01, 'warmup': 0}
        self.pde_beta_schedule = pde_beta_schedule or {'start': 0.0, 'end': 0.0, 'warmup': 0}
        
    def _encode_x(self, x):
        if self.fourier_encoder is not None:
            f = self.fourier_encoder(x)
            return torch.cat([x, f], dim=-1)
        return x

    def forward(self, x_context, y_context, x_target, num_samples=1):
        # Encode inputs if fourier enabled
        x_ctx_enc = self._encode_x(x_context)
        x_tar_enc = self._encode_x(x_target)
        
        # Inference mode: encode context -> sample z -> decode
        mu, sigma = self.latent_encoder(x_ctx_enc, y_context)
        q_z = Normal(mu, sigma)
        
        # Sample z (can support multiple samples for uncertainty estimation)
        # z shape: (B, num_samples, z_dim) if we wanted multiple, but Decoder expects (B, z_dim).
        z = mu 
        if self.training or num_samples > 1:
             z = q_z.rsample()
        
        # Decode
        y_pred = self.decoder(x_tar_enc, z)
        return y_pred, mu, sigma

    def training_step(self, batch, batch_idx):
        x_ctx, y_ctx = batch['x_context'], batch['y_context']
        x_tar, y_tar = batch['x_target'], batch['y_target']
            
        # Enable grads for PINN
        schedule_active = (self.pde_beta_schedule.get('end', 0) > 0) or (self.pde_beta_schedule.get('start', 0) > 0)
        
        if schedule_active:
            x_tar.requires_grad_(True)
        
        # Encode inputs
        x_ctx_enc = self._encode_x(x_ctx)
        x_tar_enc = self._encode_x(x_tar)
            
        # 1. Encode context -> q(z|ctx)
        mu_ctx, sigma_ctx = self.latent_encoder(x_ctx_enc, y_ctx)
        q_ctx = Normal(mu_ctx, sigma_ctx)
        
        # 2. Encode target (all) -> q(z|target)
        mu_tar, sigma_tar = self.latent_encoder(x_tar_enc, y_tar)
        q_tar = Normal(mu_tar, sigma_tar)
        
        # 3. Sample z from q_tar (for training reconstruction)
        z_sample = q_tar.rsample()
        
        # 4. Decode
        y_pred_mean = self.decoder(x_tar_enc, z_sample)

        # Reconstruction Loss = Mean Squared Error
        mse_loss = nn.functional.mse_loss(y_pred_mean, y_tar, reduction='mean')
        
        # 5. KL Divergence KL(q_tar || q_ctx)
        kl = kl_divergence(q_tar, q_ctx).mean()
        
        # Get dynamic betas
        beta_kl = get_beta(self.kl_beta_schedule, self.current_epoch)
        
        # Loss
        loss = mse_loss + beta_kl * kl 
        
        pde_loss = torch.tensor(0.0, device=self.device)
        beta_pde = get_beta(self.pde_beta_schedule, self.current_epoch)

        if beta_pde > 0 and self.pde_solver is not None:
             res = self.pde_solver.compute_residual(y_pred_mean, x_tar, **batch)
             pde_loss = torch.mean(res**2)
             loss = loss + beta_pde * pde_loss
             self.log('train_pde_loss', pde_loss, prog_bar=True)
        else:
            self.log('train_pde_loss', 0.0, prog_bar=True)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mse', mse_loss, prog_bar=True)
        self.log('train_kl', kl, prog_bar=True)
        self.log('beta_kl', beta_kl, prog_bar=False)
        self.log('beta_pde', beta_pde, prog_bar=False)
        
        # Monitor for posterior collapse
        self.log('z_mu_mean', mu_tar.mean(), prog_bar=False)
        self.log('z_sigma_mean', sigma_tar.mean(), prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_ctx, y_ctx = batch['x_context'], batch['y_context']
        x_tar, y_tar = batch['x_target'], batch['y_target']
        
        x_ctx_enc = self._encode_x(x_ctx)
        x_tar_enc = self._encode_x(x_tar)
            
        # Validation: Use context only to sample z
        mu_ctx, sigma_ctx = self.latent_encoder(x_ctx_enc, y_ctx)
        z = mu_ctx # Use mean for best guess or sample? Standard val uses mean usually.
        y_pred = self.decoder(x_tar_enc, z)
        loss = nn.functional.mse_loss(y_pred, y_tar)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
