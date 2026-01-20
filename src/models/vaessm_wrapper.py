import torch
import numpy as np
from learning.vaessm import ScalarFieldVAESSM, VAESSMParams, ObsVAESSM, VAEDecoder, VAEEncoder
from learning.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class LegacyVAEDecoder(VAEDecoder):
    def __init__(self, vaessm_params: VAESSMParams):
        super().__init__(vaessm_params)
        # Override z_dim to include deterministic state
        self.z_dim = vaessm_params.stoch_dim + vaessm_params.deter_dim
        
        # Re-initialize layers with new input dimension
        layers = [nn.Linear(self.z_dim + self.x_dim, self.h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h_dim, self.h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        # Output layers remain the same size-wise (h_dim -> y_dim)

class LegacyScalarFieldVAESSM(ScalarFieldVAESSM):
    def __init__(self, params: VAESSMParams):
        super().__init__(params)
        # Replace decoder with legacy version
        self.decoder = LegacyVAEDecoder(params)

    def forward(self, prev_h, prev_z, prev_action, dones, obs=None, query_obs=None):
        # Copy-paste of original forward but with cat([h, z]) for decoder
        
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        mask = 1.0 - dones.float()
        prev_h = prev_h * mask
        prev_z = prev_z * mask

        rnn_input = torch.cat([prev_z, prev_action], dim=-1)
        h = self.gru(rnn_input, prev_h)
        
        prior_out = self.prior_net(h)
        prior_mu = self.prior_net_mu(prior_out)
        prior_sigma = 0.9*F.softplus(self.prior_net_sigma(prior_out)) + 0.1
        prior_dist = torch.distributions.Normal(prior_mu, prior_sigma)

        if obs is not None:
            embed = self.encoder(obs)
            embed_sample = embed.rsample()
            
            post_in = torch.cat([h, embed_sample], dim=1)
            post_out = self.posterior_net(post_in)
            post_mu = self.posterior_net_mu(post_out)
            post_sigma = 0.9*F.softplus(self.posterior_net_sigma(post_out)) + 0.1
            post_dist = torch.distributions.Normal(post_mu, post_sigma)
            z = post_dist.rsample()
        else:
            post_dist = None
            z = prior_dist.rsample()

        decoded_dist = None
        if query_obs is not None:
            # Concatenate h and z for legacy decoder
            latents = torch.cat([h, z], dim=1)
            decoded_dist = self.decoder(latents, query_obs)
            
        return h, z, prior_dist, post_dist, decoded_dist

class VAESSMWrapper(BaseModel):
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        
        # Params matching the checkpoint (128 dim)
        params = VAESSMParams(
            r_dim=128,
            embed_dim=128,
            deter_dim=128,
            stoch_dim=128,
            encoder_hidden_dim=128,
            prior_hidden_dim=128,
            posterior_hidden_dim=128,
            decoder_hidden_dim=128
        )
        # Use Legacy model
        self.model = LegacyScalarFieldVAESSM(params=params).to(device)
        
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded VAESSM checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            raise

        self.model.eval()
        self.prev_h = None
        self.prev_z = None
        
        # Initialize states
        self.reset_states()

    def reset_states(self):
        self.prev_h = torch.zeros(1, self.model.params.deter_dim).to(self.device)
        self.prev_z = torch.zeros(1, self.model.params.stoch_dim).to(self.device)

    def forward(self, prev_h, prev_z, action_step, pos_obs, value_obs=None):
        # Ensure inputs are on device
        action_step = action_step.to(self.device)
        
        if value_obs is None:
            # Prediction mode
            obs_step = ObsVAESSM(
                xs=torch.tensor(pos_obs[:, 0], dtype=torch.float32).unsqueeze(0).to(self.device),
                ys=torch.tensor(pos_obs[:, 1], dtype=torch.float32).unsqueeze(0).to(self.device),
            )
            h, z, prior_dist, post_dist, pred_dist = self.model(
                prev_h=prev_h,
                prev_z=prev_z,
                prev_action=action_step,
                dones=torch.tensor([False]).to(self.device),
                query_obs=obs_step
            )
        else:
            # Inference mode (update with observation)
            obs_step = ObsVAESSM(
                xs=torch.tensor(pos_obs[:, 0], dtype=torch.float32).unsqueeze(0).to(self.device),
                ys=torch.tensor(pos_obs[:, 1], dtype=torch.float32).unsqueeze(0).to(self.device),
                values=torch.tensor(value_obs, dtype=torch.float32).squeeze().unsqueeze(0).to(self.device),
            )
            h, z, prior_dist, post_dist, pred_dist = self.model(
                prev_h=prev_h,
                prev_z=prev_z,
                prev_action=action_step,
                dones=torch.tensor([False]).to(self.device),
                obs=obs_step,
            )

        return h, z, prior_dist, post_dist, pred_dist

    def set_prev_states(self, h, z):
        self.prev_h = h
        self.prev_z = z

    def predict(self, pos_obs):
        # This method matches the signature expected by RiskMapBuilder (if it uses .predict)
        # But RiskMapBuilder in experiment_6 calls builder.build_map(learner, ...)
        # And build_map calls learner.predict(pos_obs) usually?
        # Let's check RiskMapBuilder.build_map implementation later.
        
        action_vec = torch.zeros(1, self.model.params.action_dim).to(self.device)

        h, z, prior_dist, post_dist, pred_dist = self.forward(self.prev_h, self.prev_z, action_vec, pos_obs)
        
        mean = pred_dist.mean.detach().cpu().numpy()
        std = pred_dist.stddev.detach().cpu().numpy()
        
        # Update internal state? 
        # In experiment_6, predict updates state: self.set_prev_states(h, z)
        self.set_prev_states(h, z)
        
        return mean, std

    def update(self):
        pass
