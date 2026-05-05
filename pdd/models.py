import numpy as np
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PDDTeacherEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=0),   # 74 -> 37
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 37 -> 19
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 19 -> 10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=0), # 10 -> 4
            # Output is (B, latent_dim, 4, 4)
        )

    def forward(self, x):
        feat = self.cnn(x)
        tokens = feat.flatten(2).permute(0, 2, 1)
        return tokens


class PDDStudentEncoder(nn.Module):
    def __init__(self, latent_dim=64, num_tokens=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.token_heads = nn.Linear(128 * 8, num_tokens * latent_dim * 2)

    def forward(self, x):
        feat = self.cnn(x).flatten(1)
        out = self.token_heads(feat).view(-1, self.num_tokens, self.latent_dim, 2)
        mu = out[..., 0]
        log_sigma = out[..., 1]
        return mu, log_sigma


class PDDShortcutPolicy(nn.Module):
    def __init__(self, latent_dim=64, action_dim=2, hidden_dim=256, horizon=15):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.jump_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Query input: ([traj_noisy, self_cond, goal, prev_a, heading, embs])
        # self_cond has size (horizon * action_dim)
        q_size = (horizon * action_dim) * 2 + 3 + action_dim + 2 + hidden_dim * 2
        self.q_proj = nn.Linear(q_size, hidden_dim)
        self.q_ln = nn.LayerNorm(hidden_dim)

        # Transformer for map tokens (Self-Attention)
        self.kv_proj = nn.Linear(latent_dim * 2 + 3, hidden_dim)
        self.kv_ln = nn.LayerNorm(hidden_dim)
        
        self.map_sa = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True
        )
        self.map_norm = nn.LayerNorm(hidden_dim)

        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_dim)

        # Decoder outputs the full trajectory (Deeper)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(
        self,
        action_noisy,
        t,
        d,
        goal_rel,
        prev_action,
        heading,
        tokens_mu,
        tokens_sigma,
        rel_poses,
        self_cond=None,
    ):
        """action_noisy: (B, horizon, action_dim)
        self_cond: (B, horizon, action_dim) or None
        t, d: (B,)
        goal_rel: (B, 3)
        prev_action: (B, action_dim)
        heading: (B, 2) [sin, cos]
        """
        B, H, K, D = tokens_mu.shape
        if self_cond is None:
            self_cond = torch.zeros_like(action_noisy)

        # 1. Embeddings
        t_emb = self.time_mlp(t)
        d_emb = self.jump_mlp(d)

        # 2. Build Query
        traj_flat = action_noisy.reshape(B, -1)
        self_cond_flat = self_cond.reshape(B, -1)
        q_input = torch.cat([traj_flat, self_cond_flat, goal_rel, prev_action, heading, t_emb, d_emb], dim=-1)
        query = self.q_ln(self.q_proj(q_input)).unsqueeze(1)

        # 3. Build Keys/Values from history + Map Self-Attention
        rel_poses_expanded = rel_poses.unsqueeze(2).expand(-1, -1, K, -1)
        tokens_combined = torch.cat([tokens_mu, tokens_sigma, rel_poses_expanded], dim=-1)
        kv_input_flat = tokens_combined.view(B, H * K, -1)
        
        # Project tokens to hidden dim and apply self-attention
        kv_feats = self.kv_ln(self.kv_proj(kv_input_flat))
        key_val_tokens = self.map_sa(kv_feats)
        key_val_tokens = self.map_norm(key_val_tokens)
        
        # Use same tokens for key/value or split them
        key = key_val_tokens
        value = key_val_tokens

        # 4. Attention
        attn_out, _ = self.mha(query, key, value)
        attn_out = self.attn_ln(attn_out + query) # Residual + Norm

        # 5. Decode to Trajectory Velocity
        velocity_flat = self.decoder(attn_out.squeeze(1))  # (B, horizon*action_dim)
        velocity = velocity_flat.view(B, self.horizon, self.action_dim)

        return velocity
