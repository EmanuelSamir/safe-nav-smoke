import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------
# COMPONENT 1: Perception Tokenizer (1D Causal Autoencoder)
# ----------------------------------------------------


class Tokenizer1DEncoder(nn.Module):
    def __init__(self, in_features=64, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),  # Tanh bottleneck for bounded latent space
        )

    def forward(self, x):
        """x: [Batch, Time, 64] -> z: [Batch, Time, 128]"""
        return self.net(x)


class Tokenizer1DDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, out_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_features),
            nn.Sigmoid(),  # Force values to [0,1] assuming density readings normalized
        )

    def forward(self, z):
        """z: [Batch, Time, 128] -> o_pred: [Batch, Time, 64]"""
        return self.net(z)


# ----------------------------------------------------
# TRANSFOMER PRIMITIVES (SwiGLU, RMSNorm, RoPE, QKNorm)
# ----------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def apply_rope(q, k, seq_len, head_dim):
    device = q.device
    freqs = 10000.0 ** -(torch.arange(0, head_dim, 2).float() / head_dim).to(device)
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # Broadcast heads/batches

    q_out = torch.view_as_real(q_complex * freqs_complex).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs_complex).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # QKNorm per architecture specification
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)

        # RoPE
        q, k = apply_rope(q, k, T, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Is_causal strictly limits the attention horizon (no future peeking)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ----------------------------------------------------
# COMPONENT 2: Latent Flow Dynamics (World Model Transformer)
# ----------------------------------------------------


class DynamicsTransformer(nn.Module):
    def __init__(self, latent_dim=128, action_dim=2, embed_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.latent_dim = latent_dim

        # We concatenate [action_t (2), tau (1), d (1), z_t (128)]
        in_dim = action_dim + 2 + latent_dim
        self.embed = nn.Linear(in_dim, embed_dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.out_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, latent_dim)

    def forward(self, z_state, action, tau, d):
        x_in = torch.cat([action, tau, d, z_state], dim=-1)
        x = self.embed(x_in)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return self.head(x)  # Prediction of z_clean


# ----------------------------------------------------
# FLOW MATCHING / SHORTCUT FORCING LOSS
# ----------------------------------------------------


class ShortcutForcingLoss(nn.Module):
    def __init__(self, d_min=0.01):
        super().__init__()
        self.d_min = d_min

    def get_ramp_weight(self, tau):
        # Escala: 0.9 * tau + 0.1
        return 0.9 * tau + 0.1

    def forward(self, model, z_clean, action):
        """X-Prediction Shortcut Forcing. Sequences processed in parallel."""
        B, T, D = z_clean.shape
        device = z_clean.device

        # 1. Sample Log-Normal Noise Tau and map strictly to [0, 1]
        # Standard flow matching schedule mapping Gaussian to logistic curve
        tau = torch.sigmoid(torch.randn(B, T, 1, device=device) * 1.2)

        # 50% chance for Base Flow vs Euler Bootstrap Path
        branch_mask = torch.rand(B, T, 1, device=device) < 0.5
        noise = torch.randn_like(z_clean)

        # ====== BRANCH 1: Base Flow Matching ======
        d_base = torch.full_like(tau, self.d_min)
        z_noisy_base = z_clean * (1 - tau) + noise * tau
        pred_base = model(z_noisy_base, action, tau, d_base)
        loss_base = F.mse_loss(pred_base, z_clean, reduction="none") * self.get_ramp_weight(tau)

        # ====== BRANCH 2: Euler Distillation Bootstrapping ======
        d_euler = torch.rand_like(tau) * (tau - self.d_min).clamp_min(0.0) + self.d_min
        z_noisy_big = z_clean * (1 - tau) + noise * tau
        pred_big = model(z_noisy_big, action, tau, d_euler)

        # Simulate Euler trajectory with stop-gradient `sg()`
        with torch.no_grad():
            half_d = d_euler / 2.0

            # Step A
            pred_step1 = model(z_noisy_big, action, tau, half_d)
            dir_1 = (z_noisy_big - pred_step1) / tau.clamp_min(1e-5)
            z_noisy_mid = z_noisy_big - dir_1 * half_d
            tau_mid = tau - half_d

            # Step B
            pred_step2 = model(z_noisy_mid, action, tau_mid, half_d)
            target_clean_euler = pred_step2  # Pseudo ground-truth target

        loss_euler = F.mse_loss(
            pred_big, target_clean_euler.detach(), reduction="none"
        ) * self.get_ramp_weight(tau)

        loss = torch.where(branch_mask, loss_base, loss_euler)
        return loss.mean()


# ----------------------------------------------------
# MODULE 4: The Latent Fluid World Model Orchestrator
# ----------------------------------------------------


class FluidWorldModelDreamer(pl.LightningModule):
    def __init__(self, action_dim=2, in_features=64):
        super().__init__()
        self.tokenizer_enc = Tokenizer1DEncoder(in_features=in_features)
        self.tokenizer_dec = Tokenizer1DDecoder(out_features=in_features)
        self.dynamics = DynamicsTransformer(latent_dim=128, action_dim=action_dim)
        self.shortcut_loss = ShortcutForcingLoss()

    def training_step(self, batch, batch_idx):
        # o_seq: [B, T, 64] (raw 1D lidar scans)
        # a_seq: [B, T, 2] actions representing [v, w]
        o_seq, a_seq = batch

        # 1. 1D Perception Compression (Bottleneck encoding)
        z_clean = self.tokenizer_enc(o_seq)

        # 2. Tokenizer Target Reconstruct (Autoencoding Loss)
        o_pred = self.tokenizer_dec(z_clean)
        L_tok = F.mse_loss(o_pred, o_seq)

        # 3. Flow Matching / Shortcut Forcing in the latent causal dimension
        L_shortcut = self.shortcut_loss(self.dynamics, z_clean.detach(), a_seq)

        loss = L_tok + L_shortcut

        self.log("train/L_tok", L_tok)
        self.log("train/L_shortcut", L_shortcut)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        o_seq, a_seq = batch
        B, T, _ = o_seq.shape

        z_clean = self.tokenizer_enc(o_seq)
        o_pred = self.tokenizer_dec(z_clean)
        L_tok = F.mse_loss(o_pred, o_seq)
        L_shortcut = self.shortcut_loss(self.dynamics, z_clean, a_seq)

        loss = L_tok + L_shortcut

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/L_tok", L_tok)
        self.log("val/L_shortcut", L_shortcut)

        # 1D Latent Rollout Visualization
        if batch_idx == 0 and self.logger is not None:
            # We want to see how well the model predicts the future from P frames of context
            # Context size: P frames
            P = min(4, T // 2)
            if P > 0 and T > P:
                z_context = z_clean[:1, :P, :]  # Shape: [1, P, 128]
                future_actions = a_seq[:1, P:, :]  # Shape: [1, T-P, 2]

                # Inference / Forecasting (The World Model simulating future states)
                z_imagined = self.rollout_imagination(z_context, future_actions, K=4)

                # Map imagined futures back to 1D lidar scans
                o_imagined = self.tokenizer_dec(z_imagined)

                # Convert to NumPy for plotting
                gt_scan = o_seq[0].detach().cpu().numpy()  # [T, 64]
                recon_scan = o_pred[0].detach().cpu().numpy()  # [T, 64]

                # Splice context + imagined future
                im_scan = np.zeros_like(gt_scan)
                im_scan[:P] = recon_scan[:P]  # The model 'remembers' the past
                im_scan[P:] = (
                    o_imagined[0].detach().cpu().numpy()
                )  # The model 'imagines' the future

                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                vmin, vmax = 0, 1  # Density arrays are normalized to [0, 1]

                im0 = axes[0].imshow(gt_scan, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                axes[0].set_title("Ground Truth 1D Scan (t=0 to T)")
                axes[0].set_ylabel("Time Step (t)")
                axes[0].set_xlabel("Lidar Ray (0-63)")

                im1 = axes[1].imshow(recon_scan, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                axes[1].set_title("Tokenizer Autoencoder (Bound)")
                axes[1].set_xlabel("Lidar Ray (0-63)")

                im2 = axes[2].imshow(im_scan, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                axes[2].set_title(f"Mental Rollout (Forecasting t>{P - 1})")
                axes[2].axhline(
                    y=P - 0.5,
                    color="white",
                    linestyle="--",
                    linewidth=2,
                    label="Imagination starts",
                )
                axes[2].set_xlabel("Lidar Ray (0-63)")
                axes[2].legend(loc="upper right")

                plt.tight_layout()

                tensorboard = self.logger.experiment
                tensorboard.add_figure("Val/1D_Mental_Rollout", fig, global_step=self.global_step)
                plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    def rollout_imagination(self, z_history, future_actions, K=4):
        """Real-time mental rollouts via K-step denoising flow.
        Provides high-precision physics simulation natively in latent space.
        z_history: [B, T_past, 128]
        future_actions: [B, T_future, 2]
        """
        B, T_past, D = z_history.shape
        _, T_future, _ = future_actions.shape
        device = z_history.device

        current_z_seq = z_history.clone()
        # Ensure identical time lengths by padding past actions with zeros if unavailable
        padded_acts = torch.cat(
            [torch.zeros(B, T_past, future_actions.shape[-1], device=device), future_actions], dim=1
        )

        imagined_latents = []

        for t in range(T_future):
            # 1. Generate absolute noise for initial simulation condition
            current_z = torch.randn(B, 1, D, device=device)
            current_tau = torch.ones(B, 1, 1, device=device)
            step_size = 1.0 / K

            context_z = torch.cat([current_z_seq, current_z], dim=1)
            context_a = padded_acts[:, : T_past + t + 1]

            # 2. Iterative Denoising (K refinement steps)
            for k in range(K):
                tau_seq = torch.zeros(B, context_z.size(1), 1, device=device)
                tau_seq[:, -1] = current_tau
                d_seq = torch.zeros(B, context_z.size(1), 1, device=device)
                d_seq[:, -1] = step_size

                # Predict abstract state
                pred_z_clean_seq = self.dynamics(context_z, context_a, tau_seq, d_seq)
                pred_clean_t = pred_z_clean_seq[:, -1:]

                # Euler Integration logic
                v = (context_z[:, -1:] - pred_clean_t) / current_tau.clamp_min(1e-5)
                current_tau = current_tau - step_size

                if current_tau.item() <= 1e-4:
                    current_z = pred_clean_t
                else:
                    current_z = context_z[:, -1:] - v * step_size

                context_z[:, -1:] = current_z

            imagined_latents.append(current_z)
            current_z_seq = torch.cat([current_z_seq, current_z], dim=1)

        return torch.cat(imagined_latents, dim=1)
