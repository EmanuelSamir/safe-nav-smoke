import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset, Subset

# Internal import
from .models import PDDShortcutPolicy, PDDStudentEncoder, PDDTeacherEncoder


class PDDOfflineDataset(Dataset):
    def __init__(self, dataset_path, window_size=10, horizon=15, action_dim=2, goal_workaround=True):
        self.ds = load_from_disk(dataset_path)
        self.window_size = window_size
        self.horizon = horizon
        self.action_dim = action_dim
        self.goal_workaround = goal_workaround

        self.episode_indices = {}
        for idx, row in enumerate(self.ds):
            ep_id = row["episode_id"]
            if ep_id not in self.episode_indices:
                self.episode_indices[ep_id] = []
            self.episode_indices[ep_id].append(idx)

        self.valid_indices = []
        for ep_id, indices in self.episode_indices.items():
            if len(indices) > window_size:
                for i in range(window_size, len(indices)):
                    self.valid_indices.append((ep_id, i))

        self.episode_goals = {}
        if goal_workaround:
            for ep_id, indices in self.episode_indices.items():
                last_idx = indices[-1]
                last_row = self.ds[last_idx]
                self.episode_goals[ep_id] = np.array(last_row["state"][:2])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ep_id, step_idx = self.valid_indices[idx]
        hist_indices = self.episode_indices[ep_id][step_idx - self.window_size : step_idx + 1]
        window_rows = [self.ds[i] for i in hist_indices]
        curr_row = window_rows[-1]
        curr_pose = np.array(curr_row["state"])

        # Expert trajectory as FUTURE RELATIVE POSES (Waypoints)
        # Instead of actions, we predict [dx, dy, dtheta] in local frame
        ep_steps = self.episode_indices[ep_id]
        future_traj = []
        cos_th = np.cos(-curr_pose[2])
        sin_th = np.sin(-curr_pose[2])
        
        for i in range(1, self.horizon + 1):
            target_idx = step_idx + i
            if target_idx < len(ep_steps):
                row = self.ds[ep_steps[target_idx]]
                f_pose = np.array(row["state"])
                dx_f = f_pose[0] - curr_pose[0]
                dy_f = f_pose[1] - curr_pose[1]
                
                rel_x = dx_f * cos_th - dy_f * sin_th
                rel_y = dx_f * sin_th + dy_f * cos_th
                rel_theta = (f_pose[2] - curr_pose[2] + np.pi) % (2 * np.pi) - np.pi
                # Using [dx, dy, sin(th), cos(th)] for continuous orientation
                future_traj.append([rel_x, rel_y, np.sin(rel_theta), np.cos(rel_theta)])
            else:
                # Pad with last valid waypoint
                future_traj.append(future_traj[-1] if len(future_traj) > 0 else [0.0, 0.0, 0.0, 1.0])

        # Goal calculation
        if "goal" in curr_row:
            goal_pos = np.array(curr_row["goal"])
        else:
            goal_pos = self.episode_goals.get(ep_id, curr_pose[:2])

        dx_g = goal_pos[0] - curr_pose[0]
        dy_g = goal_pos[1] - curr_pose[1]
        dist_g = np.linalg.norm([dx_g, dy_g])
        angle_to_goal = np.arctan2(dy_g, dx_g) - curr_pose[2]
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        goal_rel = np.array([dist_g / 10.0, np.sin(angle_to_goal), np.cos(angle_to_goal)])

        obs_seq, map_seq, rel_poses = [], [], []
        for r in window_rows:
            obs_seq.append(r["observation"])
            map_seq.append(r["local_map"])
            rp = np.array(r["state"])
            dx = rp[0] - curr_pose[0]
            dy = rp[1] - curr_pose[1]
            cos_th = np.cos(-curr_pose[2])
            sin_th = np.sin(-curr_pose[2])
            rel_x = dx * cos_th - dy * sin_th
            rel_y = dx * sin_th + dy * cos_th
            rel_theta = (rp[2] - curr_pose[2] + np.pi) % (2 * np.pi) - np.pi
            rel_poses.append([rel_x, rel_y, rel_theta])

        return {
            "obs": torch.tensor(obs_seq, dtype=torch.float32),
            "local_map": torch.tensor(map_seq, dtype=torch.float32).unsqueeze(1),
            "expert_traj": torch.tensor(future_traj, dtype=torch.float32),
            "prev_action": torch.tensor(
                curr_row["prev_action"] if ("prev_action" in curr_row and len(curr_row["prev_action"]) == self.action_dim) 
                else np.array([0.0, 0.0, 0.0, 1.0]) if self.action_dim == 4 else np.zeros(self.action_dim),
                dtype=torch.float32,
            ),
            "goal_rel": torch.tensor(goal_rel, dtype=torch.float32),
            "heading": torch.tensor([np.sin(curr_pose[2]), np.cos(curr_pose[2])], dtype=torch.float32),
            "rel_poses": torch.tensor(rel_poses, dtype=torch.float32),
        }


class PDDShortcutModule(pl.LightningModule):
    def __init__(
        self,
        latent_dim=64,
        action_dim=4,
        horizon=15,
        lr=1e-4,
        ema_decay=0.999,
        M=128,
        sc_ratio=0.25,
        training_mode="teacher_pretrain",
        traj_scale=4.0,
        train_prob_self_cond=0.9,
        min_snr_loss_weight=True,
        min_snr_gamma=5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.teacher = PDDTeacherEncoder(latent_dim)
        self.student = PDDStudentEncoder(latent_dim)
        self.policy = PDDShortcutPolicy(latent_dim, action_dim, horizon=horizon)
        self.policy_ema = copy.deepcopy(self.policy)
        for param in self.policy_ema.parameters():
            param.requires_grad = False
        self.lr = lr
        self.ema_decay = ema_decay
        self.M = M
        self.sc_ratio = sc_ratio
        self.training_mode = training_mode
        self.traj_scale = traj_scale

        if self.training_mode == "distill":
            # Freeze teacher in distillation stage
            self.teacher.requires_grad_(False)

    def update_ema(self):
        with torch.no_grad():
            for p, p_ema in zip(self.policy.parameters(), self.policy_ema.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def forward(self, obs, maps, prev_a, goal_rel, heading, rel_poses):
        """Basic Diffusion (Flow Matching) Inference.
        Since we no longer use Shortcut Consistency, we use more steps for quality.
        """
        # Using 8 steps for a good balance between speed and quality
        return self.multi_step_forward(obs, maps, prev_a, goal_rel, heading, rel_poses, num_steps=8)

    def multi_step_forward(self, obs, maps, prev_a, goal_rel, heading, rel_poses, num_steps=2):
        """Inference with multiple smaller jumps (e.g. 2 steps of d=0.5)"""
        B = obs.shape[0]
        if self.training_mode == "teacher_pretrain" and maps is not None:
            # Oracle Inference: Use Teacher tokens
            Tw = maps.shape[1]
            t_tokens = self.teacher(maps.view(B * Tw, 1, 74, 74))
            mu = t_tokens.view(B, Tw, -1, self.hparams.latent_dim)
            sigma = torch.zeros_like(mu)
        else:
            # Student Inference
            Tw = obs.shape[1]
            s_mu, s_log_sigma = self.student(obs.view(B * Tw, 1, 64))
            mu = s_mu.view(B, Tw, -1, self.hparams.latent_dim)
            sigma = torch.exp(s_log_sigma).view(mu.shape)
            
        xt = torch.randn(B, self.hparams.horizon, self.hparams.action_dim, device=self.device)
        d_step = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * d_step, device=self.device)
            d = torch.zeros(B, device=self.device) # Using d=0 for basic flow matching
            
            # During inference, we can also use self-conditioning if we track x0
            # For now, let's keep it simple and pass None or track the last estimate
            v_pred = self.policy_ema(xt, t, d, goal_rel, prev_a, heading, mu, sigma, rel_poses, self_cond=None)
            xt = xt + v_pred * d_step
            
        # Un-scale to return to physical units
        return xt / self.traj_scale

    def training_step(self, batch, batch_idx):
        obs, maps, expert_traj, prev_a, goal_r, heading, rel_poses = (
            batch["obs"],
            batch["local_map"],
            batch["expert_traj"],
            batch["prev_action"],
            batch["goal_rel"],
            batch["heading"],
            batch["rel_poses"],
        )
        B, Hp1, _ = obs.shape
        H = Hp1 - 1

        # 1. Latent Representations
        # Teacher: Oracle (2D maps)
        t_tokens = self.teacher(maps.view(B * (H + 1), 1, 74, 74)).view(
            B, H + 1, -1, self.hparams.latent_dim
        )
        
        # Student: Partial (1D readings)
        s_mu_all, s_log_sigma_all = self.student(obs.view(B * (H + 1), 1, 64))
        s_mu = s_mu_all.view(B, H + 1, -1, self.hparams.latent_dim)
        s_log_sigma = s_log_sigma_all.view(B, H + 1, -1, self.hparams.latent_dim)
        s_sigma = torch.exp(s_log_sigma)

        # 2. Decide Conditioning Tokens based on Mode
        if self.training_mode == "teacher_pretrain":
            # Train policy + teacher on 2D maps
            # In pre-train, we treat t_tokens as the latent condition
            cond_mu = t_tokens
            cond_sigma = torch.zeros_like(s_sigma) # Teacher is deterministic
            mode_prefix = "teacher_"
        else:
            # Train student to match teacher, and policy to work with student
            cond_mu = s_mu
            cond_sigma = s_sigma
            mode_prefix = ""

        # 3. Basic Flow-Matching Logic (Diffusion)
        # Apply traj_scale to expert trajectories
        expert_scaled = expert_traj * self.traj_scale
        
        a0 = torch.randn_like(expert_traj)
        t = torch.rand(B, device=self.device)
        
        # Linear interpolation (Optimal Transport Flow)
        xt = (1 - t)[:, None, None] * a0 + t[:, None, None] * expert_scaled
        v_target = expert_scaled - a0
        
        # 4. Self-Conditioning logic (90% of the time during training)
        self_cond = None
        if torch.rand(1) < self.hparams.train_prob_self_cond:
            with torch.no_grad():
                # Pass 1: generate estimate of x1 using EMA policy for stability
                v_initial = self.policy_ema(
                    xt, t, torch.zeros_like(t), goal_r, prev_a, heading, cond_mu, cond_sigma, rel_poses
                )
                # In Flow Matching, predicted x1 = xt + v * (1 - t)
                self_cond = xt + v_initial * (1.0 - t)[:, None, None]
                self_cond = self_cond.detach()

        # Predict velocity field
        v_pred = self.policy(
            xt,
            t,
            torch.zeros_like(t),
            goal_r,
            prev_a,
            heading,
            cond_mu,
            cond_sigma,
            rel_poses,
            self_cond=self_cond
        )
        
        # 5. Loss with Min-SNR weighting
        loss = F.mse_loss(v_pred, v_target, reduction="none")
        loss = loss.mean(dim=[1, 2]) # Mean over horizon and action_dim
        
        if self.hparams.min_snr_loss_weight:
            # SNR for Flow Matching: t^2 / (1-t)^2
            # Clamp t to avoid division by zero
            t_clamped = torch.clamp(t, 0.001, 0.999)
            snr = (t_clamped**2) / ((1 - t_clamped)**2)
            mse_loss_weight = torch.clamp(snr, max=self.hparams.min_snr_gamma) / snr
            loss = loss * mse_loss_weight

        fm_loss = loss.mean()
        
        # Shortcut Logic (Commented out per user request)
        # sc_loss = self.compute_sc_loss(...)
        sc_loss = torch.tensor(0.0, device=self.device)

        # Part C: Distillation Loss (KL-like for student latents)
        distill_loss = torch.tensor(0.0, device=self.device)
        if self.training_mode == "distill":
            eps = 1e-4
            distill_loss = torch.mean(
                ((s_mu - t_tokens) ** 2 + s_sigma**2) / (2 * eps**2) - s_log_sigma
            )

        # 4. Total Loss (Now just Flow Matching)
        total_loss = fm_loss + 0.1 * distill_loss
        
        self.log(f"train/{mode_prefix}total_loss", total_loss, prog_bar=True)
        self.log(f"train/{mode_prefix}fm_loss", fm_loss)
        if self.training_mode == "distill":
            self.log("train/distill_loss", distill_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        obs, maps, expert_traj, prev_a, goal_r, heading, rel_poses = (
            batch["obs"],
            batch["local_map"],
            batch["expert_traj"],
            batch["prev_action"],
            batch["goal_rel"],
            batch["heading"],
            batch["rel_poses"],
        )
        # 1. Prediction based on mode
        # Pre-train: Show the best teacher can do (M steps)
        # Distill: Show the one-step shortcut goal
        n_steps = self.hparams.M if self.training_mode == "teacher_pretrain" else 1
        pred_traj = self.multi_step_forward(obs, maps, prev_a, goal_r, heading, rel_poses, num_steps=n_steps)
        
        # 2. Consistency Check (always 2 steps vs 1 step to measure SC error)
        pred_1s = self.multi_step_forward(obs, maps, prev_a, goal_r, heading, rel_poses, num_steps=1)
        pred_2s = self.multi_step_forward(obs, maps, prev_a, goal_r, heading, rel_poses, num_steps=2)
        
        val_loss = F.mse_loss(pred_traj, expert_traj)
        sc_discrepancy = F.mse_loss(pred_1s, pred_2s)
        
        mode_prefix = "teacher_" if self.training_mode == "teacher_pretrain" else ""
        self.log(f"val/{mode_prefix}total_loss", val_loss, prog_bar=True)
        self.log(f"val/{mode_prefix}sc_discrepancy", sc_discrepancy)
        
        if batch_idx == 0:
            idx = torch.randint(0, expert_traj.size(0), (1,)).item()
            label_pred = f"{n_steps}-step"
            self.visualize_trajectory(
                expert_traj[idx], pred_traj[idx], pred_2s[idx], self.global_step, label_pred
            )
        return val_loss

    def visualize_trajectory(self, expert, pred_main, pred_sc, step, label_pred="Prediction"):
        try:
            import matplotlib.pyplot as plt

            expert = expert.detach().cpu().numpy()
            pred_main = pred_main.detach().cpu().numpy()
            pred_sc = pred_sc.detach().cpu().numpy()
            
            # Subplots: X, Y, Theta (reconstructed), and 2D Path
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 1. Delta X
            axes[0].plot(expert[:, 0], "g-", label="Expert", alpha=0.8)
            axes[0].plot(pred_main[:, 0], "b--", label=label_pred)
            axes[0].plot(pred_sc[:, 0], "r:", label="2-step SC Check", alpha=0.6)
            axes[0].set_title(f"Relative X - Step {step}")
            
            # 2. Delta Y
            axes[1].plot(expert[:, 1], "g-", label="Expert", alpha=0.8)
            axes[1].plot(pred_main[:, 1], "b--", label=label_pred)
            axes[1].plot(pred_sc[:, 1], "r:", label="2-step SC Check", alpha=0.6)
            axes[1].set_title("Relative Y")
            
            # 3. Angle Theta (Reconstructed from sin/cos)
            def get_theta(traj):
                return np.arctan2(traj[:, 2], traj[:, 3])
            
            axes[2].plot(get_theta(expert), "g-", label="Expert", alpha=0.8)
            axes[2].plot(get_theta(pred_main), "b--", label=label_pred)
            axes[2].plot(get_theta(pred_sc), "r:", label="2-step SC Check", alpha=0.6)
            axes[2].set_title("Relative Orientation (rad)")
            
            # 4. 2D Path (X vs Y)
            axes[3].plot(expert[:, 0], expert[:, 1], "g-", label="Expert", alpha=0.8)
            axes[3].plot(pred_main[:, 0], pred_main[:, 1], "b--", label=label_pred)
            axes[3].plot(pred_sc[:, 0], pred_sc[:, 1], "r:", label="2-step SC Check", alpha=0.6)
            axes[3].set_aspect('equal')
            axes[3].set_title("2D local path (m)")
            
            for ax in axes:
                ax.legend()
                ax.grid(True, alpha=0.3)

            if self.logger and hasattr(self.logger.experiment, "add_figure"):
                self.logger.experiment.add_figure("val/trajectory_consistency", fig, global_step=step)

            plt.close(fig)
        except Exception as e:
            print(f"Error in visualization: {e}")
            pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_ema()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)


class PDDDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32, window_size=10, horizon=15, action_dim=2, val_split=0.1):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.horizon = horizon
        self.action_dim = action_dim
        self.val_split = val_split

    def setup(self, stage=None):
        full_ds = PDDOfflineDataset(
            self.dataset_path, window_size=self.window_size, horizon=self.horizon, action_dim=self.action_dim
        )
        indices = np.arange(len(full_ds))
        val_size = int(len(full_ds) * self.val_split)
        train_size = len(full_ds) - val_size
        np.random.seed(42)
        np.random.shuffle(indices)
        self.train_ds = Subset(full_ds, indices[:train_size])
        self.val_ds = Subset(full_ds, indices[train_size:])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)
