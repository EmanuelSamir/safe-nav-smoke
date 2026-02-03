"""
Unified dataset classes for all models.
Clean, standardized structure.
"""
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Tuple, Optional

from src.models.shared.observations import Obs

class BaseSmokeDataset(Dataset):
    """Base class for handling the smoke_data.npz loading and splitting."""
    def __init__(self, 
                 data_path: str,
                 train_split: float = 0.8,
                 mode: str = 'train',
                 max_episodes: Optional[int] = None):
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found at {data_path}")
            
        loader = np.load(data_path)
        self.smoke_data = loader['smoke_data'] # (E, T, H, W)
        self.dt = float(loader.get('dt', 0.1))
        self.x_size = float(loader.get('x_size', 50.0))
        self.y_size = float(loader.get('y_size', 50.0))
        self.res = float(loader.get('resolution', 1.0))
        
        self.n_episodes, self.n_steps, self.H, self.W = self.smoke_data.shape
        
        # Train/Val split
        split_idx = int(self.n_episodes * train_split)
        if mode == 'train':
            self.smoke_data = self.smoke_data[:split_idx]
            self.n_episodes = split_idx
        elif mode == 'val':
            self.smoke_data = self.smoke_data[split_idx:]
            self.n_episodes = self.n_episodes - split_idx
            
        if max_episodes is not None:
             self.n_episodes = min(self.n_episodes, max_episodes)
             self.smoke_data = self.smoke_data[:self.n_episodes]

    def __len__(self):
        return self.n_episodes

class NonSequentialDataset(BaseSmokeDataset):
    """
    Dataset for non-sequential models.
    Returns (Context, Target) sets of (x,y,t,s) points.
    Sample ratio is used for spatial sampling, but temporal sampling
    is fixed by the sequence length.
    ctx_points_ratios: (min_ratio, max_ratio) for context points
    trg_points_ratio: ratio for target points
    """
    def __init__(self, 
                 data_path: str, 
                 sequence_length: int = 25,
                 ctx_points_ratios: tuple[float, float] = (0.1, 0.2),
                 trg_points_ratio: float = 0.5,
                 train_split: float = 0.8,
                 mode: str = 'train',
                 max_episodes: Optional[int] = None):
        
        super().__init__(data_path, train_split, mode, max_episodes)
        
        self.ctx_min_pts = int(self.H * self.W * ctx_points_ratios[0])
        self.ctx_max_pts = int(self.H * self.W * ctx_points_ratios[1])
        self.trg_pts = int(self.H * self.W * trg_points_ratio)
        self.sequence_length = sequence_length
        assert self.sequence_length < self.n_steps, f"Sequence length {self.sequence_length} must be less than number of steps {self.n_steps}"
        assert trg_points_ratio > max(ctx_points_ratios), f"Target points ratio must be greater than max context points ratio"

        self.max_diff_time = sequence_length * self.dt

    def __getitem__(self, idx: int) -> Tuple[Obs, Obs, float]:
        episode = self.smoke_data[idx]
        
        # 1. Time Window
        max_start = self.n_steps - self.sequence_length
        t_start_idx = np.random.randint(0, max_start)
        t_offset = t_start_idx * self.dt
        
        # Indices for context and target
        frame_indices = np.arange(t_start_idx, t_start_idx + self.sequence_length)
        
        # 2. Sample points using Relative Time (t - t0)
        num_ctx_pts = np.random.randint(self.ctx_min_pts, self.ctx_max_pts)
        ctx_obs = self._sample_points(episode, frame_indices, num_ctx_pts, t_offset=t_offset)
        trg_obs = self._sample_points(episode, frame_indices, self.trg_pts, t_offset=t_offset)
        
        return ctx_obs, trg_obs, t_offset

    def _sample_points(self, episode, time_indices, num_pts, t_offset: float = 0.0) -> Obs:
        """Randomly samples (x, y, t, s) points from the grid at specified time steps."""
        # Randomly pick (t, y, x) triplets
        t_idx = np.repeat(time_indices, num_pts)
        y_idx = np.random.randint(0, self.H, num_pts * len(time_indices))
        x_idx = np.random.randint(0, self.W, num_pts * len(time_indices))
        
        # Convert indices to real coordinates
        xs = torch.from_numpy(x_idx * self.res).float()
        ys = torch.from_numpy(y_idx * self.res).float()

        # Relative time: t_real - t_window_start
        ts = torch.from_numpy(t_idx * self.dt - t_offset).float()
        
        # Get smoke values
        vals = torch.from_numpy(episode[t_idx, y_idx, x_idx]).float()
        
        return Obs(xs=xs, ys=ys, ts=ts, values=vals)

class SequentialDataset(BaseSmokeDataset):
    """
    Dataset for sequential model-free models.
    Returns a sequence of Obs objects.
    """
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 25,
                 ctx_points_ratios: tuple[float, float] = (0.15, 0.25),
                 trg_points_ratio: float = 0.5,
                 train_split: float = 0.8,
                 mode: str = 'train',
                 max_episodes: Optional[int] = None):
        
        super().__init__(data_path, train_split, mode, max_episodes)
        self.ctx_min_pts = int(self.H * self.W * ctx_points_ratios[0])
        self.ctx_max_pts = int(self.H * self.W * ctx_points_ratios[1])
        self.trg_pts = int(self.H * self.W * trg_points_ratio)
        self.sequence_length = sequence_length
        assert self.sequence_length < self.n_steps, f"Sequence length {self.sequence_length} must be less than number of steps {self.n_steps}"
        assert trg_points_ratio > max(ctx_points_ratios), f"Target points ratio must be greater than max context points ratio"

    def __getitem__(self, idx: int):
        episode = self.smoke_data[idx]
        
        max_start = self.n_steps - self.sequence_length
        t_start_idx = np.random.randint(0, max_start)
        curr_seq_len = self.sequence_length
            
        frame_indices = np.arange(t_start_idx, t_start_idx + curr_seq_len)
        
        ctx_obs_seq = []
        trg_obs_seq = []
        for t_idx in frame_indices:
            num_ctx_pts = np.random.randint(self.ctx_min_pts, self.ctx_max_pts + 1) 
            ctx_obs = self._sample_points(episode, [t_idx], num_ctx_pts)
            trg_obs = self._sample_points(episode, [t_idx], self.trg_pts)

            # TODO: Fill here mask if collate is not used.
            
            ctx_obs_seq.append(ctx_obs)
            trg_obs_seq.append(trg_obs)
            
        return ctx_obs_seq, trg_obs_seq, t_start_idx

    def _sample_points(self, episode, time_indices, num_pts) -> Obs:
        """Randomly samples (x, y, t, s) points from the grid at specified time steps."""
        # Randomly pick (t, y, x) triplets
        t_idx = np.repeat(time_indices, num_pts)
        y_idx = np.random.randint(0, self.H, num_pts * len(time_indices))
        x_idx = np.random.randint(0, self.W, num_pts * len(time_indices))
        
        # Convert indices to real coordinates
        xs = torch.from_numpy(x_idx * self.res).float()
        ys = torch.from_numpy(y_idx * self.res).float()
        
        # Get smoke values
        vals = torch.from_numpy(episode[t_idx, y_idx, x_idx]).float()
        
        return Obs(xs=xs, ys=ys, values=vals)

def nonsequential_collate_fn(batch):
    ctx_list, trg_list, t0 = zip(*batch)
    B = len(batch)
    
    def pad_obs(obs_list):
        xs = [o.xs for o in obs_list]
        ys = [o.ys for o in obs_list]
        ts = [o.ts for o in obs_list]
        vs = [o.values for o in obs_list]
        
        xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
        ys_pad = pad_sequence(ys, batch_first=True, padding_value=0)
        ts_pad = pad_sequence(ts, batch_first=True, padding_value=0)
        vs_pad = pad_sequence(vs, batch_first=True, padding_value=0)
        
        # Mask: True where PADDING
        lengths = torch.tensor([len(x) for x in xs])
        S_max = max(lengths)

        mask_pad = torch.ones(len(lengths), S_max, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask_pad[i, length:] = False

        xs_pad = xs_pad.view(B, S_max, 1)
        ys_pad = ys_pad.view(B, S_max, 1)
        ts_pad = ts_pad.view(B, S_max, 1)
        vs_pad = vs_pad.view(B, S_max, 1)
        mask_pad = mask_pad.view(B, S_max, 1)

        return Obs(xs=xs_pad, ys=ys_pad, ts=ts_pad, values=vs_pad, mask=mask_pad)

    return pad_obs(ctx_list), pad_obs(trg_list), torch.tensor(t0)

def sequential_collate_fn(batch):
    ctx_seq_list, trg_seq_list, idx = zip(*batch)
    
    B = len(batch)           # Tamaño del batch
    T = len(ctx_seq_list[0])  # Pasos de tiempo

    def process_sequences(sequences):
        flat_obs = []
        for seq in sequences:
            for obs in seq:
                flat_obs.append(obs)
        
        xs = [o.xs.view(-1, 1) for o in flat_obs]
        ys = [o.ys.view(-1, 1) for o in flat_obs]
        vs = [o.values.view(-1, 1) for o in flat_obs]
        
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=0)
        vs_padded = pad_sequence(vs, batch_first=True, padding_value=0)
        
        lengths = torch.tensor([len(x) for x in xs])
        S_max = xs_padded.shape[1]
        mask_padded = torch.ones(len(lengths), S_max, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask_padded[i, length:] = False
        
        xs_final = xs_padded.view(B, T, S_max, 1)
        ys_final = ys_padded.view(B, T, S_max, 1)
        vs_final = vs_padded.view(B, T, S_max, 1)
        mask_final = mask_padded.view(B, T, S_max, 1)

        return Obs(xs=xs_final, ys=ys_final, values=vs_final, mask=mask_final)

    collated_ctx = process_sequences(ctx_seq_list)
    collated_trg = process_sequences(trg_seq_list)
    
    return collated_ctx, collated_trg, torch.tensor(idx)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    # Path to data (Adjust as needed)
    data_path = "/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/data/playback_data/global_source_400_100_2nd.npz"
    
    if not os.path.exists(data_path):
        print(f"Skipping test: Data not found at {data_path}")
    else:
        print("--- Testing NonSequentialDataset ---")
        dataset = NonSequentialDataset(data_path, sequence_length=25, max_episodes=5)
        loader = DataLoader(dataset, batch_size=4, collate_fn=nonsequential_collate_fn)
        
        ctx, trg, t0 = next(iter(loader))
        print(f"Batch t0: {t0}")
        print(f"Context xs shape: {ctx.xs.shape}, values shape: {ctx.values.shape}")
        print(f"Target xs shape: {trg.xs.shape}, values shape: {trg.values.shape}")
        
        # Visualization
        b_idx = 0
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        # Context
        # Flatten for scatter if necessary or index
        # NonSequential: (B, N)
        xs_c = ctx.xs[b_idx].numpy()
        ys_c = ctx.ys[b_idx].numpy()
        ts_c = ctx.ts[b_idx].numpy()
        vs_c = ctx.values[b_idx].numpy()
        mask_c = ctx.mask[b_idx].numpy()
        
        ts_c_unique = np.round(np.unique(ts_c), 2)
        ts_c_t0 = ts_c_unique[0]
        ts_c_tm1 = ts_c_unique[-1]
        mask_c_t0 = mask_c & (ts_c == ts_c_t0)
        mask_c_tm1 = mask_c & (ts_c == ts_c_tm1)
        
        sc1 = axs[0, 0].scatter(xs_c[mask_c_t0], ys_c[mask_c_t0], c=vs_c[mask_c_t0], cmap='viridis', s=5)
        axs[0, 0].set_title(f"Context t={ts_c_t0} (Valid: {mask_c_t0.sum()})")
        plt.colorbar(sc1, ax=axs[0, 0])
        
        sc2 = axs[0, 1].scatter(xs_c[mask_c_tm1], ys_c[mask_c_tm1], c=vs_c[mask_c_tm1], cmap='viridis', s=5)
        axs[0, 1].set_title(f"Context t={ts_c_tm1} (Valid: {mask_c_tm1.sum()})")
        plt.colorbar(sc2, ax=axs[0, 1])
        
        # Target
        xs_t = trg.xs[b_idx].numpy()
        ys_t = trg.ys[b_idx].numpy()
        vs_t = trg.values[b_idx].numpy()
        mask_t = trg.mask[b_idx].numpy()
        ts_t = trg.ts[b_idx].numpy()
        
        ts_t_unique = np.round(np.unique(ts_t), 2)
        ts_t_t0 = ts_t_unique[0]
        ts_t_tm1 = ts_t_unique[-1]
        mask_t_t0 = mask_t & (ts_t == ts_t_t0)
        mask_t_tm1 = mask_t & (ts_t == ts_t_tm1)
        
        sc2 = axs[1, 0].scatter(xs_t[mask_t_t0], ys_t[mask_t_t0], c=vs_t[mask_t_t0], cmap='viridis', s=5)
        axs[1, 0].set_title(f"Target t={ts_t_t0} (Valid: {mask_t_t0.sum()})")
        plt.colorbar(sc2, ax=axs[1, 0])
        
        sc3 = axs[1, 1].scatter(xs_t[mask_t_tm1], ys_t[mask_t_tm1], c=vs_t[mask_t_tm1], cmap='viridis', s=5)
        axs[1, 1].set_title(f"Target t={ts_t_tm1} (Valid: {mask_t_tm1.sum()})")
        plt.colorbar(sc3, ax=axs[1, 1])
        
        plt.suptitle(f"NonSequential Sample (t0={t0[b_idx]:.2f})")
        plt.tight_layout()
        plt.show()

        print("\n--- Testing SequentialDataset ---")
        seq_dataset = SequentialDataset(data_path, sequence_length=25, max_episodes=20)
        seq_loader = DataLoader(seq_dataset, batch_size=8, collate_fn=sequential_collate_fn, shuffle=True)
        
        ctx_seq, trg_seq, idx = next(iter(seq_loader))
        print(f"Sequence Length: {ctx_seq.xs.shape[1]}") # B, T, S
        print(f"Step 0 Context xs shape: {ctx_seq.xs.shape[2]}")
        
        # Visualize first step of sequence
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        b_idx = 0
        seq_idx_c0 = 0
        xs_c0 = ctx_seq.xs[b_idx, seq_idx_c0].numpy()
        ys_c0 = ctx_seq.ys[b_idx, seq_idx_c0].numpy()
        vs_c0 = ctx_seq.values[b_idx, seq_idx_c0].numpy()
        mask_c0 = ctx_seq.mask[b_idx, seq_idx_c0].numpy()
        
        sc1 = axs[0, 0].scatter(xs_c0[mask_c0], ys_c0[mask_c0], c=vs_c0[mask_c0], cmap='viridis', s=5)
        axs[0, 0].set_title("Seq Step 0 Context")
        plt.colorbar(sc1, ax=axs[0, 0])

        seq_idx_cm1 = -1
        xs_cm1 = ctx_seq.xs[b_idx, seq_idx_cm1].numpy()
        ys_cm1 = ctx_seq.ys[b_idx, seq_idx_cm1].numpy()
        vs_cm1 = ctx_seq.values[b_idx, seq_idx_cm1].numpy()
        mask_cm1 = ctx_seq.mask[b_idx, seq_idx_cm1].numpy()
        
        sc1 = axs[0, 1].scatter(xs_cm1[mask_cm1], ys_cm1[mask_cm1], c=vs_cm1[mask_cm1], cmap='viridis', s=5)
        axs[0, 1].set_title("Seq Step -1 Context")
        plt.colorbar(sc1, ax=axs[0, 1])
        
        xs_t0 = trg_seq.xs[b_idx, seq_idx_c0].numpy()
        ys_t0 = trg_seq.ys[b_idx, seq_idx_c0].numpy()
        vs_t0 = trg_seq.values[b_idx, seq_idx_c0].numpy()
        mask_t0 = trg_seq.mask[b_idx, seq_idx_c0].numpy()
        
        sc2 = axs[1, 0].scatter(xs_t0[mask_t0], ys_t0[mask_t0], c=vs_t0[mask_t0], cmap='viridis', s=5)
        axs[1, 0].set_title("Seq Step 0 Target")
        plt.colorbar(sc2, ax=axs[1, 0])

        seq_idx_tm1 = -1
        xs_tm1 = trg_seq.xs[b_idx, seq_idx_tm1].numpy()
        ys_tm1 = trg_seq.ys[b_idx, seq_idx_tm1].numpy()
        vs_tm1 = trg_seq.values[b_idx, seq_idx_tm1].numpy()
        mask_tm1 = trg_seq.mask[b_idx, seq_idx_tm1].numpy()
        
        sc2 = axs[1, 1].scatter(xs_tm1[mask_tm1], ys_tm1[mask_tm1], c=vs_tm1[mask_tm1], cmap='viridis', s=5)
        axs[1, 1].set_title("Seq Step -1 Target")
        plt.colorbar(sc2, ax=axs[1, 1])
        
        plt.suptitle("Sequential Sample Step 0")
        plt.tight_layout()
        plt.show()
