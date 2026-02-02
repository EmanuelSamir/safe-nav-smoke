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

from src.models.model_based.utils import ObsPINN
from src.models.shared.observations import Obs

class BaseSmokeDataset(Dataset):
    """Base class for handling the smoke_data.npz loading and splitting."""
    def __init__(self, 
                 data_path: str,
                 train_split: float = 0.8,
                 mode: str = 'train',
                 max_samples: Optional[int] = None):
        
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
            
        if max_samples is not None:
             self.n_episodes = min(self.n_episodes, max_samples)
             self.smoke_data = self.smoke_data[:self.n_episodes]

    def __len__(self):
        return self.n_episodes

class GlobalSmokeDataset(BaseSmokeDataset):
    """
    Dataset for global smoke forecasting using CNP-PINN.
    Returns (Context, Target) sets of (x,y,t,s) points.
    """
    def __init__(self, 
                 data_path: str, 
                 context_frames: int = 10, 
                 target_frames: int = 15,
                 min_points_ratio: float = 0.05,
                 max_points_ratio: float = 0.25,
                 train_split: float = 0.8,
                 mode: str = 'train',
                 max_samples: Optional[int] = None):
        
        super().__init__(data_path, train_split, mode, max_samples)
        
        self.context_frames = context_frames
        self.target_frames = target_frames
        
        self.min_pts = int(self.H * self.W * min_points_ratio)
        self.max_pts = int(self.H * self.W * max_points_ratio)

    def __getitem__(self, idx: int) -> Tuple[ObsPINN, ObsPINN, ObsPINN, torch.Tensor, int, float]:
        episode = self.smoke_data[idx]
        
        inflow_map = torch.from_numpy(episode[0]).float()
        
        # 1. Time Window
        max_start = self.n_steps - (self.context_frames + self.target_frames)
        t_start_idx = np.random.randint(0, max_start)
        t_offset = t_start_idx * self.dt
        
        ctx_indices = np.arange(t_start_idx, t_start_idx + self.context_frames)
        trg_indices = np.arange(t_start_idx + self.context_frames, t_start_idx + self.context_frames + self.target_frames)
        
        # 2. Variable Points (per frame)
        num_pts = np.random.randint(self.min_pts, self.max_pts + 1)
        
        ctx_obs = self._sample_points(episode, ctx_indices, num_pts, t_offset=t_offset)
        trg_obs = self._sample_points(episode, trg_indices, num_pts, t_offset=t_offset)
        
        total_xs = torch.cat([ctx_obs.xs, trg_obs.xs])
        total_ys = torch.cat([ctx_obs.ys, trg_obs.ys])
        total_ts = torch.cat([ctx_obs.ts, trg_obs.ts])
        total_vals = torch.cat([ctx_obs.values, trg_obs.values])
        total_obs = ObsPINN(xs=total_xs, ys=total_ys, ts=total_ts, values=total_vals)
        
        return ctx_obs, trg_obs, total_obs, inflow_map, idx, t_offset

    def _sample_points(self, episode, time_indices, num_pts_per_frame, t_offset: float = 0.0) -> ObsPINN:
        t_idx = np.repeat(time_indices, num_pts_per_frame)
        total_pts = len(t_idx)
        
        y_idx = np.random.randint(0, self.H, total_pts)
        x_idx = np.random.randint(0, self.W, total_pts)
        
        xs = torch.from_numpy(x_idx * self.res).float()
        ys = torch.from_numpy(y_idx * self.res).float()
        ts = torch.from_numpy(t_idx * self.dt - t_offset).float()
        vals = torch.from_numpy(episode[t_idx, y_idx, x_idx]).float()
        
        return ObsPINN(xs=xs, ys=ys, ts=ts, values=vals)


class SequentialDataset(BaseSmokeDataset):
    """
    Dataset for sequential model-free models.
    Returns a sequence of Obs objects.
    """
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 25,
                 min_points_ratio: float = 0.05,
                 max_points_ratio: float = 0.25,
                 train_split: float = 0.8,
                 mode: str = 'train',
                 max_samples: Optional[int] = None):
        
        super().__init__(data_path, train_split, mode, max_samples)
        self.sequence_length = sequence_length
        self.min_pts = int(self.H * self.W * min_points_ratio)
        self.max_pts = int(self.H * self.W * max_points_ratio)

    def __getitem__(self, idx: int):
        episode = self.smoke_data[idx]
        
        max_start = self.n_steps - self.sequence_length
        if max_start <= 0:
            t_start_idx = 0
            curr_seq_len = self.n_steps
        else:
            t_start_idx = np.random.randint(0, max_start)
            curr_seq_len = self.sequence_length
            
        frame_indices = np.arange(t_start_idx, t_start_idx + curr_seq_len)
        
        num_pts = np.random.randint(self.min_pts, self.max_pts + 1)
        
        obs_seq = []
        for t_idx in frame_indices:
            y_idx = np.random.randint(0, self.H, num_pts)
            x_idx = np.random.randint(0, self.W, num_pts)
            
            xs = torch.from_numpy(x_idx * self.res).float()
            ys = torch.from_numpy(y_idx * self.res).float()
            vals = torch.from_numpy(episode[t_idx, y_idx, x_idx]).float()
            
            # Mask is False (zeros) for valid data
            mask = torch.zeros_like(vals, dtype=torch.bool)
            
            step_obs = Obs(xs=xs, ys=ys, values=vals, mask=mask)
            obs_seq.append(step_obs)
            
        return obs_seq, idx

def pinn_collate_fn(batch):
    ctx_list, trg_list, total_list, inflow_map, ep_idx, t0 = zip(*batch)
    
    def pad_obs_pinn(obs_list):
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
        max_len = xs_pad.shape[1]
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        
        return ObsPINN(xs=xs_pad, ys=ys_pad, ts=ts_pad, values=vs_pad, mask=mask)

    return pad_obs_pinn(ctx_list), pad_obs_pinn(trg_list), pad_obs_pinn(total_list), torch.stack(inflow_map), torch.tensor(ep_idx), torch.tensor(t0)

def sequential_collate_fn(batch):
    # batch is list of (obs_seq, idx)
    batch_size = len(batch)
    seq_len = len(batch[0][0])
    
    collated_seq = []
    
    for t in range(seq_len):
        obs_t = [b[0][t] for b in batch]
        
        xs = [o.xs for o in obs_t]
        ys = [o.ys for o in obs_t]
        vs = [o.values for o in obs_t]
        
        xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
        ys_pad = pad_sequence(ys, batch_first=True, padding_value=0)
        vs_pad = pad_sequence(vs, batch_first=True, padding_value=0)
        
        lengths = torch.tensor([len(x) for x in xs])
        max_len = xs_pad.shape[1]
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        
        batched_obs = Obs(xs=xs_pad, ys=ys_pad, values=vs_pad, mask=mask)
        
        collated_seq.append(batched_obs)
        
    return collated_seq, torch.tensor([b[1] for b in batch])
