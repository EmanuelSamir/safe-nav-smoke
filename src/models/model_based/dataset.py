import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import os

from src.models.model_based.utils import ObsPINN

class GlobalSmokeDataset(Dataset):
    """
    Dataset for global smoke forecasting using CNP-PINN.
    Loads data from a 4D tensor (Episodes, Time, H, W).
    """
    def __init__(self, 
                 data_path: str, 
                 context_frames: int = 10, 
                 target_frames: int = 15,
                 info_ratio_per_frame: float = 0.2,
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
        self.context_frames = context_frames
        self.target_frames = target_frames
        self.info_ratio_per_frame = info_ratio_per_frame
        self.num_pts_per_frame = int(self.H * self.W * self.info_ratio_per_frame)
        
        # Train/Val split
        split_idx = int(self.n_episodes * train_split)
        if mode == 'train':
            self.smoke_data = self.smoke_data[:split_idx]
            self.n_episodes = split_idx
        else:
            self.smoke_data = self.smoke_data[split_idx:]
            self.n_episodes = self.n_episodes - split_idx
            
        if max_samples is not None:
             self.n_episodes = min(self.n_episodes, max_samples)
             self.smoke_data = self.smoke_data[:self.n_episodes]

    def __len__(self):
        # We can sample many windows per episode
        # For simplicity, 1 sequence per episode per epoch, but we can randomize start
        return self.n_episodes

    def __getitem__(self, idx: int) -> Tuple[ObsPINN, ObsPINN, ObsPINN, torch.Tensor, int]:
        episode = self.smoke_data[idx]
        
        # 0. Inflow map is the first frame (T=0)
        inflow_map = torch.from_numpy(episode[0]).float() # (H, W)
        
        # 1. Select a random time window
        max_start = self.n_steps - (self.context_frames + self.target_frames)
        t_start_idx = np.random.randint(0, max_start)
        t_offset = t_start_idx * self.dt # Comienzo de la ventana
        
        # Indices for context and target
        ctx_indices = np.arange(t_start_idx, t_start_idx + self.context_frames)
        trg_indices = np.arange(t_start_idx + self.context_frames, t_start_idx + self.context_frames + self.target_frames)
        
        # 2. Sample points using Relative Time (t - t0)
        ctx_obs = self._sample_points(episode, ctx_indices, self.num_pts_per_frame, t_offset=t_offset)
        trg_obs = self._sample_points(episode, trg_indices, self.num_pts_per_frame, t_offset=t_offset)
        
        # 4. Total points (Context + Target) for Posterior
        total_xs = torch.cat([ctx_obs.xs, trg_obs.xs])
        total_ys = torch.cat([ctx_obs.ys, trg_obs.ys])
        total_ts = torch.cat([ctx_obs.ts, trg_obs.ts])
        total_vals = torch.cat([ctx_obs.values, trg_obs.values])
        total_obs = ObsPINN(xs=total_xs, ys=total_ys, ts=total_ts, values=total_vals)
        
        return ctx_obs, trg_obs, total_obs, inflow_map, idx, t_offset

    def _sample_points(self, episode, time_indices, num_pts, t_offset: float = 0.0) -> ObsPINN:
        """Randomly samples (x, y, t, s) points from the grid at specified time steps."""
        # Randomly pick (t, y, x) triplets

        t_idx = np.repeat(time_indices, num_pts)
        y_idx = np.random.randint(0, self.H, num_pts * len(time_indices))
        x_idx = np.random.randint(0, self.W, num_pts * len(time_indices))
        
        # Convert indices to real coordinates
        xs = torch.from_numpy(x_idx * self.res).float()
        ys = torch.from_numpy(y_idx * self.res).float()
        # TIEMPO RELATIVO: t_real - t_window_start
        ts = torch.from_numpy(t_idx * self.dt - t_offset).float()
        
        # Get smoke values
        vals = torch.from_numpy(episode[t_idx, y_idx, x_idx]).float()
        
        return ObsPINN(xs=xs, ys=ys, ts=ts, values=vals)

def pinn_collate_fn(batch):
    """
    Simplest collate for ObsPINN. 
    """
    ctx_list, trg_list, total_list, inflow_list, ep_idx_list, t0_list = zip(*batch)
    
    def stack_obs(obs_list):
        return ObsPINN(
            xs=torch.stack([o.xs for o in obs_list]),
            ys=torch.stack([o.ys for o in obs_list]),
            ts=torch.stack([o.ts for o in obs_list]),
            values=torch.stack([o.values for o in obs_list])
        )
    
    return stack_obs(ctx_list), stack_obs(trg_list), stack_obs(total_list), torch.stack(inflow_list), torch.tensor(ep_idx_list), torch.tensor(t0_list)
