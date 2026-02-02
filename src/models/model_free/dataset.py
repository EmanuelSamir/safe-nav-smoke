"""
Unified dataset classes for all models.
Clean, standardized structure.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional
from src.models.shared.observations import Obs

class SequentialDataset(Dataset):
    """
    Dataset for sequential model-free models (RNP, SNP).
    Returns sequences of (observation, action, done) tuples.
    Supports global smoke positions and configurable inputs (action, robot_state).
    """
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 25,
                 context_spatial_ratio: float = 0.2,
                 target_spatial_ratio: float = 0.2,
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
        self.context_spatial_ratio = context_spatial_ratio
        self.target_spatial_ratio = target_spatial_ratio
        self.num_ctx_pts_per_frame = int(self.H * self.W * self.context_spatial_ratio)
        self.num_trg_pts_per_frame = int(self.H * self.W * self.target_spatial_ratio)
        self.sequence_length = np.min([sequence_length, self.n_steps])

        split_idx = int(train_split * self.n_episodes)
        if mode == 'train':
            self.smoke_data = self.smoke_data[:split_idx]
            self.n_episodes = split_idx
        elif mode == 'val':
            self.smoke_data = self.smoke_data[split_idx:]
            self.n_episodes = self.n_episodes - split_idx
        
        # self.buffer = replay_buffer
        # self.seq_len = sequence_length
        # self.indices = indices
        
        # self.use_actions = use_actions
        # self.action_dim = action_dim
        # self.use_robot_state = use_robot_state
        # self.robot_state_dim = robot_state_dim
        # self.info_ratio = info_ratio_per_frame
    
    def __len__(self):
        return self.n_episodes*self.n_steps
    
    def __getitem__(self, idx):
        episode = self.smoke_data[idx]
        
        # 1. Select a random time window
        max_start = self.n_steps - self.sequence_length
        t_start_idx = np.random.randint(0, max_start)
        
        # Indices for context and target
        frame_indices = np.arange(t_start_idx, t_start_idx + self.sequence_length)
        
        # 2. Sample points using Relative Time (t - t0)
        ctx_obs_seq = []
        trg_obs_seq = []
        total_obs_seq = []
        for t_idx in frame_indices:
            t_ctx = np.repeat(t_idx, self.num_ctx_pts_per_frame)
            ctx_obs = self._sample_points(episode, t_ctx, self.num_ctx_pts_per_frame)
            t_trg = np.repeat(t_idx, self.num_trg_pts_per_frame)
            trg_obs = self._sample_points(episode, t_trg, self.num_trg_pts_per_frame)

            ctx_obs_seq.append(ctx_obs)
            trg_obs_seq.append(trg_obs)

            total_xs = torch.cat([ctx_obs.xs, trg_obs.xs])
            total_ys = torch.cat([ctx_obs.ys, trg_obs.ys])
            total_vals = torch.cat([ctx_obs.values, trg_obs.values])
            total_obs = Obs(xs=total_xs, ys=total_ys, values=total_vals, mask=torch.ones_like(total_vals, dtype=torch.bool))
            total_obs_seq.append(total_obs)
        
        return ctx_obs_seq, trg_obs_seq, total_obs_seq, idx, t_start_idx
    
    def _sample_points(self, episode, time_indices, num_pts) -> Obs:
        """Randomly samples (x, y, t, s) points from the grid at specified time steps."""
        # Randomly pick (t, y, x) triplets

        y_idx = np.random.randint(0, self.H, num_pts)
        x_idx = np.random.randint(0, self.W, num_pts)
        
        # Convert indices to real coordinates
        xs = torch.from_numpy(x_idx * self.res).float()
        ys = torch.from_numpy(y_idx * self.res).float()
        
        # Get smoke values
        vals = torch.from_numpy(episode[time_indices, y_idx, x_idx]).float()
        
        return Obs(xs=xs, ys=ys, values=vals, mask=torch.ones_like(vals, dtype=torch.bool))


def collate_sequences(batch: List[List[Dict]]) -> List[Dict]:
    """
    Collate function for SequentialDataset.
    """
    seq_len = len(batch[0])
    batch_size = len(batch)
    
    collated = []
    for t in range(seq_len):
        # Gather all data for timestep t
        obs_list = [batch[b][t][0] for b in range(batch_size)] # ctx_obs_seq
        actions = torch.stack([batch[b][t][1] for b in range(batch_size)]) # trg_obs_seq
        robot_states = torch.stack([batch[b][t][2] for b in range(batch_size)]) # total_obs_seq
        dones = torch.stack([batch[b][t][3] for b in range(batch_size)]) # idx
        
        # Collate observations (handle variable length)
        obs_batch = collate_observations(obs_list)
        
        collated.append({
            'obs': obs_batch,
            'action': actions,
            'robot_state': robot_states,
            'done': dones
        })
    
    return collated


def collate_observations(obs_list: List[Obs]) -> Obs:
    """
    Collate a list of Obs into a single batched Obs with padding.
    """
    # Stack coordinates
    coords = [torch.stack([o.xs, o.ys], dim=-1) for o in obs_list]
    values = [o.values for o in obs_list if o.values is not None]
    
    # Pad sequences
    coords_pad = pad_sequence(coords, batch_first=True, padding_value=0.0)
    
    # Create mask (True where VALID, False where PADDING)
    # Note: Previous implementation might have inverted mask meaning? 
    # Usually in PyTorch padding mask: True = Padding (Ignore).
    # But here 'mask' logic in training scripts often implies 'valid' or uses explicit boolean.
    # The previous code: mask = (torch.arange(max_len)[None, :] < lengths[:, None])
    # This creates True for VALID elements, False for Padding.
    
    lengths = torch.tensor([len(c) for c in coords])
    max_len = coords_pad.shape[1]
    mask = (torch.arange(max_len)[None, :] < lengths[:, None])
    
    # Pad values if present
    if values:
        values_pad = pad_sequence(values, batch_first=True, padding_value=0.0)
    else:
        values_pad = None
    
    return Obs(
        xs=coords_pad[..., 0],
        ys=coords_pad[..., 1],
        values=values_pad,
        mask=mask
    )






# TODO: Bring back this approach when actions can be considered as part of the state
# class SequentialDataset(Dataset):
#     """
#     Dataset for sequential model-free models (RNP, SNP).
#     Returns sequences of (observation, action, done) tuples.
#     Supports global smoke positions and configurable inputs (action, robot_state).
#     """
#     def __init__(self, 
#                  replay_buffer, 
#                  sequence_length: int, 
#                  indices: np.ndarray,
#                  use_actions: bool = True,
#                  action_dim: int = 2,
#                  use_robot_state: bool = False,
#                  robot_state_dim: int = 0,
#                  info_ratio_per_frame: float = 1.0):
        
#         self.buffer = replay_buffer
#         self.seq_len = sequence_length
#         self.indices = indices
        
#         self.use_actions = use_actions
#         self.action_dim = action_dim
#         self.use_robot_state = use_robot_state
#         self.robot_state_dim = robot_state_dim
#         self.info_ratio = info_ratio_per_frame
        
#         # Check for global smoke positions
#         self.global_smoke_pos = None
#         if hasattr(self.buffer, 'global_data') and 'smoke_value_positions' in self.buffer.global_data:
#             self.global_smoke_pos = torch.from_numpy(self.buffer.global_data['smoke_value_positions']).float()
#             # Ensure it's 2D (N, 2)
#             if self.global_smoke_pos.dim() == 3 and self.global_smoke_pos.shape[0] == 1:
#                  self.global_smoke_pos = self.global_smoke_pos.squeeze(0)
    
#     def __len__(self):
#         return len(self.indices)
    
#     def __getitem__(self, idx):
#         start_idx = self.indices[idx]
#         sequence = []
        
#         for t in range(self.seq_len + 1):
#             step_idx = start_idx + t
#             data = self.buffer.get_from_index(step_idx)
            
#             # --- Smoke Observations ---
#             # Priority: 1. Step data, 2. Global data
#             if 'smoke_value_positions' in data:
#                 smoke_pos = torch.from_numpy(data['smoke_value_positions']).float()
#             elif self.global_smoke_pos is not None:
#                 smoke_pos = self.global_smoke_pos
#             else:
#                  # Fallback/Error? Assuming one of them exists or empty.
#                  # If empty, maybe create empty tensor?
#                  smoke_pos = torch.zeros((0, 2))
            
#             if 'smoke_values' in data:
#                 smoke_val = torch.from_numpy(data['smoke_values']).float()
#             else:
#                 smoke_val = torch.zeros((smoke_pos.shape[0]))

#             # Sub-sample if requested
#             if self.info_ratio < 1.0 and len(smoke_pos) > 0:
#                 num_points = int(len(smoke_pos) * self.info_ratio)
#                 if num_points > 0:
#                     perm = torch.randperm(len(smoke_pos))[:num_points]
#                     smoke_pos = smoke_pos[perm]
#                     smoke_val = smoke_val[perm]

#             obs = Obs(
#                 xs=smoke_pos[:, 0],
#                 ys=smoke_pos[:, 1],
#                 values=smoke_val,
#                 mask=torch.ones(len(smoke_val), dtype=torch.bool)  # All valid
#             )
            
#             # --- Actions ---
#             if self.use_actions and 'actions' in data:
#                 action = data['actions']
#                 if isinstance(action, list):
#                     action = np.array(action)
#                 action = torch.from_numpy(action).float()
#             else:
#                 action = torch.zeros(self.action_dim, dtype=torch.float32)
            
#             # --- Robot State ---
#             if self.use_robot_state and 'robot_state' in data:
#                  r_state = data['robot_state']
#                  if isinstance(r_state, list):
#                      r_state = np.array(r_state)
#                  r_state = torch.from_numpy(r_state).float()
#             else:
#                  r_state = torch.zeros(self.robot_state_dim, dtype=torch.float32)

#             done = torch.tensor([data['done']], dtype=torch.float32)
            
#             sequence.append({
#                 'obs': obs, 
#                 'action': action, 
#                 'robot_state': r_state,
#                 'done': done
#             })
        
#         return sequence


# def collate_sequences(batch: List[List[Dict]]) -> List[Dict]:
#     """
#     Collate function for SequentialDataset.
#     """
#     seq_len = len(batch[0])
#     batch_size = len(batch)
    
#     collated = []
#     for t in range(seq_len):
#         # Gather all data for timestep t
#         obs_list = [batch[b][t]['obs'] for b in range(batch_size)]
#         actions = torch.stack([batch[b][t]['action'] for b in range(batch_size)])
#         robot_states = torch.stack([batch[b][t]['robot_state'] for b in range(batch_size)])
#         dones = torch.stack([batch[b][t]['done'] for b in range(batch_size)])
        
#         # Collate observations (handle variable length)
#         obs_batch = collate_observations(obs_list)
        
#         collated.append({
#             'obs': obs_batch,
#             'action': actions,
#             'robot_state': robot_states,
#             'done': dones
#         })
    
#     return collated


# def collate_observations(obs_list: List[Obs]) -> Obs:
#     """
#     Collate a list of Obs into a single batched Obs with padding.
#     """
#     # Stack coordinates
#     coords = [torch.stack([o.xs, o.ys], dim=-1) for o in obs_list]
#     values = [o.values for o in obs_list if o.values is not None]
    
#     # Pad sequences
#     coords_pad = pad_sequence(coords, batch_first=True, padding_value=0.0)
    
#     # Create mask (True where VALID, False where PADDING)
#     # Note: Previous implementation might have inverted mask meaning? 
#     # Usually in PyTorch padding mask: True = Padding (Ignore).
#     # But here 'mask' logic in training scripts often implies 'valid' or uses explicit boolean.
#     # The previous code: mask = (torch.arange(max_len)[None, :] < lengths[:, None])
#     # This creates True for VALID elements, False for Padding.
    
#     lengths = torch.tensor([len(c) for c in coords])
#     max_len = coords_pad.shape[1]
#     mask = (torch.arange(max_len)[None, :] < lengths[:, None])
    
#     # Pad values if present
#     if values:
#         values_pad = pad_sequence(values, batch_first=True, padding_value=0.0)
#     else:
#         values_pad = None
    
#     return Obs(
#         xs=coords_pad[..., 0],
#         ys=coords_pad[..., 1],
#         values=values_pad,
#         mask=mask
#     )
