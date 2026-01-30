import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ObsSNP:
    """
    Standard observation bundle for SNP/Model-Free models.
    Contains spatial coordinates (xs, ys), measured values (values), and a mask.
    """
    xs: torch.Tensor
    ys: torch.Tensor
    values: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        return ObsSNP(
            xs=self.xs.to(device),
            ys=self.ys.to(device),
            values=self.values.to(device) if self.values is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None
        )

# Alias for compatibility if needed or transition
ObsVAESSM = ObsSNP

def collate_variable_batch(batch_list: List[ObsSNP]) -> ObsSNP:
    """
    Takes a list of observations (obs) and pads them into a single batched observation.
    """
    obs_xys = [torch.stack([b.xs, b.ys], dim=-1) for b in batch_list]
    obs_vals = [b.values for b in batch_list if b.values is not None]
    
    # Pad Sequence (obs)
    obs_xy_pad = pad_sequence(obs_xys, batch_first=True, padding_value=0)

    obs_lens = torch.tensor([len(t) for t in obs_xys])
    obs_max_len = obs_xy_pad.shape[1]
    # mask: True where PADDING is
    obs_mask = (torch.arange(obs_max_len)[None, :] >= obs_lens[:, None]).bool()

    if obs_vals:
        obs_val_pad = pad_sequence(obs_vals, batch_first=True, padding_value=0)
    else:
        obs_val_pad = None

    batch_obs = ObsSNP(
        xs=obs_xy_pad[:, :, 0],
        ys=obs_xy_pad[:, :, 1],
        values=obs_val_pad,
        mask=obs_mask
    )
    return batch_obs

class SequentialDataset(Dataset):
    """
    Dataset for training sequential models (SNP, RNP) from a generic replay buffer.
    Handles coordinate normalization and sequence windowing.
    """
    def __init__(self, replay_buffer, sequence_length: int, valid_indices: List[int]):
        self.buffer = replay_buffer
        self.seq_len = sequence_length + 1  # +1 to include the next step for target
        self.indices = valid_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        start_idx = self.indices[idx]
        
        sequence_data = []
        for t in range(self.seq_len):
            data = self.buffer.get_from_index(start_idx + t)
            step_data = self._process_step(data)
            sequence_data.append(step_data)
            
        return sequence_data

    def _process_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms raw buffer data into normalized tensors."""
        # 1. Spatial coordinates and values
        raw_pos = np.array(data["smoke_value_positions"])
        raw_vals = np.array(data["smoke_values"])
        
        # 2. Agent state for normalization (relative positions)
        r_x, r_y, r_theta = data["state"][0], data["state"][1], data["state"][2]
        
        # Normalize positions relative to agent
        if len(raw_pos) > 0:
            norm_xs = torch.from_numpy(raw_pos[:, 0] - r_x).float()
            norm_ys = torch.from_numpy(raw_pos[:, 1] - r_y).float()
            vals = torch.from_numpy(raw_vals).float().squeeze()
        else:
             norm_xs = torch.tensor([])
             norm_ys = torch.tensor([])
             vals = torch.tensor([])

        # 3. Actions (Project velocity to global frame)
        v_local, _ = data["actions"] # w_local ignored for simple [vx, vy] action vec
        global_vx = v_local * np.cos(r_theta)
        global_vy = v_local * np.sin(r_theta)
        action_vec = torch.tensor([global_vx, global_vy], dtype=torch.float32)
        
        done = torch.tensor([float(data["done"])], dtype=torch.float32)
        
        return {
            "obs": ObsSNP(xs=norm_xs, ys=norm_ys, values=vals),
            "action": action_vec,
            "done": done
        }

def sequential_collate_fn(batch: List[List[Dict[str, Any]]]):
    """
    Collates a batch of sequences into (Time, Batch, ...) format.
    Uses padding for variable numbers of observation points.
    """
    batch_size = len(batch)
    seq_len = len(batch[0])
    
    collated_seq = []
    
    for t in range(seq_len):
        # Extract all steps at time t across the batch
        steps_t = [batch[b][t] for b in range(batch_size)]
        
        # BATCH OBSERVATIONS (Variable number of points)
        obs_list = [s["obs"] for s in steps_t]
        
        batched_obs = collate_variable_batch(obs_list)
        
        # BATCH ACTIONS AND DONES
        batched_actions = torch.stack([s["action"] for s in steps_t])
        batched_dones = torch.stack([s["done"] for s in steps_t])
        
        collated_seq.append({
            "obs": batched_obs,
            "action": batched_actions,
            "done": batched_dones
        })
        
    return collated_seq

