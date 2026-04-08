import os

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset


class SmokeWorldDataset(Dataset):
    """Autoregressive Fluid World Model Dataset.
    Yields sequences of size (L, 7, 80, 80) for input and (L, 1, 80, 80) for target.
    Input Channels (7):
    - 0: Observation Splatting (O_t)
    - 1-4: Temporal Memory (mu_{t-1...t-4} warped to current frame)
    - 5: Visibility Mask (V_t)
    - 6: Relative Action (u_t - linear velocity)
    """

    def __init__(
        self,
        data_path: str,
        res: float = 0.2,
        world_size: float = 30.0,
        max_range: float = 8.0,
        seq_len: int = 8,  # Length of the sequence L
        past_frames: int = 4, # Number of past context frames
        mode: str = "train",
        sample_ratio: float = 1.0,
    ):
        self.dataset = datasets.load_from_disk(data_path)
        self.res = res
        self.world_size = world_size
        self.max_range = max_range
        self.seq_len = seq_len
        self.past_frames = past_frames

        # Patch size: 8m range -> 16m diameter -> 80 pixels
        self.crop_size = int((max_range * 2) / res)

        self.fov_deg = 90.0
        self.fov_rad = np.deg2rad(self.fov_deg)
        self.num_rays = 64  # Updated

        # Precompute geometry for _project_scanline to avoid CPU bottleneck
        center = self.crop_size // 2
        y, x = np.ogrid[: self.crop_size, : self.crop_size]
        dx = x - center
        dy = y - center
        dist_sq = dx**2 + dy**2
        max_range_px = self.max_range / self.res
        self.dist_mask = (dist_sq <= max_range_px**2) & (dist_sq > 0)
        self.pix_angle = np.arctan2(dy, dx)

        # Create valid sequence starts
        terminated = self.dataset["terminated"]
        truncated = self.dataset["truncated"]

        self.valid_starts = []
        ep_start = 0
        for i in range(len(terminated)):
            if terminated[i] or truncated[i]:
                needed = seq_len
                if (i - ep_start + 1) >= needed:
                    self.valid_starts.extend(range(ep_start, i - seq_len + 2))
                ep_start = i + 1

        # (Subsampling removed from Dataset init so PyTorch Lightning can handle random subsets per epoch)

        # Split 80/20 train/val
        split = int(0.8 * len(self.valid_starts))
        if mode == "train":
            self.valid_starts = self.valid_starts[:split]
        else:
            self.valid_starts = self.valid_starts[split:]

        print(
            f"Dataset mode {mode}: {len(self.valid_starts)} sequences. Sequence Length: {seq_len}"
        )

    def __len__(self):
        return len(self.valid_starts)

    def _get_crop(self, full_map: np.ndarray, loc: np.ndarray) -> np.ndarray:
        """Crops a patch centered at world loc. Zero-pads boundaries without slow np.pad."""
        gx = int(loc[0] / self.res)
        gy = int(loc[1] / self.res)

        half = self.crop_size // 2
        mh, mw = full_map.shape
        
        x1, x2 = gx - half, gx + half
        y1, y2 = gy - half, gy + half
        
        vx1, vx2 = max(0, x1), min(mw, x2)
        vy1, vy2 = max(0, y1), min(mh, y2)
        
        crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
        if vx1 < vx2 and vy1 < vy2:
            tx1, tx2 = vx1 - x1, vx2 - x1
            ty1, ty2 = vy1 - y1, vy2 - y1
            crop[ty1:ty2, tx1:tx2] = full_map[vy1:vy2, vx1:vx2]
            
        return crop

    def _project_scanline(self, readings: np.ndarray, angle_rad: float) -> np.ndarray:
        """Projects 1D scanline into 2D wedge using precomputed fast geometry."""
        rel_angle = (self.pix_angle - angle_rad + np.pi) % (2 * np.pi) - np.pi
        in_fov = np.abs(rel_angle) <= self.fov_rad / 2
        
        valid_mask = self.dist_mask & in_fov
        
        ray_idx = ((rel_angle[valid_mask] + self.fov_rad / 2) / self.fov_rad * (self.num_rays - 1)).astype(int)
        ray_idx = np.clip(ray_idx, 0, self.num_rays - 1)
        
        avg_density = readings[ray_idx] / self.max_range
        
        proj = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        proj[valid_mask] = avg_density
        
        return proj

    def __getitem__(self, idx: int):
        start_idx = self.valid_starts[idx]
        
        # Batch fetch all required frames in one PyArrow slice!
        min_idx = start_idx
        max_idx = start_idx + self.seq_len - 1
        block = self.dataset[min_idx : max_idx + 1]

        seq_inputs = []
        seq_targets = []

        for t_off in range(self.seq_len):
            curr_rel = t_off

            # Position at t
            loc_t = np.array(block["obs_location"][curr_rel])
            angle_t = float(block["obs_angle"][curr_rel][0])
            readings_t = np.array(block["obs_readings"][curr_rel])
            full_map_t = np.array(block["obs_full_map"][curr_rel])
            action_t = np.array(block["action"][curr_rel])

            # --- Input Channels ---
            # 1. Observation Splatting (O_t)
            o_t = self._project_scanline(readings_t, angle_t)

            # 2. Visibility Mask (V_t)
            v_t = (
                self._project_scanline(np.ones_like(readings_t) * self.max_range, angle_t) > 0
            ).astype(float)

            # 3. Action (u_t)
            u_t = np.full((self.crop_size, self.crop_size), action_t[0])

            # Stack: [o_t, v_t, u_t] -> 3 channels (Memory handled by training loop)
            input_stack = np.stack([o_t, v_t, u_t], axis=0)
            seq_inputs.append(input_stack)

            # --- Target ---
            y_t = self._get_crop(full_map_t, loc_t)
            seq_targets.append(y_t[None])

        return torch.from_numpy(np.array(seq_inputs)).float(), torch.from_numpy(
            np.array(seq_targets)
        ).float()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_path = "data/sensor_dataset_hf"

    if os.path.exists(data_path):
        ds = SmokeWorldDataset(data_path, mode="train", sample_ratio=1.0, seq_len=4, past_frames=4)

        # Visualize one sequence
        idx = 200  # Random sequence index
        x_seq, y_seq = ds[idx]  # (L, 7, 80, 80), (L, 1, 80, 80)

        L = x_seq.shape[0]
        fig, axes = plt.subplots(L, 8, figsize=(20, 3 * L))
        names = ["O_t", "Mem t-1", "Mem t-2", "Mem t-3", "Mem t-4", "V_t", "u_t", "Target GT"]

        for t in range(L):
            for i in range(7):
                im = axes[t, i].imshow(x_seq[t, i].numpy(), cmap="rainbow" if i < 5 else "magma")
                if t == 0:
                    axes[t, i].set_title(names[i])
            im = axes[t, 7].imshow(y_seq[t, 0].numpy(), cmap="rainbow")
            if t == 0:
                axes[t, 7].set_title(names[7])
            axes[t, 0].set_ylabel(f"t={t}")

        plt.suptitle(f"Autoregressive Sequence Sample - {L} Steps", fontsize=16)
        plt.tight_layout()
        plt.savefig("dataset_inspection_sequential.png")
        print("Sequential inspection saved to: dataset_inspection_sequential.png")
    else:
        print(f"Data not found at {data_path}")
