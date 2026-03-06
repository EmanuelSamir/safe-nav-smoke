"""
save_rollouts.py — Unified rollout generator for all trained models.

  --model_type fno_3d         : FNO3d (h_ctx-frame seed, true 3D spectral conv)
  --model_type conv_lstm      : ConvLSTM (stateless, h_ctx-frame seed)

Output per episode (compressed NPZ):
  ep_idx                                 : int
  time_steps                             : (N,)  int
  gt_full                                : (T, H, W)  float16
  sample                                 : (N, horizon, num_samples, H, W)  float16
  mean                                   : (N, horizon, H, W) OR (N, horizon, num_samples, H, W) float16
  std                                    : (N, horizon, H, W) OR (N, horizon, num_samples, H, W) float16
  latency                                : (N,) float

Usage:
  python scripts/save_rollouts.py \
      --model_type fno_3d \
      --ckpt outputs/2026-02-24/XX/checkpoints/best_model.pt \
      --data_path data/playback_data/test_global_source_100_100.npz \
      --output_dir saved_rollouts/fno_3d_h10p5
"""

import sys
import os
import time
import argparse
import logging

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.getcwd())

from src.models.shared.observations import Obs

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(data_path: str):
    print(f"Loading data from {data_path}...")
    raw = np.load(data_path)
    if 'smoke_data' not in raw:
        raise KeyError("'smoke_data' not found in npz")
    smoke_data = raw['smoke_data']          # (E, T, H, W)
    x_size  = float(raw.get('x_size',  50.0))
    y_size  = float(raw.get('y_size',  50.0))
    res_val = float(raw.get('resolution', 1.0))
    return smoke_data, x_size, y_size, res_val


def get_dense_obs(ep_data, t_step, device, x_size, res_val, y_size, context=True):
    """Build a dense Obs for time step t_step from episode data."""
    frame = ep_data[t_step:t_step+1]           # (1, H, W)
    _, H_grid, W_grid = frame.shape

    y_idxs, x_idxs = np.meshgrid(np.arange(H_grid), np.arange(W_grid), indexing='ij')
    y_flat = y_idxs.flatten()
    x_flat = x_idxs.flatten()
    t_flat = np.zeros(len(x_flat), dtype=int)

    vals = frame[t_flat, y_flat, x_flat] if context else np.zeros_like(x_flat, dtype=np.float32)

    xs_norm = 2.0 * (x_flat * res_val) / x_size - 1.0
    ys_norm = 2.0 * (y_flat * res_val) / y_size - 1.0

    xs  = torch.tensor(xs_norm, dtype=torch.float32).view(1, 1, -1, 1).to(device)
    ys  = torch.tensor(ys_norm, dtype=torch.float32).view(1, 1, -1, 1).to(device)
    vals_t = torch.tensor(vals,  dtype=torch.float32).view(1, 1, -1, 1).to(device)

    return Obs(xs=xs, ys=ys, values=vals_t)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_temporal_model(path: str, device, model_cls, cfg_cls, label: str):
    """Generic loader for temporal models (same checkpoint structure)."""
    print(f"Loading {label} from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp   = ckpt['hyper_parameters']

    if isinstance(hp, dict) and 'training' in hp:
        m = hp['training']['model']
    else:
        m = hp if isinstance(hp, dict) else {}

    # Build config — accept any subset of fields the checkpoint may have
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cfg_cls)}
    cfg_kwargs = {k: v for k, v in m.items() if k in valid_fields}
    cfg = cfg_cls(**cfg_kwargs)

    model = model_cls(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  {label}  h_ctx={cfg.h_ctx}  h_pred={cfg.h_pred}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model


def load_fno_3d(path: str, device):
    from src.models.fno_3d import FNO3d, FNO3dConfig
    return _load_temporal_model(path, device, FNO3d, FNO3dConfig, "FNO3d")


def load_conv_lstm(path: str, device):
    from src.models.conv_lstm import ConvLSTMModel, ConvLSTMConfig
    return _load_temporal_model(path, device, ConvLSTMModel, ConvLSTMConfig, "ConvLSTMModel")


# ---------------------------------------------------------------------------
# Per-model rollout helpers
# ---------------------------------------------------------------------------

def temporal_model_rollout(model, ep_data, t_current: int, horizon: int,
                          num_samples: int, H_grid: int, W_grid: int, device):
    """
    Rollout for temporal models which need h_ctx consecutive frames.

    Builds seed_frames = ep_data[t_current-h_ctx+1 : t_current+1]  (h_ctx, H, W)
    If t_current < h_ctx-1 the window is zero-padded at the start.
    seed_t_start is the absolute index of seed_frames[0].
    """
    h_ctx = model.cfg.h_ctx

    t_left   = t_current - h_ctx + 1
    t_left_c = max(0, t_left)               # clamp to valid range

    frames = ep_data[t_left_c : t_current + 1].astype(np.float32)   # (<=h_ctx, H, W)

    # Zero-pad if episode starts in the middle of the context window
    if t_left < 0:
        pad = np.zeros((-t_left, H_grid, W_grid), dtype=np.float32)
        frames = np.concatenate([pad, frames], axis=0)               # (h_ctx, H, W)

    seed = torch.tensor(frames, dtype=torch.float32, device=device).unsqueeze(0)  # (1, h_ctx, H, W)

    t0 = time.time()
    with torch.no_grad():
        preds_sample = model.autoregressive_forecast(
            seed,
            seed_t_start=t_left_c,
            horizon=horizon,
            num_samples=num_samples,
            mode='sample',
        )
        preds_mean = model.autoregressive_forecast(
            seed,
            seed_t_start=t_left_c,
            horizon=horizon,
            num_samples=1,
            mode='mean',
        )
    latency_ms = (time.time() - t0) * 1000.0 / num_samples

    sample_imgs = np.stack([p['sample'] for p in preds_sample], axis=0)  # (horizon, S, H, W)
    sample_mean_imgs = np.stack([p['mean']   for p in preds_sample], axis=0)  # (horizon, S, H, W)
    sample_std_imgs  = np.stack([p['std']    for p in preds_sample], axis=0)  # (horizon, S, H, W)
    
    mean_imgs = np.stack([p['mean'] for p in preds_mean], axis=0)[:, 0]  # (horizon, H, W)
    std_imgs  = np.stack([p['std']  for p in preds_mean], axis=0)[:, 0]  # (horizon, H, W)

    return sample_imgs, sample_mean_imgs, sample_std_imgs, mean_imgs, std_imgs, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Save autoregressive rollouts for any model.")
    parser.add_argument('--ckpt',          type=str, required=True,  help="Path to model checkpoint (.pt)")
    parser.add_argument('--model_type',    type=str, required=True,
                        help="fno_3d | conv_lstm")
    parser.add_argument('--data_path',     type=str, required=True,  help="Path to .npz dataset")
    parser.add_argument('--output_dir',    type=str, default='saved_rollouts')
    parser.add_argument('--horizon',       type=int, default=15,     help="Rollout horizon (steps)")
    parser.add_argument('--num_episodes',  type=int, default=None,   help="Max episodes (None=all)")
    parser.add_argument('--stride',        type=int, default=3,      help="Evaluation stride (steps)")
    parser.add_argument('--num_samples',   type=int, default=10,     help="MC samples per forecast")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model --------------------------------------------------------
    if args.model_type == 'fno_3d':
        model = load_fno_3d(args.ckpt, device)
    elif args.model_type == 'conv_lstm':
        model = load_conv_lstm(args.ckpt, device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # ---- Load data ---------------------------------------------------------
    smoke_data, x_size, y_size, res_val = load_data(args.data_path)
    _, H_grid, W_grid = smoke_data[0].shape
    print(f"Data shape: {smoke_data.shape}  Grid: {H_grid}x{W_grid}")

    n_ep = min(args.num_episodes, len(smoke_data)) if args.num_episodes else len(smoke_data)
    indices = np.random.choice(len(smoke_data), n_ep, replace=False)

    # ---- Episode loop ------------------------------------------------------
    for ep_idx in tqdm(indices, desc="Episodes"):
        ep_data  = smoke_data[ep_idx]   # (T, H, W)
        T_total  = ep_data.shape[0]

        t_start  = 10
        t_end    = T_total - args.horizon - 1

        if t_end <= t_start:
            print(f"Skipping ep {ep_idx} (too short)")
            continue

        time_steps = list(range(t_start, t_end, args.stride))
        save_dict  = {
            'ep_idx':     ep_idx,
            'time_steps': np.array(time_steps),
            'gt_full':    ep_data.astype(np.float16)
        }

        sample_list = []
        sample_mean_list = []
        sample_std_list = []
        mean_list = []
        std_list = []
        lat_list = []

        # ---- Temporal Models: stateless, h_ctx-frame seed --------------
        for t_current in tqdm(time_steps, desc=f"Ep {ep_idx}", leave=False):
            sample_imgs, sample_mean_imgs, sample_std_imgs, mean_imgs, std_imgs, lat = temporal_model_rollout(
                model, ep_data, t_current, args.horizon, args.num_samples,
                H_grid, W_grid, device)

            sample_list.append(sample_imgs)
            sample_mean_list.append(sample_mean_imgs)
            sample_std_list.append(sample_std_imgs)
            mean_list.append(mean_imgs)
            std_list.append(std_imgs)
            lat_list.append(lat)

        save_dict['sample'] = np.stack(sample_list)
        save_dict['sample_mean'] = np.stack(sample_mean_list)
        save_dict['sample_std'] = np.stack(sample_std_list)
        save_dict['mean'] = np.stack(mean_list)
        save_dict['std'] = np.stack(std_list)
        save_dict['latency'] = np.array(lat_list)

        out_file = out_dir / f"ep_{ep_idx}_rollouts.npz"
        np.savez_compressed(out_file, **save_dict)
        print(f"Saved: {out_file}")

    print(f"\nAll rollouts saved → {out_dir}")


if __name__ == "__main__":
    main()
