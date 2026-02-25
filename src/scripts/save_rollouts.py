"""
save_rollouts.py — Unified rollout generator for all trained models.

Supports:
  --model_type rnp            : RNP (recurrent, stateful)
  --model_type rnp_multistep  : RNP Multistep
  --model_type fno            : FNO2d (stateless, 1-frame seed)
  --model_type fno_3d         : FNO3d (h_ctx-frame seed, true 3D spectral conv)

Output per episode (compressed NPZ):
  ep_idx                                 : int
  time_steps                             : (N,)  int
  t_{t}_gt_horizon                       : (horizon, H, W)  float16
  t_{t}_{tag}_sample                     : (horizon, num_samples, H, W)  float16
  t_{t}_{tag}_mean                       : (horizon, H, W)  float16
  t_{t}_{tag}_std                        : (horizon, H, W)  float16
  t_{t}_{tag}_latency                    : float  ms/sample

Usage:
  python src/scripts/save_rollouts.py \
      --model_type fno_3d \
      --ckpt outputs/2026-02-24/XX/checkpoints/best_model.pt \
      --data_path data/playback_data/test_global_source_100_100.npz \
      --output_dir saved_rollouts/fno_3d_h10p5 \
      --tag fno_3d
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

def load_rnp(path: str, model_type: str, device):
    from src.models.model_free.rnp import RNP, RNPConfig
    from src.models.model_free.rnp_multistep import RNPMultistep

    print(f"Loading {model_type} from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_raw = ckpt['hyper_parameters']

    # hyper_parameters may be a flat RNPConfig dict or a nested hydra dict
    if isinstance(cfg_raw, dict) and 'training' in cfg_raw:
        model_hp = cfg_raw['training']['model']
        cfg = RNPConfig(**{k: v for k, v in model_hp.items()
                          if k in RNPConfig.__dataclass_fields__})
    else:
        cfg = RNPConfig(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw

    if model_type == 'rnp_multistep':
        fh = cfg_raw.get('training', {}).get('model', {}).get('forecast_horizon', 5) \
            if isinstance(cfg_raw, dict) else 5
        model = RNPMultistep(cfg, forecast_horizon=fh)
    else:
        model = RNP(cfg)

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def load_fno(path: str, device):
    from src.models.model_free.fno import FNO2d, FNOConfig

    print(f"Loading FNO from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp   = ckpt['hyper_parameters']

    if isinstance(hp, dict) and 'training' in hp:
        m = hp['training']['model']
    else:
        m = hp if isinstance(hp, dict) else {}

    fno_cfg = FNOConfig(
        modes1=m.get('modes1', 12),
        modes2=m.get('modes2', 12),
        width=m.get('width', 32),
        use_grid=m.get('use_grid', True),
        n_layers=m.get('n_layers', 4),
        min_std=m.get('min_std', 1e-4),
        forecast_horizon=m.get('forecast_horizon', 1),
        use_uncertainty_input=m.get('use_uncertainty_input', False),
    )
    model = FNO2d(fno_cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  FNO2d  H={fno_cfg.forecast_horizon}  "
          f"use_uncertainty_input={fno_cfg.use_uncertainty_input}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model


def _load_temporal_model(path: str, device, model_cls, cfg_cls, label: str):
    """Generic loader for FNO3d and TemporalFNO (same checkpoint structure)."""
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
    from src.models.model_free.fno_3d import FNO3d, FNO3dConfig
    return _load_temporal_model(path, device, FNO3d, FNO3dConfig, "FNO3d")


# ---------------------------------------------------------------------------
# Per-model rollout helpers
# ---------------------------------------------------------------------------

def fno_rollout(model, ep_data, t_current: int, horizon: int,
                num_samples: int, H_grid: int, W_grid: int, device):
    """
    Stateless FNO2d rollout from a single frame at t_current.
    seed: (1, H, W, 1)  — one smoke frame
    """
    frame = ep_data[t_current]                                   # (H, W)
    seed  = torch.tensor(frame, dtype=torch.float32, device=device)
    seed  = seed.unsqueeze(0).unsqueeze(-1)                      # (1, H, W, 1)

    t0 = time.time()
    with torch.no_grad():
        preds = model.autoregressive_forecast(
            seed, horizon=horizon, num_samples=num_samples)
    latency_ms = (time.time() - t0) * 1000.0 / num_samples

    sample_imgs = np.stack([p['sample'] for p in preds], axis=0)  # (horizon, S, H, W)
    mean_imgs   = np.stack([p['mean']   for p in preds], axis=0)  # (horizon, H, W)
    std_imgs    = np.stack([p['std']    for p in preds], axis=0)  # (horizon, H, W)
    return sample_imgs, mean_imgs, std_imgs, latency_ms


def temporal_fno_rollout(model, ep_data, t_current: int, horizon: int,
                          num_samples: int, H_grid: int, W_grid: int, device):
    """
    Rollout for TemporalFNO / FNO3d which need h_ctx consecutive frames.

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
        preds = model.autoregressive_forecast(
            seed,
            seed_t_start=t_left_c,
            horizon=horizon,
            num_samples=num_samples,
        )
    latency_ms = (time.time() - t0) * 1000.0 / num_samples

    sample_imgs = np.stack([p['sample'] for p in preds], axis=0)  # (horizon, S, H, W)
    mean_imgs   = np.stack([p['mean']   for p in preds], axis=0)  # (horizon, H, W)
    std_imgs    = np.stack([p['std']    for p in preds], axis=0)  # (horizon, H, W)
    return sample_imgs, mean_imgs, std_imgs, latency_ms


def rnp_rollout(model, ep_data, t_current: int, horizon: int,
                num_samples: int, H_grid: int, W_grid: int, device,
                x_size: float, y_size: float, res_val: float,
                running_state, model_type: str):
    """RNP / RNP Multistep rollout at t_current using provided running_state."""
    ctx_obs   = get_dense_obs(ep_data, t_current, device, x_size, res_val, y_size, context=True)
    query_obs = get_dense_obs(ep_data, t_current, device, x_size, res_val, y_size, context=False)

    def clone_state(state):
        if state is None:
            return None
        return [(l[0].clone(), l[1].clone()) for l in state]

    state_clone = clone_state(running_state)

    t0 = time.time()
    with torch.no_grad():
        preds = model.autoregressive_forecast(
            state=state_clone,
            context_obs=ctx_obs,
            target_obs=query_obs,
            horizon=horizon,
            num_samples=num_samples,
        )
    latency_ms = (time.time() - t0) * 1000.0 / num_samples

    sample_imgs, mean_imgs, std_imgs = [], [], []
    for step_preds in preds:
        s    = step_preds['sample'].values.squeeze(-1).squeeze(1)   # (S, P)
        m    = step_preds['mean'].values.squeeze(-1).squeeze(1)     # (1||S, P)
        std  = step_preds['std'].values.squeeze(-1).squeeze(1)      # (1||S, P)
        sample_imgs.append(s.view(num_samples, H_grid, W_grid).detach().cpu().to(torch.float16).numpy())
        mean_imgs.append(m.view(H_grid, W_grid).detach().cpu().to(torch.float16).numpy())
        std_imgs.append(std.view(H_grid, W_grid).detach().cpu().to(torch.float16).numpy())

    sample_imgs = np.stack(sample_imgs, axis=0)   # (horizon, S, H, W)
    mean_imgs   = np.stack(mean_imgs,   axis=0)   # (horizon, H, W)
    std_imgs    = np.stack(std_imgs,    axis=0)   # (horizon, H, W)

    return sample_imgs, mean_imgs, std_imgs, latency_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Save autoregressive rollouts for any model.")
    parser.add_argument('--ckpt',          type=str, required=True,  help="Path to model checkpoint (.pt)")
    parser.add_argument('--model_type',    type=str, required=True,
                        help="rnp | rnp_multistep | fno | fno_3d")
    parser.add_argument('--data_path',     type=str, required=True,  help="Path to .npz dataset")
    parser.add_argument('--output_dir',    type=str, default='saved_rollouts')
    parser.add_argument('--horizon',       type=int, default=15,     help="Rollout horizon (steps)")
    parser.add_argument('--num_episodes',  type=int, default=None,   help="Max episodes (None=all)")
    parser.add_argument('--stride',        type=int, default=3,      help="Evaluation stride (steps)")
    parser.add_argument('--num_samples',   type=int, default=10,     help="MC samples per forecast")
    parser.add_argument('--tag',           type=str, default=None,
                        help="Key prefix in NPZ (defaults to model_type)")
    args = parser.parse_args()

    tag = args.tag or args.model_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model --------------------------------------------------------
    is_fno      = args.model_type == 'fno'
    is_temporal = args.model_type == 'fno_3d'

    if is_fno:
        model = load_fno(args.ckpt, device)
    elif is_temporal:
        model = load_fno_3d(args.ckpt, device)
    else:
        model = load_rnp(args.ckpt, args.model_type, device)

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
        }

        # ---- RNP: sequential state update ----------------------------------
        if not is_fno and not is_temporal:
            running_state = model.init_state(1, device)

            for t_current in tqdm(range(t_end), desc=f"Ep {ep_idx}", leave=False):
                ctx_obs = get_dense_obs(ep_data, t_current, device,
                                        x_size, res_val, y_size, context=True)
                with torch.no_grad():
                    out = model(running_state, context_obs=ctx_obs, target_obs=None)
                    running_state = out.state

                if t_current >= t_start and (t_current - t_start) % args.stride == 0:
                    gt = ep_data[t_current+1 : t_current+1+args.horizon]
                    save_dict[f't_{t_current}_gt_horizon'] = gt.astype(np.float16)

                    sample_imgs, mean_imgs, std_imgs, lat = rnp_rollout(
                        model, ep_data, t_current, args.horizon, args.num_samples,
                        H_grid, W_grid, device, x_size, y_size, res_val,
                        running_state, args.model_type)

                    save_dict[f't_{t_current}_{tag}_sample']  = sample_imgs
                    save_dict[f't_{t_current}_{tag}_mean']    = mean_imgs
                    save_dict[f't_{t_current}_{tag}_std']     = std_imgs
                    save_dict[f't_{t_current}_{tag}_latency'] = lat

        # ---- FNO2d: stateless, 1-frame seed --------------------------------
        elif is_fno:
            for t_current in tqdm(time_steps, desc=f"Ep {ep_idx}", leave=False):
                gt = ep_data[t_current+1 : t_current+1+args.horizon]
                save_dict[f't_{t_current}_gt_horizon'] = gt.astype(np.float16)

                sample_imgs, mean_imgs, std_imgs, lat = fno_rollout(
                    model, ep_data, t_current, args.horizon, args.num_samples,
                    H_grid, W_grid, device)

                save_dict[f't_{t_current}_{tag}_sample']  = sample_imgs
                save_dict[f't_{t_current}_{tag}_mean']    = mean_imgs
                save_dict[f't_{t_current}_{tag}_std']     = std_imgs
                save_dict[f't_{t_current}_{tag}_latency'] = lat

        # ---- TemporalFNO / FNO3d: stateless, h_ctx-frame seed --------------
        else:   # is_temporal
            for t_current in tqdm(time_steps, desc=f"Ep {ep_idx}", leave=False):
                gt = ep_data[t_current+1 : t_current+1+args.horizon]
                save_dict[f't_{t_current}_gt_horizon'] = gt.astype(np.float16)

                sample_imgs, mean_imgs, std_imgs, lat = temporal_fno_rollout(
                    model, ep_data, t_current, args.horizon, args.num_samples,
                    H_grid, W_grid, device)

                save_dict[f't_{t_current}_{tag}_sample']  = sample_imgs
                save_dict[f't_{t_current}_{tag}_mean']    = mean_imgs
                save_dict[f't_{t_current}_{tag}_std']     = std_imgs
                save_dict[f't_{t_current}_{tag}_latency'] = lat

        out_file = out_dir / f"ep_{ep_idx}_rollouts.npz"
        np.savez_compressed(out_file, **save_dict)
        print(f"Saved: {out_file}")

    print(f"\nAll rollouts saved → {out_dir}")


if __name__ == "__main__":
    main()
