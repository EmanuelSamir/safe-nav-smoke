"""
save_gp_rollouts.py
-------------------
Runs only the Gaussian Process baseline on smoke data and saves the results
in the SAME npz format as save_rnp_rollouts.py, so benchmark.py can load
them with key_prefix="gp".

Key design decisions
--------------------
* No RNP / Multistep: this script exists only to generate the GP baseline.
* Time feature: the GP uses Δt = (t_idx - t_context_start) * dt as its
  temporal coordinate so that the kernel operates on a physically meaningful,
  always-positive time axis that resets for each episode.
* Output format (per episode .npz):
    time_steps                          int array of evaluated t_current values
    t_{t}_gt_horizon                    (horizon, H, W)  float16
    t_{t}_gp_sample                     (horizon, num_samples, H_out, W_out) float16
    t_{t}_gp_mean                       (horizon, H_out, W_out)  float16
    t_{t}_gp_std                        (horizon, H_out, W_out)  float16
    t_{t}_gp_latency                    float  ms per horizon-step
    eval_shape                          [H_out, W_out]
"""

import sys
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import time
import argparse

# Add src to path
sys.path.append(os.getcwd())

from src.models.gaussian_process import GaussianProcess, OnlineKernel

log = logging.getLogger(__name__)


# =============================================================================
# GP Wrapper
# =============================================================================

class GPWrapper:
    """
    Thin wrapper around GaussianProcess that keeps track of a running
    time origin so the temporal feature is always Δt ≥ 0.

    The GP feature vector per observation is [x_real, y_real, Δt]
    where Δt = (t_idx - t_origin) * dt.
    """

    def __init__(self, dt: float):
        self.dt       = dt
        self.t_origin = 0          # set to t_start at the beginning of each episode
        self._init_gp()

    def _init_gp(self):
        self.gp = GaussianProcess(online=True, online_kernel=OnlineKernel.Matern)

    def reset(self, t_origin: int = 0):
        """
        Reset the GP state and set a new time origin.
        Call this before feeding each fresh context window so that
        Δt = 0 always corresponds to the first context frame.
        """
        self.t_origin = t_origin
        self._init_gp()

    def _delta_t(self, t_idx: int) -> float:
        """Δt ≥ 0 relative to the start of the current context window."""
        return (t_idx - self.t_origin) * self.dt

    def update(self, xs_real: np.ndarray, ys_real: np.ndarray,
               vals: np.ndarray, t_idx: int):
        """Feed one frame of context observations into the GP."""
        dt_val = self._delta_t(t_idx)
        ts     = np.full_like(xs_real, dt_val, dtype=np.float64)
        X      = np.stack([xs_real.flatten(),
                           ys_real.flatten(),
                           ts.flatten()], axis=1)           # (P, 3)
        # track_data requires both arrays to be the same ndim:
        # X is 2D (P, 3) so y must be 2D (P, 1)
        y      = vals.flatten().reshape(-1, 1).astype(np.float64)
        self.gp.track_data(X, y)
        self.gp.update()

    def predict(self, xs_real: np.ndarray, ys_real: np.ndarray,
                t_idx: int):
        """Predict at query points for a future time step."""
        dt_val = self._delta_t(t_idx)
        ts     = np.full_like(xs_real, dt_val, dtype=np.float64)
        X_q    = np.stack([xs_real.flatten(),
                           ys_real.flatten(),
                           ts.flatten()], axis=1)           # (P, 3)
        mean, std = self.gp.predict(X_q)
        # OnlineGP returns (P, 1) arrays — flatten to (P,)
        return mean.flatten(), std.flatten()


# =============================================================================
# Observation helpers
# =============================================================================

def get_context_np(ep_data: np.ndarray, t_step: int,
                   res_val: float, context_points: int):
    """
    Sample `context_points` random pixel observations from frame t_step.
    Returns real-scale (x, y, values) arrays — no torch needed.
    """
    frame = ep_data[t_step]          # (H, W)
    H, W  = frame.shape

    y_idxs = np.random.randint(0, H, size=context_points)
    x_idxs = np.random.randint(0, W, size=context_points)

    return {
        'xs':   x_idxs * res_val,    # (P,)  real-scale
        'ys':   y_idxs * res_val,
        'vals': frame[y_idxs, x_idxs].astype(np.float32),
    }


def build_query_grid(H_grid: int, W_grid: int,
                     res_val: float, downsample_factor: int = 1):
    """
    Build a downsampled spatial query grid.
    Uses indexing='ij' → y along first axis (row), x along second (col).

    Returns:
        xs_real, ys_real : flattened real-scale coordinates  (P,)
        shape            : (H_out, W_out)
    """
    y_range = np.arange(0, H_grid, downsample_factor)
    x_range = np.arange(0, W_grid, downsample_factor)

    y_idxs, x_idxs = np.meshgrid(y_range, x_range, indexing='ij')

    xs_real = x_idxs.flatten() * res_val
    ys_real = y_idxs.flatten() * res_val

    return xs_real, ys_real, (len(y_range), len(x_range))


# =============================================================================
# Data loader
# =============================================================================

def load_data(data_path: str):
    print(f"Loading data from {data_path}...")
    try:
        raw = np.load(data_path)
        if 'smoke_data' not in raw:
            raise KeyError("'smoke_data' not found in npz")
        smoke_data = raw['smoke_data']                       # (E, T, H, W)
        x_size  = float(raw.get('x_size',     50.0))
        y_size  = float(raw.get('y_size',     50.0))
        res_val = float(raw.get('resolution',  1.0))
        dt      = float(raw.get('dt',          0.1))
        return smoke_data, x_size, y_size, res_val, dt
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Save GP baseline rollouts in benchmark-compatible npz format."
    )
    parser.add_argument('--data_path',      type=str, required=True)
    parser.add_argument('--output_dir',     type=str, default='saved_rollouts/run_gp')
    parser.add_argument('--horizon',        type=int, default=15,
                        help='Forecast horizon (steps)')
    parser.add_argument('--num_episodes',   type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--stride',         type=int, default=3,
                        help='Stride between evaluated time steps')
    parser.add_argument('--num_samples',    type=int, default=10,
                        help='Gaussian samples drawn from GP posterior per step')
    parser.add_argument('--downsample_eval',type=int, default=1,
                        help='Spatial downsample factor for query grid (speeds up GP)')
    parser.add_argument('--context_points', type=int, default=100,
                        help='Random context points fed to GP per frame')
    parser.add_argument('--context_window', type=int, default=10,
                        help='Number of past frames fed as context to the GP at each '
                             'evaluation step (mirrors RNP rolling context window)')
    parser.add_argument('--t_start',        type=int, default=10,
                        help='First time step to start rolling from (must be >= context_window)')
    parser.add_argument('--t_end_max',      type=int, default=None,
                        help='Hard cap on t_end per episode (None = full trajectory)')

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke_data, x_size, y_size, res_val, dt = load_data(args.data_path)
    _, H_grid, W_grid = smoke_data[0].shape

    num_eval = min(args.num_episodes, len(smoke_data))
    indices  = np.linspace(0, len(smoke_data) - 1, num_eval, dtype=int)

    # Pre-build the query grid (same for every episode)
    xs_q, ys_q, (H_out, W_out) = build_query_grid(
        H_grid, W_grid, res_val, args.downsample_eval
    )
    print(f"Query grid: {H_out}×{W_out} = {H_out*W_out} points  "
          f"(downsample={args.downsample_eval})")

    rng = np.random.default_rng()

    for ep_idx in tqdm(indices, desc="Episodes"):
        ep_data = smoke_data[ep_idx]            # (T, H, W)
        T_total = ep_data.shape[0]

        t_start = args.t_start
        t_end   = T_total - args.horizon - 1
        if args.t_end_max is not None:
            t_end = min(t_end, args.t_end_max)

        if t_end <= t_start:
            print(f"  Skipping ep {ep_idx} (too short)")
            continue

        stride     = args.stride
        time_steps = list(range(t_start, t_end, stride))

        save_dict = {
            'ep_idx':     ep_idx,
            'time_steps': np.array(time_steps),
            'eval_shape': np.array([H_out, W_out]),
        }

        gp = GPWrapper(dt=dt)
        ctx_window = args.context_window

        # Ensure t_start leaves enough room for the context window
        effective_t_start = max(t_start, ctx_window)

        # ── Rolling forecast: at each stride step, reset GP with the last
        #    context_window frames, then predict horizon steps ahead.
        #    This exactly mirrors the RNP rolling context setup.
        for t_current in tqdm(range(effective_t_start, t_end, stride),
                              desc=f"Ep {ep_idx}", leave=False):

            # --- Reset GP and feed the last `context_window` frames ---
            ctx_t0 = t_current - ctx_window + 1   # first frame of context window
            gp.reset(t_origin=ctx_t0)             # Δt = 0 at the start of context

            for t_ctx in range(ctx_t0, t_current + 1):
                ctx = get_context_np(ep_data, t_ctx, res_val, args.context_points)
                gp.update(ctx['xs'], ctx['ys'], ctx['vals'], t_idx=t_ctx)

            # ── Ground truth horizon ──────────────────────────────────────────
            gt_frames = ep_data[t_current + 1 : t_current + 1 + args.horizon]
            save_dict[f't_{t_current}_gt_horizon'] = gt_frames.astype(np.float16)

            # ── GP forecast ───────────────────────────────────────────────────
            t0 = time.time()
            mean_list, std_list = [], []
            for h in range(1, args.horizon + 1):
                m, s = gp.predict(xs_q, ys_q, t_idx=t_current + h)
                mean_list.append(m.reshape(H_out, W_out).astype(np.float32))
                std_list.append( s.reshape(H_out, W_out).astype(np.float32))

            latency_per_step = (time.time() - t0) * 1000.0 / args.horizon

            gp_mean = np.stack(mean_list, axis=0).astype(np.float16)  # (H, H_out, W_out)
            gp_std  = np.stack(std_list,  axis=0).astype(np.float16)

            # Posterior samples: N(mean, std)  → (horizon, num_samples, H_out, W_out)
            noise   = rng.standard_normal(
                (args.horizon, args.num_samples, H_out, W_out)
            ).astype(np.float32)
            gp_sample = (
                gp_mean[:, None, :, :] + gp_std[:, None, :, :] * noise
            ).astype(np.float16)

            save_dict[f't_{t_current}_gp_sample']  = gp_sample
            save_dict[f't_{t_current}_gp_mean']    = gp_mean
            save_dict[f't_{t_current}_gp_std']     = gp_std
            save_dict[f't_{t_current}_gp_latency'] = latency_per_step

        out_file = out_dir / f"ep_{ep_idx}_gp_rollouts.npz"
        np.savez_compressed(out_file, **save_dict)
        print(f"  Saved → {out_file}")

    print(f"\nAll GP rollouts saved in {out_dir}")
    print("To add GP to benchmark.py, use:")
    print(f'  {{"folder": "{out_dir}", "label": "GP", "key_prefix": "gp"}}')


if __name__ == "__main__":
    main()
