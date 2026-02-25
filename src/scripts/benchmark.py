# %%
# =============================================================================
# BENCHMARK — Compare RNP rollout variants vs. Persistence baseline
# Run interactively cell-by-cell in VS Code / Jupyter
# =============================================================================
import numpy as np
import time
import matplotlib.pyplot as plt
import tqdm
import os
from scipy import stats
import torch

# =============================================================================
# SECTION 0 — Configuration
#   Edit this cell to point at your three rollout folders and give them names.
# =============================================================================
# %%

RUNS = [
    {
        "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_rnp_bias",
        "label":  "RNP (bias)",
    },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_rnp_no_bias",
    #     "label":  "RNP (no bias)",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_multistep_bias",
    #     "label":  "Multistep (bias)",
    # },
    {
        "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_multistep_no_bias",
        "label":  "Multistep (no bias)",
    },
    {
        "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_gp",
        "label":  "GP (no bias)",
    },
    {
        "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_bias",
        "label":  "FNO (bias)",
    },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_no_bias",
    #     "label":  "FNO (no bias)",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_ms_h3",
    #     "label":  "FNO multistep 3",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_ms_h5",
    #     "label":  "FNO multistep 5",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_ms_h8",
    #     "label":  "FNO multistep 8",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_bias_sample",
    #     "label":  "FNO bias sample",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_uncertainty",
    #     "label":  "FNO uncertainty",
    # },
    {
        "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d",
        "label":  "FNO-3D",
    },
]

MAX_HORIZON = 15      # max horizon stored in the .npz files (1-indexed count)
CVAR_LEVELS  = [0.75, 0.90, 0.95]   # CVaR α levels to evaluate

# =============================================================================
# SECTION 1 — Helper utilities
# =============================================================================
# %%

def detect_prefix_from_data(data, time_steps) -> str:
    """
    Auto-detect the key prefix by scanning the first available time step.
    Works regardless of whether the tag is 'fno', 'fno_bias', 'rnp', etc.
    Returns the prefix (everything between 't_{t}_' and '_sample').
    """
    for t in time_steps[:5]:
        for key in data.keys():
            prefix = f"t_{t}_"
            suffix = "_sample"
            if key.startswith(prefix) and key.endswith(suffix):
                return key[len(prefix):-len(suffix)]
    return "rnp"   # safe fallback


def detect_sample_key(folder: str) -> str:
    """Legacy heuristic — kept for backwards compat but prefer detect_prefix_from_data."""
    if "multistep" in folder:
        return "multistep_sample"
    if "gp" in folder:
        return "gp_sample"
    if "fno" in folder:
        return "fno_sample"
    return "rnp_sample"


def cvar(pred_mean: np.ndarray, pred_std: np.ndarray, alpha: float) -> np.ndarray:
    """
    CVaR_α  (Expected Shortfall) of a 1-D array at confidence level α.
    Returns the mean of the worst (1-α) fraction of values.
    A higher value means heavier right-tail risk.
    """
    cvar_vals = pred_mean + pred_std * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)
    cvar_vals = np.clip(cvar_vals, 0.0, 1.0)
    return cvar_vals

def coverage_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.clip(gt.astype(np.float32) - pred.astype(np.float32), 0.0, None)

def conservatism_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.clip(pred.astype(np.float32) - gt.astype(np.float32), 0.0, None)

def load_run_data(run: dict) -> list[dict]:
    """
    Load every .npz episode file from run['folder'].
    Prefix is auto-detected from the actual npz keys (robust for any tag).
    An explicit 'key_prefix' in run overrides auto-detection.
    """
    folder = run["folder"]
    files  = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
    episodes = []
    for fname in files:
        path = os.path.join(folder, fname)
        data = np.load(path)
        time_steps = data["time_steps"]

        if "key_prefix" in run:
            pfx = run["key_prefix"]
        else:
            pfx = detect_prefix_from_data(data, time_steps)

        episodes.append({
            "data":              data,
            "time_steps":        time_steps,
            "sample_key_prefix": pfx,
        })
    return episodes

def fuse_gaussians(mu_tensor, sigma_tensor):
    """
    mu_tensor: (horizon, 10, H, W)
    sigma_tensor: (horizon, 10, H, W)
    """
    # 1. Calcular varianzas y precisiones
    varianzas = np.square(sigma_tensor)
    # Evitar división por cero sumando un epsilon muy pequeño si es necesario
    precisiones = 1.0 / (varianzas + 1e-8)
    
    # 2. Varianza combinada (Suma de precisiones invertida)
    # Colapsamos el eje 1 (los 10 modelos)
    precision_total = np.sum(precisiones, axis=1)
    var_final = 1.0 / precision_total
    
    # 3. Media combinada (Promedio ponderado por precisión)
    mu_final = np.sum(mu_tensor * precisiones, axis=1) * var_final
    
    return mu_final, np.sqrt(var_final)

def collect_metrics_for_horizon(episodes: list[dict], h: int) -> dict:
    """
    Aggregate per-episode scalar metrics at horizon index h (0-indexed).

    Returns dict with lists:
        model_mae, model_coverage_mean,
        persistence_mae, persistence_coverage_mean,
        model_cvar_{alpha}, persistence_cvar_{alpha}   (for each CVAR_LEVELS)
        model_is_more_conservative                      (list of bools)
        latencies_ms
    """
    result = {
        "model_mae":                [],
        "model_coverage_mean":      [],
        "model_conservatism_mean":  [],
        "persistence_mae":          [],
        "persistence_coverage_mean":[],
        "persistence_conservatism_mean":[],
        "model_is_more_conservative":[],
        "model_is_more_comprehensive":[],
        "latencies_ms":             [],
    }
    for a in CVAR_LEVELS:
        result[f"model_mae_cvar_{a}"]       = []
        result[f"model_coverage_cvar_{a}"]  = []
        result[f"model_conservatism_cvar_{a}"]  = []

    for ep in episodes:
        data       = ep["data"]
        time_steps = ep["time_steps"]
        pfx        = ep["sample_key_prefix"]

        for i, t in enumerate(time_steps):
            sample_key      = f"t_{t}_{pfx}_sample"
            gt_key          = f"t_{t}_gt_horizon"
            latency_key     = f"t_{t}_{pfx}_latency"
            sample_mean_key = f"t_{t}_{pfx}_mean"
            sample_std_key  = f"t_{t}_{pfx}_std"

            if sample_key not in data or gt_key not in data:
                continue
            if h >= data[gt_key].shape[0]:
                continue

            rollout   = data[sample_key].astype(np.float32)   # (horizon, N, H, W)
            gt_all    = data[gt_key].astype(np.float32)        # (horizon, H, W)
            gt_h      = gt_all[h]                              # (H, W)
            samples_h = rollout[h]                             # (N, H, W)
            mean_h    = samples_h.mean(axis=0)                 # (H, W) ensemble mean

            # Distribution parameters at horizon h (model μ, σ)
            has_dist = sample_mean_key in data and sample_std_key in data
            mu_h     = data[sample_mean_key].astype(np.float32)[h] if has_dist else mean_h
            std_h    = data[sample_std_key ].astype(np.float32)[h] if has_dist else samples_h.std(axis=0)

            # — Model metrics —
            mae_m = float(np.abs(mean_h - gt_h).mean())
            cov_m = float(coverage_error(mean_h, gt_h).mean())
            cons_m = float(conservatism_error(mean_h, gt_h).mean())
            result["model_mae"].append(mae_m)
            result["model_coverage_mean"].append(cov_m)
            result["model_conservatism_mean"].append(cons_m)

            for a in CVAR_LEVELS:
                cvar_h = cvar(mu_h, std_h, a)
                result[f"model_mae_cvar_{a}"].append(
                    float(np.abs(cvar_h - gt_h).mean()))
                result[f"model_coverage_cvar_{a}"].append(
                    float(coverage_error(cvar_h, gt_h).mean()))
                result[f"model_conservatism_cvar_{a}"].append(
                    float(conservatism_error(cvar_h, gt_h).mean()))

            # — Latency —
            if latency_key in data:
                result["latencies_ms"].append(float(data[latency_key]))

            # — Persistence metrics —
            if i > 0:
                t_prev      = time_steps[i - 1]
                gt_prev_key = f"t_{t_prev}_gt_horizon"
                if gt_prev_key in data:
                    gt_prev_all = data[gt_prev_key].astype(np.float32)
                    gt_current  = gt_prev_all[-1]              # frame 'now' at step i
                    cov_p       = float(coverage_error(gt_current, gt_h).mean())
                    mae_p       = float(np.abs(gt_current - gt_h).mean())
                    cons_p      = float(conservatism_error(gt_current, gt_h).mean())
                    result["persistence_mae"].append(mae_p)
                    result["persistence_coverage_mean"].append(cov_p)
                    result["persistence_conservatism_mean"].append(cons_p)

                    result["model_is_more_conservative"].append(
                        float(cons_m) > float(cons_p))

                    result["model_is_more_comprehensive"].append(
                        float(cov_m) < float(cov_p))

    return result


# =============================================================================
# SECTION 2 — Load all run data
# =============================================================================
# %%

print("Loading rollout files …")
all_run_episodes = []
for run in RUNS:
    folder = run["folder"]
    if not os.path.isdir(folder):
        print(f"  [SKIP] {folder} does not exist")
        all_run_episodes.append(None)
        continue
    eps = load_run_data(run)
    all_run_episodes.append(eps)
    print(f"  {run['label']}: {len(eps)} episodes from {folder}")

print("Done.")


# =============================================================================
# SECTION 3 — Collect metrics across ALL horizons
# =============================================================================
# %%

all_metrics = []   # one dict per run, indexed by horizon

for run_idx, (run, episodes) in enumerate(zip(RUNS, all_run_episodes)):
    if episodes is None:
        all_metrics.append(None)
        continue

    per_horizon = {}  # h -> metric dict
    for h in tqdm.trange(MAX_HORIZON, desc=f"Metrics [{run['label']}]"):
        per_horizon[h] = collect_metrics_for_horizon(episodes, h)
    all_metrics.append(per_horizon)

print("Metric collection complete.")

# =============================================================================
# SECTION 4 — MAE across all horizons (line plot)
# =============================================================================
# %%

horizons = list(range(1, MAX_HORIZON + 1))  # 1-indexed for display
fig, axes = plt.subplots(3, 1, figsize=(5, 13))
fig.suptitle("MAE vs Forecast Horizon", fontsize=14)

import matplotlib.cm as cm
_cmap   = cm.get_cmap('rainbow', len(RUNS))
colors  = [_cmap(i) for i in range(len(RUNS))]
ls_model = "-"
ls_pers  = "--"

ax_mae, ax_cov, ax_cons = axes

persistence_counted = False
for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c = colors[run_idx]
    lbl = run["label"]

    mae_m  = [np.mean(per_horizon[h]["model_mae"])            if per_horizon[h]["model_mae"]            else np.nan for h in range(MAX_HORIZON)]
    cov_m  = [np.mean(per_horizon[h]["model_coverage_mean"])  if per_horizon[h]["model_coverage_mean"]  else np.nan for h in range(MAX_HORIZON)]
    cons_m = [np.mean(per_horizon[h]["model_conservatism_mean"]) if per_horizon[h]["model_conservatism_mean"] else np.nan for h in range(MAX_HORIZON)]

    if not persistence_counted:
        mae_p  = [np.mean(per_horizon[h]["persistence_mae"])      if per_horizon[h]["persistence_mae"]      else np.nan for h in range(MAX_HORIZON)]
        cov_p  = [np.mean(per_horizon[h]["persistence_coverage_mean"]) if per_horizon[h]["persistence_coverage_mean"] else np.nan for h in range(MAX_HORIZON)]
        cons_p = [np.mean(per_horizon[h]["persistence_conservatism_mean"]) if per_horizon[h]["persistence_conservatism_mean"] else np.nan for h in range(MAX_HORIZON)]
        ax_mae.plot(horizons, mae_p, color=c, ls=ls_pers,  marker="s", ms=4, label=f"Persistence MAE", alpha=0.6)
        ax_cov.plot(horizons, cov_p, color=c, ls=ls_pers,  marker="s", ms=4, label=f"Persistence Coverage", alpha=0.6)
        ax_cons.plot(horizons, cons_p, color=c, ls=ls_pers,  marker="s", ms=4, label=f"Persistence Conservatism", alpha=0.6)
        persistence_counted = True

    ax_mae.plot(horizons, mae_m, color=c, ls=ls_model, marker="o", ms=4, label=f"{lbl} — model")
    ax_cov.plot(horizons, cov_m, color=c, ls=ls_model, marker="o", ms=4, label=f"{lbl} — model")
    ax_cons.plot(horizons, cons_m, color=c, ls=ls_model, marker="o", ms=4, label=f"{lbl} — model")

ax_mae.set_xlabel("Horizon step"); ax_mae.set_ylabel("MAE")
ax_mae.set_title("Mean Absolute Error"); ax_mae.legend(fontsize=7); ax_mae.grid(alpha=0.3)

ax_cov.set_xlabel("Horizon step"); ax_cov.set_ylabel("Coverage error (mean)")
ax_cov.set_title("Coverage (under-prediction penalty)"); ax_cov.legend(fontsize=7); ax_cov.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 5 — CVaR comparison across all horizons
#   For path planning: a model is SAFER if its CVaR > ground truth CVaR
#   (i.e. it over-predicts risk in the tail → conservative for drone nav)
# =============================================================================
# %%

fig, axes = plt.subplots(1, len(CVAR_LEVELS), figsize=(5 * len(CVAR_LEVELS), 5), sharey=False)
fig.suptitle("CVaR (Expected Shortfall) by α — higher = heavier tail risk estimated", fontsize=13)

persistence_counted = False
for axi, a in enumerate(CVAR_LEVELS):
    ax = axes[axi]
    for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
        if per_horizon is None:
            continue
        c   = colors[run_idx]
        lbl = run["label"]

        cv_m = [np.mean(per_horizon[h][f"model_cvar_{a}"])       if per_horizon[h][f"model_cvar_{a}"]       else np.nan for h in range(MAX_HORIZON)]
        if not persistence_counted:
            cv_p = [np.mean(per_horizon[h][f"persistence_mae"]) if per_horizon[h][f"persistence_mae"] else np.nan for h in range(MAX_HORIZON)]
            ax.plot(horizons, cv_p, color=c, ls="--", marker="s", ms=4, label=f"Persistence — MAE", alpha=0.6)
            persistence_counted = True

        ax.plot(horizons, cv_m, color=c, ls="-",  marker="o", ms=4, label=f"{lbl} — model")

    ax.set_title(f"CVaR α={a}")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("CVaR value")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 6 — Conservatism rate (% of steps where model coverage > persistence)
#   For drone path planning: higher = model is more conservative / safer.
# =============================================================================
# %%

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Conservatism Rate vs Horizon\n(% timesteps where model coverage > persistence coverage)")

for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c   = colors[run_idx]
    lbl = run["label"]

    rates = []
    for h in range(MAX_HORIZON):
        vals = per_horizon[h]["model_is_more_conservative"]
        rates.append(np.mean(vals) * 100.0 if vals else np.nan)

    ax.plot(horizons, rates, color=c, ls="-", marker="o", ms=4, label=lbl)

ax.axhline(50, color="gray", ls=":", lw=1, label="50% (neutral)")
ax.set_xlabel("Horizon step")
ax.set_ylabel("% more conservative than persistence")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 7 — Latency bar plot (ms per sample)
# =============================================================================
# %%

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title("Inference Latency per Sample (ms)")

bar_positions = np.arange(len(RUNS))
bar_width     = 0.5

for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        ax.bar(bar_positions[run_idx], 0, color=colors[run_idx], alpha=0.7, label=run["label"])
        continue

    # Latency is per-time-step not per-horizon; grab from h=0
    lats = per_horizon[0]["latencies_ms"]
    mean_lat = np.mean(lats) if lats else 0.0
    std_lat  = np.std(lats)  if lats else 0.0

    ax.bar(
        bar_positions[run_idx], mean_lat,
        yerr=std_lat, capsize=6,
        color=colors[run_idx], alpha=0.8,
        label=f"{run['label']} ({mean_lat:.1f} ± {std_lat:.1f} ms)"
    )

ax.set_xticks(bar_positions)
ax.set_xticklabels([r["label"] for r in RUNS], rotation=15, ha="right")
ax.set_ylabel("Latency (ms / sample)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 8 — Summary table at a single chosen horizon
# =============================================================================
# %%

HORIZON_DISPLAY = 10   # change to the horizon step (0-indexed) you want to inspect

print(f"\n{'='*70}")
print(f"Summary at horizon step h={HORIZON_DISPLAY} (i.e. {HORIZON_DISPLAY+1} steps ahead)")
print(f"{'='*70}")
header = f"{'Model':<22} {'MAE':>8} {'CovErr':>8} {'CVaR75':>8} {'CVaR90':>8} {'CVaR95':>8} {'Conserv%':>9} {'Lat(ms)':>8}"
print(header)
print("-" * len(header))

for run, per_horizon in zip(RUNS, all_metrics):
    if per_horizon is None:
        print(f"{run['label']:<22}  [no data]")
        continue
    m  = per_horizon[HORIZON_DISPLAY]
    mae   = np.mean(m["model_mae"])           if m["model_mae"]           else float("nan")
    cov   = np.mean(m["model_coverage_mean"]) if m["model_coverage_mean"] else float("nan")
    cv75  = np.mean(m["model_cvar_0.75"])     if m["model_cvar_0.75"]     else float("nan")
    cv90  = np.mean(m["model_cvar_0.9"])      if m["model_cvar_0.9"]      else float("nan")
    cv95  = np.mean(m["model_cvar_0.95"])     if m["model_cvar_0.95"]     else float("nan")
    cons  = np.mean(m["model_is_more_conservative"]) * 100 if m["model_is_more_conservative"] else float("nan")
    lat   = np.mean(per_horizon[0]["latencies_ms"]) if per_horizon[0]["latencies_ms"] else float("nan")
    print(f"{run['label']:<22} {mae:>8.4f} {cov:>8.4f} {cv75:>8.4f} {cv90:>8.4f} {cv95:>8.4f} {cons:>8.1f}% {lat:>8.1f}")

# — Persistence row for reference —
print("-" * len(header))
# Use first valid run for persistence
for run, per_horizon in zip(RUNS, all_metrics):
    if per_horizon is None:
        continue
    m     = per_horizon[HORIZON_DISPLAY]
    mae_p = np.mean(m["persistence_mae"])           if m["persistence_mae"]           else float("nan")
    cov_p = np.mean(m["persistence_coverage_mean"]) if m["persistence_coverage_mean"] else float("nan")
    cv75p = np.mean(m["persistence_cvar_0.75"])     if m["persistence_cvar_0.75"]     else float("nan")
    cv90p = np.mean(m["persistence_cvar_0.9"])      if m["persistence_cvar_0.9"]      else float("nan")
    cv95p = np.mean(m["persistence_cvar_0.95"])     if m["persistence_cvar_0.95"]     else float("nan")
    print(f"{'Persistence (ref)':<22} {mae_p:>8.4f} {cov_p:>8.4f} {cv75p:>8.4f} {cv90p:>8.4f} {cv95p:>8.4f} {'N/A':>9} {'N/A':>8}")
    break

# =============================================================================
# SECTION 9 — Visual inspection: GT vs Model mean at chosen horizon
# =============================================================================
# %%

EPISODE_IDX  = 0   # which episode file to inspect
TIME_IDX     = 1   # which time step within that episode (i-th stride step)

HORIZON_VIS  = 10  # 0-indexed horizon to visualise

for run_idx, (run, episodes) in enumerate(zip(RUNS, all_run_episodes)):
    if episodes is None or EPISODE_IDX >= len(episodes):
        continue

    ep    = episodes[EPISODE_IDX]
    data  = ep["data"]
    pfx   = ep["sample_key_prefix"]
    time_steps = ep["time_steps"]

    if TIME_IDX >= len(time_steps):
        continue
    t = time_steps[TIME_IDX]

    sample_key = f"t_{t}_{pfx}_sample"
    mean_key   = f"t_{t}_{pfx}_mean"
    gt_key     = f"t_{t}_gt_horizon"

    if sample_key not in data or gt_key not in data:
        print(f"[{run['label']}] keys not found for t={t}"); continue

    gt_h      = data[gt_key].astype(np.float32)[HORIZON_VIS]
    samples_h = data[sample_key].astype(np.float32)[HORIZON_VIS]  # (N, H, W)
    mean_h    = samples_h.mean(axis=0)
    std_h     = samples_h.std(axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{run['label']} — ep {EPISODE_IDX}, t={t}, h={HORIZON_VIS+1}", fontsize=12)

    vmin, vmax = gt_h.min(), gt_h.max()

    im0 = axes[0].imshow(gt_h,    origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth"); plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mean_h,  origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Model Mean"); plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(std_h,   origin="lower", cmap="magma")
    axes[2].set_title("Model Std"); plt.colorbar(im2, ax=axes[2])

    err = np.abs(mean_h - gt_h)
    im3 = axes[3].imshow(err,     origin="lower", cmap="RdYlGn_r")
    axes[3].set_title(f"Abs Error (MAE={err.mean():.4f})"); plt.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.show()
