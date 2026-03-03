# %%
# =============================================================================
# BENCHMARK — Compare RNP rollout variants vs. Persistence baseline
# Run interactively cell-by-cell in VS Code / Jupyter
# =============================================================================
import numpy as np
import time
import matplotlib.pyplot as plt

# IEEE Paper Plot Formatting
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
})

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
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_rnp_bias",
    #     "label":  "RNP (bias)",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_rnp_no_bias",
    #     "label":  "RNP (no bias)",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_multistep_bias",
    #     "label":  "Multistep (bias)",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_multistep_no_bias",
    #     "label":  "Multistep (no bias)",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/run_gp",
    #     "label":  "GP (no bias)",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_bias",
    #     "label":  "FNO (bias)",
    # },
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
        "label":  "PFNO",
    },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d_decoupled_h5",
    #     "label":  "FNO-3D Decoupled Horizon 5",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d_decoupled_h15",
    #     "label":  "FNO-3D Decoupled Horizon 15",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d_last",
    #     "label":  "FNO-3D Last",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d_decoupled_last",
    #     "label":  "FNO-3D Decoupled Last",
    # },
    {
        "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/conv_lstm_last",
        "label":  "ConvLSTM",
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
    if "conv" in folder:
        return "conv_lstm_sample"
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

def c_tp(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.minimum(pred, gt).sum())

def c_fp(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.maximum(pred - gt, 0.0).sum())

def c_fn(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.maximum(gt - pred, 0.0).sum())

def soft_f_beta(pred: np.ndarray, gt: np.ndarray, beta: float = 2.0) -> float:
    ctp = c_tp(pred, gt)
    cfp = c_fp(pred, gt)
    cfn = c_fn(pred, gt)
    
    precision = ctp / (ctp + cfp + 1e-8)
    recall = ctp / (ctp + cfn + 1e-8)
    
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + 1e-8)
    return float(f_beta)

def soft_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.sum(pred * gt)
    union = np.sum(pred + gt - pred * gt)
    return float(intersection / (union + 1e-8))

def overbound_percentage_active(pred: np.ndarray, gt: np.ndarray, thresh=1e-3) -> float:
    active = gt > thresh
    if not np.any(active):
        return 100.0
    return float((pred[active] >= gt[active]).mean() * 100.0)

def overbound_percentage_domain(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred >= gt).mean() * 100.0)

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
    mu_tensor: (S, H, W)
    sigma_tensor: (S, H, W)
    """
    # 1. Calcular varianzas y precisiones
    varianzas = np.square(sigma_tensor)
    # Evitar división por cero sumando un epsilon muy pequeño si es necesario
    precisiones = 1.0 / (varianzas + 1e-8)
    
    # 2. Varianza combinada (Suma de precisiones invertida)
    # Colapsamos el eje 0 (los S modelos)
    precision_total = np.sum(precisiones, axis=0)
    var_final = 1.0 / precision_total
    
    # 3. Media combinada (Promedio ponderado por precisión)
    mu_final = np.sum(mu_tensor * precisiones, axis=0) * var_final
    
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
        "model_soft_f2":            [],
        "model_soft_iou":           [],
        "fused_mae":                [],
        "fused_coverage_mean":      [],
        "fused_conservatism_mean":  [],
        "fused_soft_f2":            [],
        "fused_soft_iou":           [],
        "persistence_mae":          [],
        "persistence_coverage_mean":[],
        "persistence_conservatism_mean":[],
        "persistence_soft_f2":      [],
        "persistence_soft_iou":     [],
        "model_is_more_conservative":[],
        "model_is_more_comprehensive":[],
        "model_overbound_active":   [],
        "model_overbound_domain":   [],
        "fused_overbound_active":   [],
        "fused_overbound_domain":   [],
        "persistence_overbound_active": [],
        "persistence_overbound_domain": [],
        "latencies_ms":             [],
    }
    for a in CVAR_LEVELS:
        result[f"model_mae_cvar_{a}"]       = []
        result[f"model_coverage_cvar_{a}"]  = []
        result[f"model_conservatism_cvar_{a}"]  = []
        result[f"fused_mae_cvar_{a}"]       = []
        result[f"fused_coverage_cvar_{a}"]  = []
        result[f"fused_conservatism_cvar_{a}"]  = []

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
            mu_raw   = data[sample_mean_key].astype(np.float32)[h] if has_dist else mean_h
            std_raw  = data[sample_std_key ].astype(np.float32)[h] if has_dist else samples_h.std(axis=0)

            if mu_raw.ndim == 3 and mu_raw.shape[0] > 1:
                mu_base  = mu_raw.mean(axis=0)
                # Pooled std dev
                if std_raw is not None:
                    std_base = np.sqrt((std_raw**2).mean(axis=0) + mu_raw.var(axis=0))
                else:
                    std_base = samples_h.std(axis=0)
                mu_fused, std_fused = fuse_gaussians(mu_raw, std_raw)
            else:
                mu_base  = mu_raw[0] if mu_raw.ndim == 3 else mu_raw
                std_base = std_raw[0] if std_raw is not None and std_raw.ndim == 3 else std_raw
                mu_fused, std_fused = mu_base, std_base

            # — Model metrics (Base) —
            mae_m = float(np.abs(mu_base - gt_h).mean())
            cov_m = float(coverage_error(mu_base, gt_h).mean())
            cons_m = float(conservatism_error(mu_base, gt_h).mean())
            result["model_mae"].append(mae_m)
            result["model_coverage_mean"].append(cov_m)
            result["model_conservatism_mean"].append(cons_m)
            result["model_soft_f2"].append(soft_f_beta(mu_base, gt_h, beta=2.0))
            result["model_soft_iou"].append(soft_iou(mu_base, gt_h))
            result["model_overbound_active"].append(overbound_percentage_active(mu_base, gt_h))
            result["model_overbound_domain"].append(overbound_percentage_domain(mu_base, gt_h))
            
            # — Fused Model metrics —
            mae_f = float(np.abs(mu_fused - gt_h).mean())
            cov_f = float(coverage_error(mu_fused, gt_h).mean())
            cons_f = float(conservatism_error(mu_fused, gt_h).mean())
            result["fused_mae"].append(mae_f)
            result["fused_coverage_mean"].append(cov_f)
            result["fused_conservatism_mean"].append(cons_f)
            result["fused_soft_f2"].append(soft_f_beta(mu_fused, gt_h, beta=2.0))
            result["fused_soft_iou"].append(soft_iou(mu_fused, gt_h))
            result["fused_overbound_active"].append(overbound_percentage_active(mu_fused, gt_h))
            result["fused_overbound_domain"].append(overbound_percentage_domain(mu_fused, gt_h))

            for a in CVAR_LEVELS:
                cvar_h = cvar(mu_base, std_base, a)
                result[f"model_mae_cvar_{a}"].append(float(np.abs(cvar_h - gt_h).mean()))
                result[f"model_coverage_cvar_{a}"].append(float(coverage_error(cvar_h, gt_h).mean()))
                result[f"model_conservatism_cvar_{a}"].append(float(conservatism_error(cvar_h, gt_h).mean()))
                
                cvar_f = cvar(mu_fused, std_fused, a)
                result[f"fused_mae_cvar_{a}"].append(float(np.abs(cvar_f - gt_h).mean()))
                result[f"fused_coverage_cvar_{a}"].append(float(coverage_error(cvar_f, gt_h).mean()))
                result[f"fused_conservatism_cvar_{a}"].append(float(conservatism_error(cvar_f, gt_h).mean()))

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
                    result["persistence_soft_f2"].append(soft_f_beta(gt_current, gt_h, beta=2.0))
                    result["persistence_soft_iou"].append(soft_iou(gt_current, gt_h))
                    result["persistence_overbound_active"].append(overbound_percentage_active(gt_current, gt_h))
                    result["persistence_overbound_domain"].append(overbound_percentage_domain(gt_current, gt_h))

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
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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

    fused_mae  = [np.mean(per_horizon[h]["fused_mae"])            if per_horizon[h]["fused_mae"]            else np.nan for h in range(MAX_HORIZON)]
    fused_cov  = [np.mean(per_horizon[h]["fused_coverage_mean"])  if per_horizon[h]["fused_coverage_mean"]  else np.nan for h in range(MAX_HORIZON)]
    fused_cons = [np.mean(per_horizon[h]["fused_conservatism_mean"]) if per_horizon[h]["fused_conservatism_mean"] else np.nan for h in range(MAX_HORIZON)]

    if not persistence_counted:
        mae_p  = [np.mean(per_horizon[h]["persistence_mae"])      if per_horizon[h]["persistence_mae"]      else np.nan for h in range(MAX_HORIZON)]
        cov_p  = [np.mean(per_horizon[h]["persistence_coverage_mean"]) if per_horizon[h]["persistence_coverage_mean"] else np.nan for h in range(MAX_HORIZON)]
        cons_p = [np.mean(per_horizon[h]["persistence_conservatism_mean"]) if per_horizon[h]["persistence_conservatism_mean"] else np.nan for h in range(MAX_HORIZON)]
        ax_mae.plot(horizons, mae_p, color=c, ls=ls_pers,  marker="s", ms=4, label=f"Persistence MAE", alpha=0.6)
        ax_cov.plot(horizons, cov_p, color=c, ls=ls_pers,  marker="s", ms=4, label=f"Persistence Coverage", alpha=0.6)
        ax_cons.plot(horizons, cons_p, color=c, ls=ls_pers,  marker="s", ms=4, label=f"Persistence Conservatism", alpha=0.6)
        persistence_counted = True

    ax_mae.plot(horizons, mae_m, color=c, ls=ls_model, marker="o", ms=4, label=f"{lbl} — base")
    ax_cov.plot(horizons, cov_m, color=c, ls=ls_model, marker="o", ms=4, label=f"{lbl} — base")
    ax_cons.plot(horizons, cons_m, color=c, ls=ls_model, marker="o", ms=4, label=f"{lbl} — base")

    ax_mae.plot(horizons, fused_mae, color=c, ls=":", marker="x", ms=4, label=f"{lbl} — fused")
    ax_cov.plot(horizons, fused_cov, color=c, ls=":", marker="x", ms=4, label=f"{lbl} — fused")
    ax_cons.plot(horizons, fused_cons, color=c, ls=":", marker="x", ms=4, label=f"{lbl} — fused")

ax_mae.set_xlabel("Horizon step"); ax_mae.set_ylabel("MAE")
ax_mae.set_title("Mean Absolute Error"); ax_mae.legend(fontsize=7); ax_mae.grid(alpha=0.3)

ax_cov.set_xlabel("Horizon step"); ax_cov.set_ylabel("Coverage error (mean)")
ax_cov.set_title("Coverage (under-prediction penalty)"); ax_cov.legend(fontsize=7); ax_cov.grid(alpha=0.3)

ax_cons.set_xlabel("Horizon step"); ax_cons.set_ylabel("Conservatism error (mean)")
ax_cons.set_title("Conservatism (over-prediction penalty)"); ax_cons.legend(fontsize=7); ax_cons.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 5 — CVaR comparison across all horizons
#   For path planning: a model is SAFER if its CVaR > ground truth CVaR
#   (i.e. it over-predicts risk in the tail → conservative for drone nav)
# =============================================================================
# %%

metrics = ["mae", "coverage", "conservatism"]
fig, axes = plt.subplots(len(CVAR_LEVELS), len(metrics), figsize=(5 * len(metrics), 5 * len(CVAR_LEVELS)), sharey=False)
fig.suptitle("CVaR (Expected Shortfall) by α — higher = heavier tail risk estimated", fontsize=13)

for axi, a in enumerate(CVAR_LEVELS):
    for j, metric in enumerate(metrics):
        ax = axes[axi, j]
        persistence_counted = False
        for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
            if per_horizon is None:
                continue
            c   = colors[run_idx]
            lbl = run["label"]

            m = [np.mean(per_horizon[h][f"model_{metric}_cvar_{a}"])       if per_horizon[h][f"model_{metric}_cvar_{a}"]       else np.nan for h in range(MAX_HORIZON)]
            f_m = [np.mean(per_horizon[h][f"fused_{metric}_cvar_{a}"])       if per_horizon[h][f"fused_{metric}_cvar_{a}"]       else np.nan for h in range(MAX_HORIZON)]
            
            if not persistence_counted:
                new_metric = metric
                if metric != "mae":
                    new_metric += "_mean"
                p = [np.mean(per_horizon[h][f"persistence_{new_metric}"]) if per_horizon[h][f"persistence_{new_metric}"] else np.nan for h in range(MAX_HORIZON)]
                ax.plot(horizons, p, color=c, ls="--", marker="s", ms=4, label=f"Persistence — {metric}", alpha=0.6)
                persistence_counted = True

            ax.plot(horizons, m, color=c, ls="-",  marker="o", ms=4, label=f"{lbl} — base")
            ax.plot(horizons, f_m, color=c, ls=":", marker="x", ms=4, label=f"{lbl} — fused")

            ax.set_title(f"CVaR {metric} α={a}")
            ax.set_xlabel("Horizon step")
            ax.set_ylabel(metric)
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
# SECTION 6.1 — Comprehensive rate (% of steps where model coverage > persistence)
#   For drone path planning: higher = model is more conservative / safer.
# =============================================================================
# %%

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Comprehensive Rate vs Horizon\n(% timesteps where model coverage > persistence coverage)")

for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c   = colors[run_idx]
    lbl = run["label"]

    rates = []
    for h in range(MAX_HORIZON):
        vals = per_horizon[h]["model_is_more_comprehensive"]
        rates.append(np.mean(vals) * 100.0 if vals else np.nan)

    ax.plot(horizons, rates, color=c, ls="-", marker="o", ms=4, label=lbl)

ax.axhline(50, color="gray", ls=":", lw=1, label="50% (neutral)")
ax.set_xlabel("Horizon step")
ax.set_ylabel("% more comprehensive than persistence")
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

HORIZON_DISPLAY = 10   # 0-indexed horizon step to inspect

def _g(lst, default=float("nan")):
    """Safe mean of a list, returns default if empty."""
    return float(np.mean(lst)) if lst else default

print(f"\n{'='*90}")
print(f"Summary at horizon step h={HORIZON_DISPLAY}  ({HORIZON_DISPLAY+1} steps ahead)")
print(f"{'='*90}")
header = (f"{'Model':<22} {'MAE':>7} {'Cov':>7} {'Cons':>7} "
          f"{'MAE@75':>7} {'MAE@90':>7} {'MAE@95':>7} "
          f"{'Cov@75':>7} {'Cons@75':>7} "
          f"{'Consrv%':>8} {'Compr%':>7} {'Lat(ms)':>8}")
print(header)
print("-" * len(header))

for run, per_horizon in zip(RUNS, all_metrics):
    if per_horizon is None:
        print(f"{run['label']:<22}  [no data]")
        continue
    m   = per_horizon[HORIZON_DISPLAY]
    mae   = _g(m["model_mae"])
    cov   = _g(m["model_coverage_mean"])
    cons  = _g(m["model_conservatism_mean"])
    cm75  = _g(m["model_mae_cvar_0.75"])
    cm90  = _g(m["model_mae_cvar_0.9"])
    cm95  = _g(m["model_mae_cvar_0.95"])
    cv75  = _g(m["model_coverage_cvar_0.75"])
    cs75  = _g(m["model_conservatism_cvar_0.75"])
    pct_c = _g(m["model_is_more_conservative"]) * 100
    pct_r = _g(m["model_is_more_comprehensive"]) * 100
    lat   = _g(per_horizon[0]["latencies_ms"])
    
    lbl = run['label'] + " (Base)"
    print(f"{lbl:<22} {mae:>7.4f} {cov:>7.4f} {cons:>7.4f} "
          f"{cm75:>7.4f} {cm90:>7.4f} {cm95:>7.4f} "
          f"{cv75:>7.4f} {cs75:>7.4f} "
          f"{pct_c:>7.1f}% {pct_r:>6.1f}% {lat:>8.1f}")
          
    f_mae   = _g(m["fused_mae"])
    f_cov   = _g(m["fused_coverage_mean"])
    f_cons  = _g(m["fused_conservatism_mean"])
    f_cm75  = _g(m["fused_mae_cvar_0.75"])
    f_cm90  = _g(m["fused_mae_cvar_0.9"])
    f_cm95  = _g(m["fused_mae_cvar_0.95"])
    f_cv75  = _g(m["fused_coverage_cvar_0.75"])
    f_cs75  = _g(m["fused_conservatism_cvar_0.75"])
    
    lbl_fused = run['label'] + " (Fused)"
    print(f"{lbl_fused:<22} {f_mae:>7.4f} {f_cov:>7.4f} {f_cons:>7.4f} "
          f"{f_cm75:>7.4f} {f_cm90:>7.4f} {f_cm95:>7.4f} "
          f"{f_cv75:>7.4f} {f_cs75:>7.4f} "
          f"{'-':>8} {'-':>7} {'-':>8}")

# — Persistence row —
print("-" * len(header))
for run, per_horizon in zip(RUNS, all_metrics):
    if per_horizon is None:
        continue
    m     = per_horizon[HORIZON_DISPLAY]
    mae_p = _g(m["persistence_mae"])
    cov_p = _g(m["persistence_coverage_mean"])
    con_p = _g(m["persistence_conservatism_mean"])
    print(f"{'Persistence (ref)':<22} {mae_p:>7.4f} {cov_p:>7.4f} {con_p:>7.4f} "
          f"{'N/A':>7} {'N/A':>7} {'N/A':>7} "
          f"{'N/A':>7} {'N/A':>7} "
          f"{'N/A':>8} {'N/A':>7} {'N/A':>8}")
    break

# =============================================================================
# SECTION 9 — Visual inspection PER RUN: GT, μ, σ, Coverage, Conservatism, CVaR
#   Persistence is shown first (same for all runs).
# =============================================================================
# %%

EPISODE_IDX = 0   # which episode to inspect
TIME_IDX    = 3   # i-th stride step within the episode (must be >= 1 for persistence)
HORIZON_VIS = 9  # 0-indexed forecast horizon to visualise

def _im(ax, img, title, **kw):
    im = ax.imshow(img, origin="lower", **kw)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def _plot_vis(label, gt_h, pred_h, sigma_h, t, mu_lbl, cvar_label="CVaR-75"):
    """Plot the standard 2×4 inspection grid for one predicted frame."""
    mae_map    = np.abs(pred_h - gt_h)
    cov_map    = coverage_error(pred_h, gt_h)
    cons_map   = conservatism_error(pred_h, gt_h)
    cvar75     = cvar(pred_h, sigma_h, 0.75)
    cvar_cov75 = coverage_error(cvar75, gt_h)
    cvar_con75 = conservatism_error(cvar75, gt_h)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"{label} — ep {EPISODE_IDX}, t={t}, h={HORIZON_VIS+1}", fontsize=13)

    _im(axes[0,0], gt_h,    "Ground Truth",                    cmap="inferno", vmin=0, vmax=1)
    _im(axes[0,1], pred_h,  mu_lbl,                            cmap="inferno", vmin=0, vmax=1)
    _im(axes[0,2], sigma_h, "σ",                               cmap="magma",   vmin=0)
    _im(axes[0,3], mae_map, f"MAE  ({mae_map.mean():.4f})",    cmap="RdYlGn_r",vmin=0, vmax=0.5)

    _im(axes[1,0], cov_map,    f"Coverage (under-pred)  {cov_map.mean():.4f}",    cmap="Reds",  vmin=0, vmax=0.5)
    _im(axes[1,1], cons_map,   f"Conservatism (over-pred) {cons_map.mean():.4f}", cmap="Blues", vmin=0, vmax=0.5)
    _im(axes[1,2], cvar_cov75, f"{cvar_label} Coverage  {cvar_cov75.mean():.4f}", cmap="Reds",  vmin=0, vmax=0.5)
    _im(axes[1,3], cvar_con75, f"{cvar_label} Conservatism {cvar_con75.mean():.4f}", cmap="Blues", vmin=0, vmax=0.5)

    plt.tight_layout()
    plt.show()


# ---- Persistence (compute once from first valid run) -----------------------
_pers_plotted = False
for run_idx, (run, episodes) in enumerate(zip(RUNS, all_run_episodes)):
    if episodes is None or EPISODE_IDX >= len(episodes):
        continue
    ep         = episodes[EPISODE_IDX]
    data       = ep["data"]
    time_steps = ep["time_steps"]

    if TIME_IDX >= len(time_steps) or TIME_IDX < 1:
        break
    t      = time_steps[TIME_IDX]
    t_prev = time_steps[TIME_IDX - 1]
    stride = t - t_prev
    gt_key      = f"t_{t}_gt_horizon"
    gt_prev_key = f"t_{t_prev}_gt_horizon"

    if gt_key not in data or gt_prev_key not in data:
        break
    gt_h        = data[gt_key].astype(np.float32)[HORIZON_VIS]   # (H, W)
    # Frame at time t = data[t_prev_gt_horizon][stride - 1]
    # because t_prev_gt_horizon stores frames at [t_prev+1 .. t_prev+horizon]
    # → index stride-1 = (t - t_prev) - 1 = frame at t
    pers_frame  = data[gt_prev_key].astype(np.float32)[stride - 1]  # (H, W)

    _plot_vis("Persistence (baseline)", gt_h, pers_frame,
              np.zeros_like(pers_frame),   # persistence has no uncertainty
              t, "Persistence frame", cvar_label="CVaR-75 (σ=0)")
    _pers_plotted = True
    break   # same for all runs → plot once

if not _pers_plotted:
    print("[Persistence] TIME_IDX must be >= 1 and prev GT key must exist.")


# ---- Model runs -------------------------------------------------------------
for run_idx, (run, episodes) in enumerate(zip(RUNS, all_run_episodes)):
    if episodes is None or EPISODE_IDX >= len(episodes):
        continue

    ep         = episodes[EPISODE_IDX]
    data       = ep["data"]
    pfx        = ep["sample_key_prefix"]
    time_steps = ep["time_steps"]

    if TIME_IDX >= len(time_steps):
        continue
    t = time_steps[TIME_IDX]

    sample_key = f"t_{t}_{pfx}_sample"
    mean_key   = f"t_{t}_{pfx}_mean"
    std_key    = f"t_{t}_{pfx}_std"
    gt_key     = f"t_{t}_gt_horizon"

    if sample_key not in data or gt_key not in data:
        print(f"[{run['label']}] keys not found for t={t}")
        continue

    gt_h      = data[gt_key].astype(np.float32)[HORIZON_VIS]      # (H, W)
    samples_h = data[sample_key].astype(np.float32)[HORIZON_VIS]  # (N, H, W)
    mean_h    = samples_h.mean(axis=0)
    std_h     = samples_h.std(axis=0)

    has_dist = mean_key in data and std_key in data
    mu_raw   = data[mean_key].astype(np.float32)[HORIZON_VIS] if has_dist else mean_h
    std_raw  = data[std_key ].astype(np.float32)[HORIZON_VIS] if has_dist else std_h
    
    if mu_raw.ndim == 3 and mu_raw.shape[0] > 1:
        mu_base = mu_raw.mean(axis=0)
        if std_raw is not None:
             std_base = np.sqrt((std_raw**2).mean(axis=0) + mu_raw.var(axis=0))
        else:
             std_base = std_h
        mu_fused, std_fused = fuse_gaussians(mu_raw, std_raw)
        
        _plot_vis(run["label"] + " (Base)", gt_h, mu_base, std_base, t, "Model μ (dist)")
        _plot_vis(run["label"] + " (Fused)", gt_h, mu_fused, std_fused, t, "Model μ (fused)")
    else:
        mu_base = mu_raw[0] if mu_raw.ndim == 3 else mu_raw
        std_base = std_raw[0] if std_raw is not None and std_raw.ndim == 3 else std_raw
        mu_lbl   = "Model μ (dist)" if has_dist else "Model μ (samples)"
        _plot_vis(run["label"], gt_h, mu_base, std_base, t, mu_lbl)

# =============================================================================
# SECTION 10 — Continuous F2 / Soft IoU Plots
# =============================================================================
# %%

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Spatial Forecasting Quality (Higher is better)", fontsize=14)

ax_f2, ax_iou = axes

persistence_counted = False
for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c = colors[run_idx]
    lbl = run["label"]

    f2_m   = [np.mean(per_horizon[h]["model_soft_f2"]) if per_horizon[h]["model_soft_f2"] else np.nan for h in range(MAX_HORIZON)]
    iou_m  = [np.mean(per_horizon[h]["model_soft_iou"]) if per_horizon[h]["model_soft_iou"] else np.nan for h in range(MAX_HORIZON)]

    f2_f   = [np.mean(per_horizon[h]["fused_soft_f2"]) if per_horizon[h]["fused_soft_f2"] else np.nan for h in range(MAX_HORIZON)]
    iou_f  = [np.mean(per_horizon[h]["fused_soft_iou"]) if per_horizon[h]["fused_soft_iou"] else np.nan for h in range(MAX_HORIZON)]

    if not persistence_counted:
        f2_p  = [np.mean(per_horizon[h]["persistence_soft_f2"]) if per_horizon[h]["persistence_soft_f2"] else np.nan for h in range(MAX_HORIZON)]
        iou_p = [np.mean(per_horizon[h]["persistence_soft_iou"]) if per_horizon[h]["persistence_soft_iou"] else np.nan for h in range(MAX_HORIZON)]
        
        ax_f2.plot(horizons, f2_p, color=c, ls="--", marker="s", ms=4, label="Persistence", alpha=0.6)
        ax_iou.plot(horizons, iou_p, color=c, ls="--", marker="s", ms=4, label="Persistence", alpha=0.6)
        persistence_counted = True

    ax_f2.plot(horizons, f2_m, color=c, ls="-", marker="o", ms=4, label=f"{lbl} — base")
    ax_iou.plot(horizons, iou_m, color=c, ls="-", marker="o", ms=4, label=f"{lbl} — base")

    ax_f2.plot(horizons, f2_f, color=c, ls=":", marker="x", ms=4, label=f"{lbl} — fused")
    ax_iou.plot(horizons, iou_f, color=c, ls=":", marker="x", ms=4, label=f"{lbl} — fused")

ax_f2.set_xlabel("Horizon step")
ax_f2.set_ylabel("Soft F2 Score")
ax_f2.set_title("Continuous F2 (Penalizes Underprediction)")
ax_f2.legend(fontsize=7)
ax_f2.grid(alpha=0.3)
ax_f2.set_ylim(0, 1.05)

ax_iou.set_xlabel("Horizon step")
ax_iou.set_ylabel("Soft IoU")
ax_iou.set_title("Soft IoU (Spatial Overlap)")
ax_iou.legend(fontsize=7)
ax_iou.grid(alpha=0.3)
ax_iou.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 11 — IEEE Paper Final Plots and Texts
# =============================================================================
# %%

print(f"\n{'='*90}")
print("IEEE PAPER EXTRACTED VALUES")
print(f"{'='*90}")

horizon_time_s = MAX_HORIZON * 0.1
print(f"Total Horizon Time: {horizon_time_s:.1f} s (N={MAX_HORIZON} steps)")

# Find PFNO latency and overbound
pfno_run_idx = -1
for i, r in enumerate(RUNS):
    if r["label"] == "PFNO":
        pfno_run_idx = i

if pfno_run_idx != -1 and all_metrics[pfno_run_idx] is not None:
    lats = all_metrics[pfno_run_idx][0]["latencies_ms"]
    pfno_lat_ms = np.mean(lats) if lats else 0.0
    pfno_total_lat = pfno_lat_ms * MAX_HORIZON
    pfno_overbound_active = np.mean([np.mean(all_metrics[pfno_run_idx][h]["model_overbound_active"]) for h in range(MAX_HORIZON)])
    pfno_overbound_domain = np.mean([np.mean(all_metrics[pfno_run_idx][h]["model_overbound_domain"]) for h in range(MAX_HORIZON)])

    print(f"PFNO inference rate per rollout step: {pfno_lat_ms:.2f} ms")
    print(f"PFNO yielding a total {horizon_time_s:.1f} s horizon computation time of roughly {pfno_total_lat:.2f} ms")
    print(f"PFNO successfully overbounds the active smoke front in nearly {pfno_overbound_active:.2f}% of configurations")
    print(f"(Domain overbound: {pfno_overbound_domain:.2f}%)\n")

# -- IEEE Paper Plots --
# Plot 1: Overbound percentage / Coverage (strictly upper bounds)
# Plot 2: MAE over horizon

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.set_xlabel("Forecast Horizon (s)")
ax1.set_ylabel("Active Front Overbound (%)")    

horizon_times = [h * 0.1 for h in horizons]
persistence_counted = False

for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c = colors[run_idx]
    lbl = run["label"]

    ob_m = [np.mean(per_horizon[h]["model_overbound_active"]) if per_horizon[h]["model_overbound_active"] else np.nan for h in range(MAX_HORIZON)]
    
    if not persistence_counted:
        ob_p = [np.mean(per_horizon[h]["persistence_overbound_active"]) if per_horizon[h]["persistence_overbound_active"] else np.nan for h in range(MAX_HORIZON)]
        ax1.plot(horizon_times, ob_p, color="gray", ls="--", marker="s", ms=4, label="Persistence", zorder=2)
        persistence_counted = True

    ax1.plot(horizon_times, ob_m, color=c, ls="-", marker="o", ms=4, label=lbl, zorder=3)

ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_title("Safety Boundary Overestimation")
plt.tight_layout()
plt.savefig("ieee_coverage_plot.png")
plt.show()

# MAE Comparison plot
fig, ax2 = plt.subplots(figsize=(6, 4))
ax2.set_xlabel("Forecast Horizon (s)")
ax2.set_ylabel("Mean Absolute Error")

persistence_counted = False
for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c = colors[run_idx]
    lbl = run["label"]

    mae_m = [np.mean(per_horizon[h]["model_mae"]) if per_horizon[h]["model_mae"] else np.nan for h in range(MAX_HORIZON)]
    
    if not persistence_counted:
        mae_p = [np.mean(per_horizon[h]["persistence_mae"]) if per_horizon[h]["persistence_mae"] else np.nan for h in range(MAX_HORIZON)]
        ax2.plot(horizon_times, mae_p, color="gray", ls="--", marker="s", ms=4, label="Persistence", zorder=2)
        persistence_counted = True

    ax2.plot(horizon_times, mae_m, color=c, ls="-", marker="o", ms=4, label=lbl, zorder=3)

ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_title("Spatiotemporal Accuracy (MAE)")
plt.tight_layout()
plt.savefig("ieee_mae_plot.png")
plt.show()

# %%
