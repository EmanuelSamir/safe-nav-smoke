
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
    "mathtext.fontset": "stix",
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

RUNS = [
    {
        "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d",
        "label":  "PFNO",
    },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d_nll_1e3",
    #     "label":  "PFNO",
    # },
    # {
    #     "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/fno_3d_beta_nll",
    #     "label":  "PFNO",
    # },
    {
        "folder": "/home/emunoz/dev/safe-nav-smoke/saved_rollouts/conv_lstm_last",
        "label":  "ConvLSTM",
    },
]

MAX_HORIZON = 20      # max horizon stored in the .npz files (1-indexed count)
CVAR_LEVELS  = [0.5, 0.75, 0.90, 0.95]   # CVaR α levels to evaluate

# =============================================================================
# SECTION 1 — Helper utilities
# =============================================================================
# %%


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
    Returns just paths, data is lazy loaded in metric collection to save memory.
    """
    folder = run["folder"]
    import re
    def try_extract_ep_idx(fname):
        match = re.search(r'ep_(\d+)', fname)
        return int(match.group(1)) if match else fname
        
    files  = sorted([f for f in os.listdir(folder) if f.endswith(".npz")], key=try_extract_ep_idx)
    episodes = [{"path": os.path.join(folder, fname)} for fname in files]
    return episodes

def fuse_gaussians(mu_tensor, sigma_tensor):
    """
    mu_tensor: (S, H, W)
    sigma_tensor: (S, H, W)
    """
    # 1. Media combinada: El promedio simple de las trayectorias
    # E[mu]
    mu_final = np.mean(mu_tensor, axis=0)
    
    # 2. Varianza combinada (Ley de Varianza Total)
    # Var_total = E[sigma^2] + Var(mu)
    
    # Incertidumbre interna promedio (Aleatoriedad del proceso)
    varianzas_internas = np.square(sigma_tensor)
    esperanza_varianzas = np.mean(varianzas_internas, axis=0)
    
    # Incertidumbre por discrepancia (Qué tanto se separan los rollouts)
    varianza_de_medias = np.var(mu_tensor, axis=0)
    
    var_final = esperanza_varianzas + varianza_de_medias
    
    return mu_final, np.sqrt(var_final)


def collect_metrics_for_all_horizons(episodes: list[dict], max_horizon: int) -> dict:
    """
    Aggregate per-episode scalar metrics for ALL horizons at once.
    This avoids redundant lazy loading from disk.
    
    Returns dict mapping horizon index h -> dict mapping variant -> metric -> list of values.
    Variants include "base", "fused", "persistence", and CVaR variants like "base_cvar_0.75".
    """
    results_per_h = {}
    
    # We will define standard metrics to evaluate for each variant
    def evaluate_metrics(pred, gt):
        return {
            "mae": float(np.abs(pred - gt).mean()),
            "coverage": float(coverage_error(pred, gt).mean()),
            "conservatism": float(conservatism_error(pred, gt).mean()),
            "soft_f2": soft_f_beta(pred, gt, beta=2.0),
            "soft_iou": soft_iou(pred, gt),
            "overbound_active": overbound_percentage_active(pred, gt),
            "overbound_domain": overbound_percentage_domain(pred, gt),
        }

    for h in range(max_horizon):
        results_per_h[h] = {"latencies_ms": []}

    for ep in tqdm.tqdm(episodes, desc="Processing episodes"):
        data       = np.load(ep["path"])
        if "sample_mean" not in data or "sample_std" not in data:
            continue
        time_steps = data["time_steps"]

        for i, t in enumerate(time_steps):
            # 1. Ground Truth
            gt_all = data["gt_full"][t + 1 : t + 1 + max_horizon].astype(np.float32)
            gt_current = data["gt_full"][t].astype(np.float32)

            max_h_avail = gt_all.shape[0]
            if max_h_avail == 0: continue

            # 2. Extract model predictions
            rollout = data["sample"][i].astype(np.float32)
            has_dist = "mean" in data and "std" in data
            mu_raw_all  = data["mean"][i].astype(np.float32) if has_dist else None
            std_raw_all = data["std"][i].astype(np.float32) if has_dist else None
            
            has_sample_dist = "sample_mean" in data and "sample_std" in data
            mu_sample_all  = data["sample_mean"][i].astype(np.float32) if has_sample_dist else None
            std_sample_all = data["sample_std"][i].astype(np.float32) if has_sample_dist else None

            latency_val = float(data["latency"][i]) if "latency" in data else None

            for h in range(min(max_horizon, max_h_avail)):
                result = results_per_h[h]
                
                gt_h      = gt_all[h]                              # (H, W)
                samples_h = rollout[h]                             # (N, H, W)
                mean_h    = samples_h.mean(axis=0)                 # (H, W) ensemble mean

                # Distribution parameters at horizon h (model μ, σ)
                mu_raw   = mu_raw_all[h] if mu_raw_all is not None else mean_h
                std_raw  = std_raw_all[h] if std_raw_all is not None else samples_h.std(axis=0)
                
                mu_sample = mu_sample_all[h] if mu_sample_all is not None else None
                std_sample = std_sample_all[h] if std_sample_all is not None else None

                mu_base  = mu_raw[0] if mu_raw.ndim == 3 else mu_raw
                std_base = std_raw[0] if std_raw is not None and std_raw.ndim == 3 else std_raw
                
                mu_fused, std_fused = fuse_gaussians(mu_sample, std_sample)

                # Prepare variants
                variants_dict = {
                    "base": mu_base,
                    "fused": mu_fused,
                }
                if gt_current is not None:
                    variants_dict["persistence"] = gt_current
                    
                for a in CVAR_LEVELS:
                    variants_dict[f"base_cvar_{a}"] = cvar(mu_base, std_base, a)
                    variants_dict[f"fused_cvar_{a}"] = cvar(mu_fused, std_fused, a)

                # Evaluate metrics for all variants
                for var_name, var_pred in variants_dict.items():
                    if var_name not in result:
                        result[var_name] = {}
                    metrics_vals = evaluate_metrics(var_pred, gt_h)
                    for m_name, m_val in metrics_vals.items():
                        if m_name not in result[var_name]:
                            result[var_name][m_name] = []
                        result[var_name][m_name].append(m_val)

                # — Latency —
                if latency_val is not None:
                    result["latencies_ms"].append(latency_val)

    return results_per_h


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

    print(f"Metrics [{run['label']}]")
    per_horizon = collect_metrics_for_all_horizons(episodes, MAX_HORIZON)
    all_metrics.append(per_horizon)

print("Metric collection complete.")

# %%
horizons = np.array(list(range(1, MAX_HORIZON + 1))) / 10.0
import matplotlib.cm as cm
_cmap = cm.get_cmap('Set2', len(RUNS))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]#[_cmap(i) for i in range(len(RUNS))]

# Overbound comparison in percentage - Need to compare between active and not active. MSE error it is another plot
fig, axes = plt.subplots(1, 2, figsize=(12, ))

CVAR_TEST = [0.5, 0.9]

for metric_idx, (metric_key, metric_opts) in enumerate(zip(["coverage", "conservatism"], [{"title": "Under-prediction error", "y_label": "Error ($\max(y_{\mathrm{true}} - y_{\mathrm{pred}}, 0)$)"}, {"title": "Over-prediction error", "y_label": "Error ($\max(y_{\mathrm{pred}} - y_{\mathrm{true}}, 0)$)"}]),):
    a = axes[metric_idx]
    for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
        c = colors[run_idx]
        run_lbl = run["label"]

        if run_idx == 0:
            var_name = "persistence"
            vals = []
            for h in range(MAX_HORIZON):
                if var_name in per_horizon[h] and metric_key in per_horizon[h][var_name]:
                    data_list = per_horizon[h][var_name][metric_key]
                    vals.append(np.mean(data_list) if data_list else np.nan)
                else:
                    vals.append(np.nan)
        
            label = "Persistence"
            
            color = "gray"
            a.plot(horizons, vals, color=color, ls="--", 
                    marker="s", ms=4, alpha=0.7,
                    label=label)

        var_name = "base"
        vals = []
        for h in range(MAX_HORIZON):
            if var_name in per_horizon[h] and metric_key in per_horizon[h][var_name]:
                data_list = per_horizon[h][var_name][metric_key]
                vals.append(np.mean(data_list) if data_list else np.nan)
            else:
                vals.append(np.nan)

        label = run_lbl
        
        color = c
        a.plot(horizons, vals, color=color, ls="-", 
                marker="o", ms=4, alpha=1.0,
                label=label)

        if run_idx == 0:
            for cvar_alpha in CVAR_TEST:
                var_name = f"base_cvar_{cvar_alpha}"
                vals = []
                for h in range(MAX_HORIZON):
                    if var_name in per_horizon[h] and metric_key in per_horizon[h][var_name]:
                        data_list = per_horizon[h][var_name][metric_key]
                        vals.append(np.mean(data_list) if data_list else np.nan)
                    else:
                        vals.append(np.nan)

                label = run_lbl
                
                color = c
                a.plot(horizons, vals, color=color, ls="--", 
                        marker="x", ms=4, alpha=cvar_alpha,
                        label=f"{label} (CVaR α={cvar_alpha})")

        a.set_xlabel("Forecasting Horizon (s)", fontsize=13)
        a.set_ylabel(metric_opts["y_label"], fontsize=13)
        a.set_title(metric_opts["title"], fontsize=13)
        a.legend(fontsize=7)
        a.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 3.1 — Appendix Grid Plot: PFNO, PFNO CVaR, ConvLSTM
# =============================================================================
# %%
EPISODE_IDX_APP = 4
TIME_IDX_APP    = 2
HORIZON_VIS_APP = 14
CVAR_ALPHA_APP  = 0.9

appendix_data = []

for run_idx, (run, episodes) in enumerate(zip(RUNS, all_run_episodes)):
    if episodes is None or EPISODE_IDX_APP >= len(episodes):
        continue
    
    ep = episodes[EPISODE_IDX_APP]
    data = np.load(ep["path"])
    
    if "sample_mean" not in data or "sample_std" not in data:
        print(f"[{run['label']}] Skipping deprecated rollout")
        continue

    time_steps = data["time_steps"]
    if TIME_IDX_APP >= len(time_steps):
        continue
        
    t = time_steps[TIME_IDX_APP]
    gt_h = data["gt_full"][t + 1 + HORIZON_VIS_APP].astype(np.float32)
    samples_h = data["sample"][TIME_IDX_APP, HORIZON_VIS_APP].astype(np.float32)
    has_dist = "mean" in data and "std" in data
    
    mu_raw = data["mean"][TIME_IDX_APP, HORIZON_VIS_APP].astype(np.float32) if has_dist else None
    std_raw = data["std"][TIME_IDX_APP, HORIZON_VIS_APP].astype(np.float32) if has_dist else None
    
    mean_h = samples_h.mean(axis=0)
    std_h = samples_h.std(axis=0)
    if mu_raw is None: mu_raw = mean_h
    if std_raw is None: std_raw = std_h
    
    mu_base = mu_raw[0] if mu_raw.ndim == 3 else mu_raw
    std_base = std_raw[0] if std_raw is not None and std_raw.ndim == 3 else std_raw
    
    if run["label"] == "PFNO":
        appendix_data.append({
            "label": "PFNO $\\mu$",
            "gt": gt_h,
            "pred": mu_base
        })
        cvar_pred = cvar(mu_base, std_base, CVAR_ALPHA_APP)
        appendix_data.append({
            "label": f"PFNO CVaR\n($\\alpha={CVAR_ALPHA_APP}$)",
            "gt": gt_h,
            "pred": cvar_pred
        })
    elif run["label"] == "ConvLSTM":
        appendix_data.append({
            "label": "ConvLSTM",
            "gt": gt_h,
            "pred": mu_base
        })

if len(appendix_data) == 3:
    # Aumentamos un poco el tamaño de la figura para acomodar las barras de color horizontales abajo y el título
    fig, axes = plt.subplots(3, 5, figsize=(11, 5.5), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    
    # Añadimos un título principal (suptitle) indicando qué rollout, paso y horizonte estamos viendo
    fig.suptitle(f"Example — Episode {EPISODE_IDX_APP}, Time Step {t}, Forecast Horizon {HORIZON_VIS_APP+1}", 
                 fontsize=14, y=0.98)
    
    col_titles = ["GT", "Prediction", "Under-prediction", "Over-prediction", "MAE"]
    ims = []
    
    for row_idx, row_data in enumerate(appendix_data):
        gt = row_data["gt"]
        pred = row_data["pred"]
        label = row_data["label"]
        
        cov_map = coverage_error(pred, gt)
        cons_map = conservatism_error(pred, gt)
        err_map = np.abs(pred - gt)
        
        # GT
        ax_gt = axes[row_idx, 0]
        im_gt = ax_gt.imshow(gt, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax_gt.set_ylabel(label, fontsize=12)
        
        # Pred
        ax_pred = axes[row_idx, 1]
        im_pred = ax_pred.imshow(pred, origin="lower", cmap="inferno", vmin=0, vmax=1)
        
        # Coverage
        ax_cov = axes[row_idx, 2]
        im_cov = ax_cov.imshow(cov_map, origin="lower", cmap="Reds", vmin=0, vmax=0.5)
        
        # Conservatism
        ax_cons = axes[row_idx, 3]
        im_cons = ax_cons.imshow(cons_map, origin="lower", cmap="Blues", vmin=0, vmax=0.5)
        
        # Error
        ax_err = axes[row_idx, 4]
        im_err = ax_err.imshow(err_map, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=0.5)
        
        for col_idx in range(5):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=12)
                
        # Guardamos la referencia de imagen de la última fila para crear los colorbars
        if row_idx == 2:
            ims = [im_gt, im_pred, im_cov, im_cons, im_err]

    # Añadimos una barra de color en la parte inferior de cada columna
    for col_idx, im in enumerate(ims):
        # ax=axes[:, col_idx] hace que robe un poco de espacio equitativamente a toda la columna
        cbar = fig.colorbar(im, ax=axes[:, col_idx], orientation='horizontal', 
                            fraction=0.04, pad=0.08, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.locator_params(nbins=3) # Limitamos a pocos ticks (ej: 0.0, 0.2, 0.5)
                
    # Puede que tight_layout tire un warning con ax=... pero funciona bien
    plt.tight_layout(pad=0.2)
    plt.show()

# %%


# =============================================================================
# GENERAL PLOTTING UTILITY
# =============================================================================
# %%

def plot_horizon_metrics(all_metrics, runs, active_variants, metrics_to_plot, title="Metrics vs Horizon"):
    """
    active_variants: dict mapping variant_name -> dict of style/label options
                     e.g. {"base": {"ls": "-", "marker": "o", "label": "Base"},
                           "fused": {"ls": ":", "marker": "x", "label": "Fused"},
                           "persistence": {"ls": "--", "marker": "s", "label": "Persistence", "is_ref": True}}
    metrics_to_plot: list of tuples (metric_key, y_label, title)
    """
    horizons = list(range(1, MAX_HORIZON + 1))
    import matplotlib.cm as cm
    _cmap = cm.get_cmap('rainbow', len(runs))
    colors = [_cmap(i) for i in range(len(runs))]
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14)
    
    for ax_idx, (metric_key, y_label, m_title) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        plotted_ref = False
        
        for run_idx, (run, per_horizon) in enumerate(zip(runs, all_metrics)):
            if per_horizon is None:
                continue
            c = colors[run_idx]
            run_lbl = run["label"]
            
            for var_name, var_opts in active_variants.items():
                is_ref = var_opts.get("is_ref", False)
                if is_ref and plotted_ref:
                    continue  # Only plot reference once (it's the same across runs)
                
                vals = []
                for h in range(MAX_HORIZON):
                    if var_name in per_horizon[h] and metric_key in per_horizon[h][var_name]:
                        data_list = per_horizon[h][var_name][metric_key]
                        vals.append(np.mean(data_list) if data_list else np.nan)
                    else:
                        vals.append(np.nan)
                
                label = var_opts.get("label", var_name)
                # If not reference, prepend run label
                if not is_ref:
                    label = f"{run_lbl} — {label}"
                
                color = "gray" if is_ref else c
                ax.plot(horizons, vals, color=color, ls=var_opts.get("ls", "-"), 
                        marker=var_opts.get("marker", "o"), ms=4, alpha=var_opts.get("alpha", 1.0),
                        label=label)
                
                if is_ref:
                    plotted_ref = True

        ax.set_xlabel("Horizon step")
        ax.set_ylabel(y_label)
        ax.set_title(m_title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# =============================================================================
# SECTION 4 — MAE across all horizons (line plot)
# =============================================================================
# %%

variants_sec4 = {
    "base":        {"ls": "-",  "marker": "o", "label": "base"},
    # "fused":       {"ls": ":",  "marker": "x", "label": "fused"},
    "persistence": {"ls": "--", "marker": "s", "label": "Persistence", "is_ref": True, "alpha": 0.6}
}

metrics_sec4 = [
    ("mae", "MAE", "Mean Absolute Error"),
    ("coverage", "Coverage error (mean)", "Coverage (under-prediction penalty)"),
    ("conservatism", "Conservatism error (mean)", "Conservatism (over-prediction penalty)")
]

plot_horizon_metrics(all_metrics, RUNS, variants_sec4, metrics_sec4, "MAE, Coverage & Conservatism vs Forecast Horizon")

# =============================================================================
# SECTION 5 — CVaR comparison across all horizons
# =============================================================================
# %%

for a in CVAR_LEVELS:
    variants_cvar = {
        f"base_cvar_{a}":  {"ls": "-",  "marker": "o", "label": "base"},
        # f"fused_cvar_{a}": {"ls": ":",  "marker": "x", "label": "fused"},
        "persistence":       {"ls": "--", "marker": "s", "label": "Persistence", "is_ref": True, "alpha": 0.6}
    }
    metrics_cvar = [
        ("mae", "MAE", f"CVaR MAE α={a}"),
        ("coverage", "Coverage", f"CVaR Coverage α={a}"),
        ("conservatism", "Conservatism", f"CVaR Conservatism α={a}")
    ]
    plot_horizon_metrics(all_metrics, RUNS, variants_cvar, metrics_cvar, f"CVaR (Expected Shortfall) α={a} vs Forecast Horizon")

# =============================================================================
# SECTION 6 — Conservatism rate (% of steps where model is more conservative than reference)
#   For drone path planning: higher = model is more conservative / safer.
# =============================================================================
# %%

COMPARISON_VARIANT = "base"          # What you want to evaluate (e.g., "base", "fused", "base_cvar_0.75")
REFERENCE_VARIANT  = "persistence"   # The baseline to beat (e.g., "persistence", "fused")

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title(f"Conservatism Rate vs Horizon\n(% timesteps where '{COMPARISON_VARIANT}' conservatism > '{REFERENCE_VARIANT}' conservatism)")
horizons = list(range(1, MAX_HORIZON + 1))
import matplotlib.cm as cm
_cmap = cm.get_cmap('rainbow', len(RUNS))
colors = [_cmap(i) for i in range(len(RUNS))]

for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c   = colors[run_idx]
    lbl = run["label"]

    rates = []
    for h in range(MAX_HORIZON):
        if COMPARISON_VARIANT in per_horizon[h] and REFERENCE_VARIANT in per_horizon[h]:
            cons_comp = per_horizon[h][COMPARISON_VARIANT]["conservatism"]
            cons_ref  = per_horizon[h][REFERENCE_VARIANT]["conservatism"]
            if cons_comp and cons_ref:
               m_is_more = [float(cm) > float(cp) for cm, cp in zip(cons_comp, cons_ref)]
               rates.append(np.mean(m_is_more) * 100.0)
            else:
               rates.append(np.nan)
        else:
            rates.append(np.nan)

    ax.plot(horizons, rates, color=c, ls="-", marker="o", ms=4, label=lbl)

ax.axhline(50, color="gray", ls=":", lw=1, label="50% (neutral)")
ax.set_xlabel("Horizon step")
ax.set_ylabel(f"% more conservative than {REFERENCE_VARIANT}")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 6.1 — Comprehensive rate (% of steps where model coverage < reference coverage i.e. better coverage)
# =============================================================================
# %%

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title(f"Comprehensive Rate vs Horizon\n(% timesteps where '{COMPARISON_VARIANT}' coverage error < '{REFERENCE_VARIANT}' coverage error)")

for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c   = colors[run_idx]
    lbl = run["label"]

    rates = []
    for h in range(MAX_HORIZON):
        if COMPARISON_VARIANT in per_horizon[h] and REFERENCE_VARIANT in per_horizon[h]:
            cov_comp = per_horizon[h][COMPARISON_VARIANT]["coverage"]
            cov_ref  = per_horizon[h][REFERENCE_VARIANT]["coverage"]
            if cov_comp and cov_ref:
                m_is_less = [float(cm) < float(cp) for cm, cp in zip(cov_comp, cov_ref)]
                rates.append(np.mean(m_is_less) * 100.0)
            else:
               rates.append(np.nan)
        else:
            rates.append(np.nan)

    ax.plot(horizons, rates, color=c, ls="-", marker="o", ms=4, label=lbl)

ax.axhline(50, color="gray", ls=":", lw=1, label="50% (neutral)")
ax.set_xlabel("Horizon step")
ax.set_ylabel(f"% more comprehensive than {REFERENCE_VARIANT}")
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

def _g(d, variant, metric, default=float("nan")):
    """Safe mean of a particular variant's metric."""
    if variant in d and metric in d[variant]:
        lst = d[variant][metric]
        return float(np.mean(lst)) if lst else default
    return default

print(f"\n{'='*90}")
print(f"Summary at horizon step h={HORIZON_DISPLAY}  ({HORIZON_DISPLAY+1} steps ahead)")
print(f"{'='*90}")
header = (f"{'Model':<22} {'MAE':>7} {'Cov':>7} {'Cons':>7} "
          f"{'MAE@75':>7} {'MAE@90':>7} {'MAE@95':>7} "
          f"{'Cov@75':>7} {'Cons@75':>7} "
          f"{'Lat(ms)':>8}")
print(header)
print("-" * len(header))

for run, per_horizon in zip(RUNS, all_metrics):
    if per_horizon is None:
        print(f"{run['label']:<22}  [no data]")
        continue
    m = per_horizon[HORIZON_DISPLAY]
    
    for var_name, lbl_suffix in [("base", "Base"), ("fused", "Fused")]:
        mae   = _g(m, var_name, "mae")
        cov   = _g(m, var_name, "coverage")
        cons  = _g(m, var_name, "conservatism")
        
        cm75  = _g(m, f"{var_name}_cvar_0.75", "mae")
        cm90  = _g(m, f"{var_name}_cvar_0.9", "mae")
        cm95  = _g(m, f"{var_name}_cvar_0.95", "mae")
        cv75  = _g(m, f"{var_name}_cvar_0.75", "coverage")
        cs75  = _g(m, f"{var_name}_cvar_0.75", "conservatism")
        
        lat = float(np.mean(per_horizon[0]["latencies_ms"])) if per_horizon[0]["latencies_ms"] else float("nan")
        if var_name == "fused":
            lat = float("nan") # We only measure base latency usually
            
        lbl = f"{run['label']} ({lbl_suffix})"
        print(f"{lbl:<22} {mae:>7.4f} {cov:>7.4f} {cons:>7.4f} "
              f"{cm75:>7.4f} {cm90:>7.4f} {cm95:>7.4f} "
              f"{cv75:>7.4f} {cs75:>7.4f} "
              f"{lat:>8.1f}")

# — Persistence row —
print("-" * len(header))
for run, per_horizon in zip(RUNS, all_metrics):
    if per_horizon is None:
        continue
    m = per_horizon[HORIZON_DISPLAY]
    mae_p = _g(m, "persistence", "mae")
    cov_p = _g(m, "persistence", "coverage")
    con_p = _g(m, "persistence", "conservatism")
    print(f"{'Persistence (ref)':<22} {mae_p:>7.4f} {cov_p:>7.4f} {con_p:>7.4f} "
          f"{'N/A':>7} {'N/A':>7} {'N/A':>7} "
          f"{'N/A':>7} {'N/A':>7} "
          f"{'N/A':>8}")
    break

# =============================================================================
# SECTION 9 — Visual inspection PER RUN: GT, μ, σ, Coverage, Conservatism, CVaR
# =============================================================================
# %%

EPISODE_IDX = 2   # which episode to inspect
TIME_IDX    = 3   # i-th stride step within the episode (must be >= 1 for persistence)
HORIZON_VIS = 15  # 0-indexed forecast horizon to visualise

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
    data       = np.load(ep["path"])
    time_steps = data["time_steps"]

    if TIME_IDX >= len(time_steps) or TIME_IDX < 1:
        continue
        
    t = time_steps[TIME_IDX]
    
    gt_h       = data["gt_full"][t + 1 + HORIZON_VIS].astype(np.float32)
    pers_frame = data["gt_full"][t].astype(np.float32)

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
    data       = np.load(ep["path"])
    
    if "sample_mean" not in data or "sample_std" not in data:
        print(f"[{run['label']}] Skipping visual inspection for deprecated rollout {ep['path']}")
        continue

    time_steps = data["time_steps"]

    if TIME_IDX >= len(time_steps):
        continue
    t = time_steps[TIME_IDX]

    gt_h = data["gt_full"][t + 1 + HORIZON_VIS].astype(np.float32)
    samples_h = data["sample"][TIME_IDX, HORIZON_VIS].astype(np.float32)
    has_dist  = "mean" in data and "std" in data
    mu_raw    = data["mean"][TIME_IDX, HORIZON_VIS].astype(np.float32) if has_dist else None
    std_raw   = data["std"][TIME_IDX, HORIZON_VIS].astype(np.float32) if has_dist else None
    
    has_sample_dist = "sample_mean" in data and "sample_std" in data
    mu_sample_raw   = data["sample_mean"][TIME_IDX, HORIZON_VIS].astype(np.float32) if has_sample_dist else None
    std_sample_raw  = data["sample_std"][TIME_IDX, HORIZON_VIS].astype(np.float32) if has_sample_dist else None

    mean_h = samples_h.mean(axis=0)
    std_h  = samples_h.std(axis=0)

    if mu_raw is None: mu_raw = mean_h
    if std_raw is None: std_raw = std_h
    
    mu_base = mu_raw[0] if mu_raw.ndim == 3 else mu_raw
    std_base = std_raw[0] if std_raw is not None and std_raw.ndim == 3 else std_raw
    mu_lbl   = "Model μ (dist)" if has_dist else "Model μ (samples)"
    
    mu_fused, std_fused = fuse_gaussians(mu_sample_raw, std_sample_raw)
    
    _plot_vis(run["label"] + " (Base)", gt_h, mu_base, std_base, t, mu_lbl)
    _plot_vis(run["label"] + " (Fused)", gt_h, mu_fused, std_fused, t, "Model μ (fused)")

# =============================================================================
# SECTION 10 — Continuous F2 / Soft IoU Plots
# =============================================================================
# %%

variants_sec10 = {
    "base":        {"ls": "-",  "marker": "o", "label": "base"},
    "fused":       {"ls": ":",  "marker": "x", "label": "fused"},
    "persistence": {"ls": "--", "marker": "s", "label": "Persistence", "is_ref": True, "alpha": 0.6}
}

metrics_sec10 = [
    ("soft_f2", "Soft F2 Score", "Continuous F2 (Penalizes Underprediction)"),
    ("soft_iou", "Soft IoU", "Soft IoU (Spatial Overlap)")
]

plot_horizon_metrics(all_metrics, RUNS, variants_sec10, metrics_sec10, "Spatial Forecasting Quality (Higher is better)")


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
    if r["label"] == "PFNO-old" or r["label"] == "PFNO":
        pfno_run_idx = i
        break

if pfno_run_idx != -1 and all_metrics[pfno_run_idx] is not None:
    lats = all_metrics[pfno_run_idx][0]["latencies_ms"]
    pfno_lat_ms = np.mean(lats) if lats else 0.0
    pfno_total_lat = pfno_lat_ms * MAX_HORIZON
    pfno_overbound_active = np.mean([np.mean(all_metrics[pfno_run_idx][h]["base"]["overbound_active"]) for h in range(MAX_HORIZON) if "base" in all_metrics[pfno_run_idx][h]])
    pfno_overbound_domain = np.mean([np.mean(all_metrics[pfno_run_idx][h]["base"]["overbound_domain"]) for h in range(MAX_HORIZON) if "base" in all_metrics[pfno_run_idx][h]])

    print(f"PFNO inference rate per rollout step: {pfno_lat_ms:.2f} ms")
    print(f"PFNO yielding a total {horizon_time_s:.1f} s horizon computation time of roughly {pfno_total_lat:.2f} ms")
    print(f"PFNO successfully overbounds the active smoke front in nearly {pfno_overbound_active:.2f}% of configurations")
    print(f"(Domain overbound: {pfno_overbound_domain:.2f}%)\n")

# -- IEEE Paper Plots --
# Configure which variants to plot in the paper graphics:
IEEE_VARIANTS = {
    "base": {"ls": "-", "marker": "o", "label": "Model (Base)"},
    # "fused": {"ls": ":", "marker": "x", "label": "Model (Fused)"},
    # "base_cvar_0.75": {"ls": "-", "marker": "v", "label": "Base (CVaR 0.75)"},
    "persistence": {"ls": "--", "marker": "s", "label": "Persistence", "is_ref": True, "color": "gray"}
}

# Plot 1: Overbound percentage / Coverage (strictly upper bounds)

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.set_xlabel("Forecast Horizon (s)")
ax1.set_ylabel("Active Front Overbound (%)")    

horizon_times = [h * 0.1 for h in range(1, MAX_HORIZON + 1)]
plotted_refs = set()

for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c = colors[run_idx]
    lbl = run["label"]

    for var_name, var_opts in IEEE_VARIANTS.items():
        is_ref = var_opts.get("is_ref", False)
        if is_ref and var_name in plotted_refs:
            continue
            
        ob_vals = []
        for h in range(MAX_HORIZON):
            if var_name in per_horizon[h] and per_horizon[h][var_name]["overbound_active"]:
                ob_vals.append(np.mean(per_horizon[h][var_name]["overbound_active"]))
            else:
                ob_vals.append(np.nan)
                
        label = var_opts.get("label", var_name)
        if not is_ref:
            label = f"{lbl} — {label}"
        color = var_opts.get("color", c)
        
        ax1.plot(horizon_times, ob_vals, color=color, ls=var_opts.get("ls", "-"), 
                 marker=var_opts.get("marker", "o"), ms=4, label=label, zorder=2 if is_ref else 3)
        
        if is_ref:
            plotted_refs.add(var_name)

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

plotted_refs = set()
for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
    if per_horizon is None:
        continue
    c = colors[run_idx]
    lbl = run["label"]

    for var_name, var_opts in IEEE_VARIANTS.items():
        is_ref = var_opts.get("is_ref", False)
        if is_ref and var_name in plotted_refs:
            continue
            
        mae_vals = []
        for h in range(MAX_HORIZON):
            if var_name in per_horizon[h] and per_horizon[h][var_name]["mae"]:
                mae_vals.append(np.mean(per_horizon[h][var_name]["mae"]))
            else:
                mae_vals.append(np.nan)
                
        label = var_opts.get("label", var_name)
        if not is_ref:
            label = f"{lbl} — {label}"
        color = var_opts.get("color", c)
        
        ax2.plot(horizon_times, mae_vals, color=color, ls=var_opts.get("ls", "-"), 
                 marker=var_opts.get("marker", "o"), ms=4, label=label, zorder=2 if is_ref else 3)
        
        if is_ref:
            plotted_refs.add(var_name)

ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_title("Spatiotemporal Accuracy (MAE)")
plt.tight_layout()
plt.savefig("ieee_mae_plot.png")
plt.show()

# %%
