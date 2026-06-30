# %%
# =============================================================================
# 1. SETUP & RAW DATA CACHING
# =============================================================================
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import stats
from collections import defaultdict

sys.path.append(os.getcwd())
from src.visualization.plot_utils import set_ieee_plot_formatting
from projects.single_agent.step_04_evaluation.schema import EvaluationConfig

# Apply IEEE formatting
set_ieee_plot_formatting()

def load_config() -> EvaluationConfig:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f) or {}
    return EvaluationConfig.model_validate(yaml_data)

cfg = load_config()

# Construct RUNS from config
RUNS = [
    {
        "folder": os.path.join("projects/single_agent/step_04_evaluation", cfg.output_dir, m.name),
        "label":  m.name,
    }
    for m in cfg.models
]

def load_run_data(run: dict) -> list[str]:
    """Returns a list of .npz paths for a run."""
    folder = run["folder"]
    if not os.path.exists(folder):
        return []
    import re
    def try_extract_ep_idx(fname):
        match = re.search(r'ep_(\d+)', fname)
        return int(match.group(1)) if match else fname
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")], key=try_extract_ep_idx)
    return [os.path.join(folder, fname) for fname in files]

print("Loading raw rollout arrays into memory (This might take a while)...")
raw_run_data = []  # List of runs -> List of episodes -> Dict of raw arrays
for run in RUNS:
    ep_paths = load_run_data(run)
    episodes = []
    for path in tqdm.tqdm(ep_paths, desc=f"Loading {run['label']}"):
        data = np.load(path)
        if "mean" in data and "std" in data:
            # Load into memory to avoid closing file issues
            episodes.append({
                "path": path,
                "time_steps": data["time_steps"].copy(),
                "gt_full": data["gt_full"].copy(),
                "mean": data["mean"].copy(),
                "std": data["std"].copy(),
                "latency": data["latency"].copy() if "latency" in data else None
            })
    raw_run_data.append(episodes if episodes else None)

print("Data successfully loaded into RAM!")

# %%
# =============================================================================
# 2. METRICS CALCULATION
# Calcula las métricas a partir de los datos en RAM. Toma solo segundos.
# =============================================================================

# -- Hyperparameters --
MAX_HORIZON = cfg.max_horizon_eval
CVAR_LEVELS_TEST = [0.5, 0.75, 0.90, 0.95]
F2_BETA = 2.0
ACTIVE_THRESH = 1e-3

def cvar(pred_mean: np.ndarray, pred_std: np.ndarray, alpha: float) -> np.ndarray:
    cvar_vals = pred_mean + pred_std * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)
    return np.clip(cvar_vals, 0.0, 1.0)

def coverage_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.clip(gt.astype(np.float32) - pred.astype(np.float32), 0.0, None)

def conservatism_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.clip(pred.astype(np.float32) - gt.astype(np.float32), 0.0, None)

def overbound_percentage_active(pred: np.ndarray, gt: np.ndarray, thresh: float) -> float:
    active = gt > thresh
    if not np.any(active): return 100.0
    return float((pred[active] >= gt[active]).mean() * 100.0)

def soft_f_beta(pred, gt, beta):
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)
    num = (1 + beta**2) * tp
    den = (1 + beta**2) * tp + (beta**2) * fn + fp
    return float(num / (den + 1e-8))

def evaluate_metrics(pred, gt):
    return {
        "mae": float(np.abs(pred - gt).mean()),
        "coverage": float(coverage_error(pred, gt).mean()),
        "conservatism": float(conservatism_error(pred, gt).mean()),
        "soft_f2": soft_f_beta(pred, gt, beta=F2_BETA),
        "overbound_active": overbound_percentage_active(pred, gt, thresh=ACTIVE_THRESH),
    }

print("Calculating metrics...")
all_metrics = []
for run_idx, (run, episodes) in enumerate(zip(RUNS, raw_run_data)):
    if episodes is None or len(episodes) == 0:
        all_metrics.append(None)
        continue
    
    results_per_h = {h: {"latencies_ms": []} for h in range(MAX_HORIZON)}
    
    for ep in tqdm.tqdm(episodes, desc=f"Metrics [{run['label']}]"):
        for i, t in enumerate(ep["time_steps"]):
            gt_all = ep["gt_full"][t + 1 : t + 1 + MAX_HORIZON]
            gt_current = ep["gt_full"][t]
            max_h_avail = gt_all.shape[0]
            if max_h_avail == 0: continue
                
            mu_raw_all = ep["mean"][i]
            std_raw_all = ep["std"][i]
            if ep["latency"] is not None:
                latency_val = float(ep["latency"][i])
            else:
                latency_val = None

            for h in range(min(MAX_HORIZON, max_h_avail)):
                gt_h = gt_all[h]
                mu_base = mu_raw_all[h]
                std_base = std_raw_all[h]

                variants_dict = {"base": mu_base, "persistence": gt_current}
                for a in CVAR_LEVELS_TEST:
                    variants_dict[f"base_cvar_{a}"] = cvar(mu_base, std_base, a)

                res = results_per_h[h]
                for var_name, var_pred in variants_dict.items():
                    if var_name not in res: res[var_name] = defaultdict(list)
                    m_vals = evaluate_metrics(var_pred, gt_h)
                    for k, v in m_vals.items():
                        res[var_name][k].append(v)

                if latency_val is not None:
                    res["latencies_ms"].append(latency_val)
                    
    all_metrics.append(results_per_h)

print("Metrics ready for plotting!")

# %%
# =============================================================================
# 3. PLOTTING UTILS
# =============================================================================

def plot_horizon_metrics(all_metrics, runs, active_variants, metrics_to_plot, title="Metrics vs Horizon"):
    horizons = list(range(1, MAX_HORIZON + 1))
    import matplotlib.cm as cm
    _cmap = cm.get_cmap('rainbow', len(runs))
    colors = [_cmap(i) for i in range(len(runs))]
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1: axes = [axes]
    fig.suptitle(title, fontsize=14)
    
    for ax_idx, (metric_key, y_label, m_title) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        plotted_ref = False
        
        for run_idx, (run, per_horizon) in enumerate(zip(runs, all_metrics)):
            if per_horizon is None: continue
            c = colors[run_idx]
            run_lbl = run["label"]
            
            for var_name, var_opts in active_variants.items():
                is_ref = var_opts.get("is_ref", False)
                if is_ref and plotted_ref: continue
                
                vals = []
                for h in range(MAX_HORIZON):
                    if var_name in per_horizon[h] and metric_key in per_horizon[h][var_name]:
                        data_list = per_horizon[h][var_name][metric_key]
                        vals.append(np.mean(data_list) if data_list else np.nan)
                    else:
                        vals.append(np.nan)
                
                label = var_opts.get("label", var_name)
                if not is_ref: label = f"{run_lbl} — {label}"
                color = "gray" if is_ref else c
                
                ax.plot(horizons, vals, color=color, ls=var_opts.get("ls", "-"), 
                        marker=var_opts.get("marker", "o"), ms=4, alpha=var_opts.get("alpha", 1.0),
                        label=label)
                if is_ref: plotted_ref = True

        ax.set_xlabel("Horizon step")
        ax.set_ylabel(y_label)
        ax.set_title(m_title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# 4. PLOT: MAE, Coverage & Conservatism
# Modifica los diccionarios para graficar diferentes variantes
# =============================================================================
if any(m is not None for m in all_metrics):
    variants = {
        "base":        {"ls": "-",  "marker": "o", "label": "Mean Prediction"},
        "persistence": {"ls": "--", "marker": "s", "label": "Persistence", "is_ref": True, "alpha": 0.6}
    }
    
    metrics = [
        ("mae", "MAE", "Mean Absolute Error"),
        ("coverage", "Coverage error (mean)", "Coverage (under-prediction penalty)"),
        ("conservatism", "Conservatism error (mean)", "Conservatism (over-prediction penalty)")
    ]
    plot_horizon_metrics(all_metrics, RUNS, variants, metrics, "Base Metrics vs Forecast Horizon")
else:
    print("No metrics computed yet.")

# %%
# =============================================================================
# 5. PLOT: CVaR Evaluation (Expected Shortfall)
# =============================================================================
if any(m is not None for m in all_metrics):
    cvar_alpha = 0.90 # Change this to test other alpha levels in CVAR_LEVELS_TEST
    variants_cvar = {
        f"base_cvar_{cvar_alpha}":  {"ls": "-",  "marker": "x", "label": f"CVaR (a={cvar_alpha})"},
        "persistence":              {"ls": "--", "marker": "s", "label": "Persistence", "is_ref": True, "alpha": 0.6}
    }
    metrics_cvar = [
        ("mae", "MAE", f"CVaR MAE a={cvar_alpha}"),
        ("coverage", "Coverage", f"CVaR Coverage a={cvar_alpha}")
    ]
    plot_horizon_metrics(all_metrics, RUNS, variants_cvar, metrics_cvar, f"CVaR a={cvar_alpha} vs Forecast Horizon")

# %%
# =============================================================================
# 6. SUMMARY TABLE
# =============================================================================
HORIZON_DISPLAY = 10   # 0-indexed horizon step to inspect

if any(m is not None for m in all_metrics):
    print(f"\n{'='*90}")
    print(f"Summary at horizon step h={HORIZON_DISPLAY}  ({HORIZON_DISPLAY+1} steps ahead)")
    print(f"{'='*90}")
    
    header = f"{'Model':<22} {'MAE':>7} {'Cov':>7} {'Cons':>7} {'Lat(ms)':>8}"
    print(header)
    print("-" * len(header))
    
    for run, per_horizon in zip(RUNS, all_metrics):
        if per_horizon is None:
            print(f"{run['label']:<22}  [no data]")
            continue
            
        m = per_horizon[HORIZON_DISPLAY]
        var_name = "base"
        
        def _g(d, var, met):
            return float(np.mean(d[var][met])) if var in d and met in d[var] and d[var][met] else float("nan")

        mae   = _g(m, var_name, "mae")
        cov   = _g(m, var_name, "coverage")
        cons  = _g(m, var_name, "conservatism")
        
        lats = per_horizon[0]["latencies_ms"]
        lat = float(np.mean(lats)) if lats else float("nan")
            
        print(f"{run['label']:<22} {mae:>7.4f} {cov:>7.4f} {cons:>7.4f} {lat:>8.1f}")
    
    # Persistence row
    print("-" * len(header))
    for run, per_horizon in zip(RUNS, all_metrics):
        if per_horizon is not None:
            m = per_horizon[HORIZON_DISPLAY]
            mae_p = _g(m, "persistence", "mae")
            cov_p = _g(m, "persistence", "coverage")
            con_p = _g(m, "persistence", "conservatism")
            print(f"{'Persistence (ref)':<22} {mae_p:>7.4f} {cov_p:>7.4f} {con_p:>7.4f} {'N/A':>8}")
            break

# %%
# =============================================================================
# 7. VISUAL INSPECTION GRID
# =============================================================================
EPISODE_IDX = 0   
TIME_IDX    = 0   
HORIZON_VIS = 15  

if len(raw_run_data) > 0 and raw_run_data[0] is not None:
    # We take the first valid model to compare against ground truth
    ep = raw_run_data[0][EPISODE_IDX]
    t = ep["time_steps"][TIME_IDX]
    
    gt_h = ep["gt_full"][t + 1 + HORIZON_VIS]
    pred_h = ep["mean"][TIME_IDX, HORIZON_VIS]
    std_h = ep["std"][TIME_IDX, HORIZON_VIS]
    
    cov_map = coverage_error(pred_h, gt_h)
    mae_map = np.abs(pred_h - gt_h)
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(f"Visual Inspection — ep {EPISODE_IDX}, t={t}, h={HORIZON_VIS+1}", fontsize=13)
    
    def _im(ax, img, title, cmap, vmax=None):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    _im(axes[0], gt_h, "Ground Truth", "inferno", 1)
    _im(axes[1], pred_h, "Predicted Mean", "inferno", 1)
    _im(axes[2], std_h, "Predicted Std", "magma", None)
    _im(axes[3], cov_map, f"Coverage Error (μ={cov_map.mean():.4f})", "Reds", 0.5)
    
    plt.tight_layout()
    plt.show()
