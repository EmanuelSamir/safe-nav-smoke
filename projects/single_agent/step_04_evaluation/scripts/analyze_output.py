# %%
# =============================================================================
# 1. SETUP & RAW DATA CACHING
# =============================================================================
import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Si estás en VS Code Interactive, también ayuda poner esta directiva mágica:
%matplotlib inline
import tqdm
from scipy import stats
from collections import defaultdict

sys.path.append("/home/emunoz/dev/safe-nav-smoke")
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

# Toggle this for quick testing
DRY_RUN = True  # Set to False to load all episodes
MAX_EPISODES_DRY_RUN = 10

# Construct RUNS from config
RUNS = [
    {
        "folder": os.path.join("/home/emunoz/dev/safe-nav-smoke", "outputs", cfg.project_name, cfg.sub_project_name, m.name),
        "label":  m.name,
        "config_path": m.config_path,
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
    
    if DRY_RUN:
        ep_paths = ep_paths[:MAX_EPISODES_DRY_RUN]
        print(f"DRY RUN: Limiting to {MAX_EPISODES_DRY_RUN} episodes for {run['label']}")

    episodes = []
    for path in tqdm.tqdm(ep_paths, desc=f"Loading {run['label']}"):
        data = np.load(path)
        if "mean" in data:
            # Load into memory to avoid closing file issues
            mean_data = data["mean"].copy()
            std_data = data["std"].copy() if "std" in data else np.zeros_like(mean_data)
            
            episodes.append({
                "path": path,
                "time_steps": data["time_steps"].copy(),
                "gt_full": data["gt_full"].copy(),
                "mean": mean_data,
                "std": std_data,
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
    
    # Read h_ctx from the model config
    model_cfg_data = {}
    if os.path.exists(run.get("config_path", "")):
        with open(run["config_path"], "r") as f:
            model_cfg_data = yaml.safe_load(f) or {}
    h_ctx = model_cfg_data.get("model", {}).get("h_ctx", 1)

    for ep in tqdm.tqdm(episodes, desc=f"Metrics [{run['label']}]"):
        for i, t in enumerate(ep["time_steps"]):
            t_start = t + 1
            gt_all = ep["gt_full"][t_start : t_start + MAX_HORIZON]
            gt_current = ep["gt_full"][t_start - 1]
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
TIME_IDX    = 10   
HORIZON_VIS = 10  

def _im(ax, img, title, **kw):
    im = ax.imshow(img, origin="lower", **kw)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def _plot_vis(label, gt_h, pred_h, sigma_h, t, mu_lbl, cvar_label="CVaR-75"):
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

if len(raw_run_data) > 0 and raw_run_data[0] is not None:
    # Persistence
    ep = raw_run_data[0][EPISODE_IDX]
    t = ep["time_steps"][TIME_IDX]
    gt_h = ep["gt_full"][t + 1 + HORIZON_VIS].astype(np.float32)
    pers_frame = ep["gt_full"][t].astype(np.float32)
    _plot_vis("Persistence (baseline)", gt_h, pers_frame, np.zeros_like(pers_frame), t, "Persistence frame", cvar_label="CVaR-75 (σ=0)")
    
    # Models
    for run_idx, (run, episodes) in enumerate(zip(RUNS, raw_run_data)):
        if episodes is None or EPISODE_IDX >= len(episodes): continue
        ep = episodes[EPISODE_IDX]
        t = ep["time_steps"][TIME_IDX]
        gt_h_m = ep["gt_full"][t + 1 + HORIZON_VIS].astype(np.float32)
        
        mu_raw = ep["mean"][TIME_IDX, HORIZON_VIS].astype(np.float32)
        std_raw = ep["std"][TIME_IDX, HORIZON_VIS].astype(np.float32)
        
        _plot_vis(run["label"] + " (Base)", gt_h_m, mu_raw, std_raw, t, "Model μ")

# %%
# %%
# =============================================================================
# 8. PROBABILISTIC EVALUATION & DESEMPATE (NLL, CRPS, COVERAGE & SHARPNESS)
# =============================================================================
HORIZON_PROB = 2  # El paso del horizonte (0-indexed) que quieres evaluar

if len(raw_run_data) > 0:
    print(f"\n{'='*115}")
    print(f"Probabilistic Desempate at horizon step h={HORIZON_PROB} ({HORIZON_PROB+1} steps ahead)")
    print(f"{'='*115}")
    
    header = f"{'Model':<22} {'NLL':>10} {'CRPS':>10} {'Cov 95%':>12} {'Sharpness':>12}"
    print(header)
    print("-" * len(header))
    
    for run_idx, (run, episodes) in enumerate(zip(RUNS, raw_run_data)):
        if episodes is None or len(episodes) == 0:
            print(f"{run['label']:<22}  [no data]")
            continue
            
        model_cfg_data = {}
        if os.path.exists(run.get("config_path", "")):
            with open(run["config_path"], "r") as f:
                model_cfg_data = yaml.safe_load(f) or {}
        h_ctx = model_cfg_data.get("model", {}).get("h_ctx", 1)
        
        # Listas para acumular las métricas de todos los episodios en este horizonte
        nll_list = []
        crps_list = []
        cov95_list = []
        sharp_list = []
        
        for ep in episodes:
            for i, t in enumerate(ep["time_steps"]):
                t_start = t + 1
                gt_all = ep["gt_full"][t_start : t_start + MAX_HORIZON]
                max_h_avail = gt_all.shape[0]
                
                if HORIZON_PROB >= max_h_avail: 
                    continue
                    
                gt_h = gt_all[HORIZON_PROB]
                mu_h = ep["mean"][i, HORIZON_PROB]
                std_h = ep["std"][i, HORIZON_PROB]
                
                # --- Cálculo de Métricas Probabilísticas ---
                eps = 1e-6
                safe_std = np.maximum(std_h, eps)
                
                # 1. NLL Gaussiano
                nll = 0.5 * np.log(2 * np.pi * (safe_std**2)) + ((gt_h - mu_h) ** 2) / (2 * (safe_std**2))
                nll_list.append(nll.mean())
                
                # 2. CRPS Analítico por píxel
                z = (gt_h - mu_h) / safe_std
                norm_cdf = stats.norm.cdf(z)
                norm_pdf = stats.norm.pdf(z)
                crps = safe_std * (z * (2 * norm_cdf - 1) + 2 * norm_pdf - 1 / np.sqrt(np.pi))
                crps_list.append(crps.mean())
                
                # 3. Cobertura empírica al 95% (Z-score = 1.96)
                lower_bound = mu_h - 1.96 * std_h
                upper_bound = mu_h + 1.96 * std_h
                inside = (gt_h >= lower_bound) & (gt_h <= upper_bound)
                cov95_list.append(inside.mean() * 100.0)
                
                # 4. Sharpness (Magnitud promedio de la incertidumbre)
                sharp_list.append(std_h.mean())
                
        # Promediar los resultados del modelo
        avg_nll = np.mean(nll_list) if nll_list else float("nan")
        avg_crps = np.mean(crps_list) if crps_list else float("nan")
        avg_cov95 = np.mean(cov95_list) if cov95_list else float("nan")
        avg_sharp = np.mean(sharp_list) if sharp_list else float("nan")
        
        print(f"{run['label']:<22} {avg_nll:>10.2f} {avg_crps:>10.4f} {avg_cov95:>11.1f}% {avg_sharp:>12.4f}")
    print("-" * len(header))
else:
    print("No raw data available for probabilistic evaluation.")

# %%
# =============================================================================
# 9. PROBABILISTIC VISUALIZATION: CALIBRATION & FAILURE ZONES
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

EPISODE_IDX = 1
TIME_IDX    = 5
HORIZON_VIS = 1

if len(raw_run_data) >= 2 and all(r is not None for r in raw_run_data[:2]):
    ep_m0 = raw_run_data[0][EPISODE_IDX]
    ep_m1 = raw_run_data[1][EPISODE_IDX]
    
    t = ep_m0["time_steps"][TIME_IDX]
    gt_h = ep_m0["gt_full"][t + 1 + HORIZON_VIS]
    
    # Modelo 0 (ej. PFNO)
    pred_m0 = ep_m0["mean"][TIME_IDX, HORIZON_VIS]
    std_m0  = ep_m0["std"][TIME_IDX, HORIZON_VIS]
    
    # Modelo 1 (ej. ConvLSTM)
    pred_m1 = ep_m1["mean"][TIME_IDX, HORIZON_VIS]
    std_m1  = ep_m1["std"][TIME_IDX, HORIZON_VIS]
    
    def compute_calibration_zones(mean, std, gt):
        """
        Clasifica cada píxel en base al intervalo de confianza del 95% (1.96 * std)
        0: Bien calibrado (El GT cae dentro del intervalo)
        1: Subestimado / Exceso de confianza (¡Peligro! El GT es mayor que el límite superior)
        -1: Sobreestimado / Conservador (El GT es menor que el límite inferior)
        """
        z_score = 1.96
        upper_bound = mean + z_score * std
        lower_bound = mean - z_score * std
        
        zones = np.zeros_like(gt)                     # 0 = Calibrado (Gris)
        zones[gt > upper_bound] = 1                   # 1 = Subestimado / Humo imprevisto (Rojo)
        zones[gt < lower_bound] = -1                  # -1 = Sobreestimado / Alarma falsa (Azul)
        return zones

    zones_m0 = compute_calibration_zones(pred_m0, std_m0, gt_h)
    zones_m1 = compute_calibration_zones(pred_m1, std_m1, gt_h)
    
    # Configuración de la figura: 2 filas x 4 columnas
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"Spatial Calibration & Failure Zones Analysis\nEpisode {EPISODE_IDX}, t={t}, Horizon={HORIZON_VIS+1} steps ahead", fontsize=14, y=0.98)
    
    # Definimos un mapa de colores discreto para las zonas de falla
    # -1: Azul (Sobreestimado), 0: Gris claro (Calibrado), 1: Rojo (Subestimado)
    cmap_zones = ListedColormap(['#1f77b4', '#e0e0e0', '#d62728']) 
    
    def _plot_cell(ax, img, title, cmap, vmax=None, vmin=None):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        return im

    # --- FILA 1: MODELO 0 ---
    _plot_cell(axes[0, 0], gt_h, "Ground Truth (Humo Real)", "inferno", vmin=0, vmax=1)
    _plot_cell(axes[0, 1], pred_m0, f"{RUNS[0]['label']}\nPredicted Mean ($\mu$)", "inferno", vmin=0, vmax=1)
    _plot_cell(axes[0, 2], std_m0, f"{RUNS[0]['label']}\nPredicted Std ($\sigma$)", "magma")
    
    im_z0 = _plot_cell(axes[0, 3], zones_m0, f"{RUNS[0]['label']}\nCalibration Zones (95% CI)", cmap_zones, vmin=-1, vmax=1)
    cbar0 = plt.colorbar(im_z0, ax=axes[0, 3], fraction=0.046, pad=0.04, ticks=[-0.66, 0, 0.66])
    cbar0.ax.set_yticklabels(['Sobreestimado\n(Azul)', 'Calibrado\n(Gris)', 'Subestimado\n(Rojo)'], fontsize=8)

    # --- FILA 2: MODELO 1 ---
    _plot_cell(axes[1, 0], gt_h, "Ground Truth (Humo Real)", "inferno", vmin=0, vmax=1)
    _plot_cell(axes[1, 1], pred_m1, f"{RUNS[1]['label']}\nPredicted Mean ($\mu$)", "inferno", vmin=0, vmax=1)
    _plot_cell(axes[1, 2], std_m1, f"{RUNS[1]['label']}\nPredicted Std ($\sigma$)", "magma")
    
    im_z1 = _plot_cell(axes[1, 3], zones_m1, f"{RUNS[1]['label']}\nCalibration Zones (95% CI)", cmap_zones, vmin=-1, vmax=1)
    cbar1 = plt.colorbar(im_z1, ax=axes[1, 3], fraction=0.046, pad=0.04, ticks=[-0.66, 0, 0.66])
    cbar1.ax.set_yticklabels(['Sobreestimado\n(Azul)', 'Calibrado\n(Gris)', 'Subestimado\n(Rojo)'], fontsize=8)
    
    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# 10. CONSERVATISM & COMPREHENSIVE RATES
# =============================================================================
if any(m is not None for m in all_metrics):
    COMPARISON_VARIANT = "base"
    REFERENCE_VARIANT  = "persistence"

    import matplotlib.cm as cm
    _cmap = cm.get_cmap('rainbow', len(RUNS))
    colors = [_cmap(i) for i in range(len(RUNS))]
    horizons = list(range(1, MAX_HORIZON + 1))

    # Conservatism Rate
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Conservatism Rate vs Horizon\n(% timesteps where '{COMPARISON_VARIANT}' conservatism > '{REFERENCE_VARIANT}' conservatism)")

    for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
        if per_horizon is None: continue
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

    # Comprehensive Rate
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Comprehensive Rate vs Horizon\n(% timesteps where '{COMPARISON_VARIANT}' coverage error < '{REFERENCE_VARIANT}' coverage error)")

    for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
        if per_horizon is None: continue
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

# %%
# =============================================================================
# 11. IEEE PAPER FINAL PLOTS AND TEXTS
# =============================================================================
if any(m is not None for m in all_metrics):
    print(f"\n{'='*90}")
    print("IEEE PAPER EXTRACTED VALUES")
    print(f"{'='*90}")

    horizon_time_s = MAX_HORIZON * 0.1
    print(f"Total Horizon Time: {horizon_time_s:.1f} s (N={MAX_HORIZON} steps)")

    run_idx_target = 0
    if len(all_metrics) > run_idx_target and all_metrics[run_idx_target] is not None:
        lats = all_metrics[run_idx_target][0]["latencies_ms"]
        pfno_lat_ms = np.mean(lats) if lats else 0.0
        pfno_total_lat = pfno_lat_ms * MAX_HORIZON
        pfno_overbound_active = np.mean([np.mean(all_metrics[run_idx_target][h]["base"]["overbound_active"]) for h in range(MAX_HORIZON) if "base" in all_metrics[run_idx_target][h]])

        print(f"Model ({RUNS[run_idx_target]['label']}) inference rate per rollout step: {pfno_lat_ms:.2f} ms")
        print(f"Yielding a total {horizon_time_s:.1f} s horizon computation time of roughly {pfno_total_lat:.2f} ms")
        print(f"Successfully overbounds the active smoke front in nearly {pfno_overbound_active:.2f}% of configurations\n")

    IEEE_VARIANTS = {
        "base": {"ls": "-", "marker": "o", "label": "Model (Base)"},
        "persistence": {"ls": "--", "marker": "s", "label": "Persistence", "is_ref": True, "color": "gray"}
    }

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel("Forecast Horizon (s)")
    ax1.set_ylabel("Active Front Overbound (%)")    

    horizon_times = [h * 0.1 for h in range(1, MAX_HORIZON + 1)]
    plotted_refs = set()

    for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
        if per_horizon is None: continue
        c = colors[run_idx]
        lbl = run["label"]

        for var_name, var_opts in IEEE_VARIANTS.items():
            is_ref = var_opts.get("is_ref", False)
            if is_ref and var_name in plotted_refs: continue
                
            ob_vals = []
            for h in range(MAX_HORIZON):
                if var_name in per_horizon[h] and per_horizon[h][var_name]["overbound_active"]:
                    ob_vals.append(np.mean(per_horizon[h][var_name]["overbound_active"]))
                else:
                    ob_vals.append(np.nan)
                    
            label = var_opts.get("label", var_name)
            if not is_ref: label = f"{lbl} — {label}"
            color = var_opts.get("color", c)
            
            ax1.plot(horizon_times, ob_vals, color=color, ls=var_opts.get("ls", "-"), 
                     marker=var_opts.get("marker", "o"), ms=4, label=label, zorder=2 if is_ref else 3)
            if is_ref: plotted_refs.add(var_name)

    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title("Safety Boundary Overestimation")
    plt.tight_layout()
    plt.savefig("ieee_coverage_plot.png")
    plt.show()

    fig, ax2 = plt.subplots(figsize=(6, 4))
    ax2.set_xlabel("Forecast Horizon (s)")
    ax2.set_ylabel("Mean Absolute Error")

    plotted_refs = set()
    for run_idx, (run, per_horizon) in enumerate(zip(RUNS, all_metrics)):
        if per_horizon is None: continue
        c = colors[run_idx]
        lbl = run["label"]

        for var_name, var_opts in IEEE_VARIANTS.items():
            is_ref = var_opts.get("is_ref", False)
            if is_ref and var_name in plotted_refs: continue
                
            mae_vals = []
            for h in range(MAX_HORIZON):
                if var_name in per_horizon[h] and per_horizon[h][var_name]["mae"]:
                    mae_vals.append(np.mean(per_horizon[h][var_name]["mae"]))
                else:
                    mae_vals.append(np.nan)
                    
            label = var_opts.get("label", var_name)
            if not is_ref: label = f"{lbl} — {label}"
            color = var_opts.get("color", c)
            
            ax2.plot(horizon_times, mae_vals, color=color, ls=var_opts.get("ls", "-"), 
                     marker=var_opts.get("marker", "o"), ms=4, label=label, zorder=2 if is_ref else 3)
            if is_ref: plotted_refs.add(var_name)

    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title("Spatiotemporal Accuracy (MAE)")
    plt.tight_layout()
    plt.savefig("ieee_mae_plot.png")
    plt.show()
# %%