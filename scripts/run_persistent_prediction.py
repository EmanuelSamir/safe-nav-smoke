#!/usr/bin/env python3
"""
FNO Multi-Agent Persistent Prediction Proof-of-Concept.

This script runs a multi-agent crossing benchmark with local mapping,
maintaining 15 global rollouts updated over time via collage patching.
It separates aleatoric and epistemic uncertainty and plots them in a 4-panel dashboard.
"""
import os
import sys
import time
import argparse
from collections import deque

import numpy as np
import scipy.stats as stats
import torch
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

# Add project root to python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

# Robust imports supporting various pythonpath configurations
try:
    from agents.basic_robot import RobotParams
except ImportError:
    from src.agents.basic_robot import RobotParams

try:
    from controllers.multi_dual_guard_cbf_ctrl import MultiDualGuardCBFCtrl
except ImportError:
    from src.controllers.multi_dual_guard_cbf_ctrl import MultiDualGuardCBFCtrl

try:
    from controllers.mppi_ctrl import MPPICtrlParams
except ImportError:
    from src.controllers.mppi_ctrl import MPPICtrlParams

try:
    from envs.smoke_env import SmokeEnv, EnvParams, SmokeParams
except ImportError:
    try:
        from env.smoke_env import SmokeEnv, EnvParams, SmokeParams
    except ImportError:
        from src.env.smoke_env import SmokeEnv, EnvParams, SmokeParams

try:
    from envs.simulator.sensor import GlobalSensorParams, GlobalSensor, DownwardsSensorParams
except ImportError:
    try:
        from env.simulator.sensor import GlobalSensorParams, GlobalSensor, DownwardsSensorParams
    except ImportError:
        from src.env.simulator.sensor import GlobalSensorParams, GlobalSensor, DownwardsSensorParams

try:
    from models.fno import FNO, FNOConfig
except ImportError:
    from src.models.fno import FNO, FNOConfig

try:
    from wrappers.smoke_forecast_wrapper import SmokeForecastWrapper
except ImportError:
    from src.wrappers.smoke_forecast_wrapper import SmokeForecastWrapper

# Inject training schema classes into __main__ to avoid pickle unpickling AttributeError
try:
    from training.train_fno import (
        FNOTrainingConfigSchema, FNOTrainingGlobalSchema, TrainingDataConfig,
        TrainingLossConfig, TrainingOptimizerConfig, TrainingCheckpointConfig,
        TrainingVisualizerConfig
    )
except ImportError:
    from src.training.train_fno import (
        FNOTrainingConfigSchema, FNOTrainingGlobalSchema, TrainingDataConfig,
        TrainingLossConfig, TrainingOptimizerConfig, TrainingCheckpointConfig,
        TrainingVisualizerConfig
    )

import __main__
__main__.FNOTrainingConfigSchema = FNOTrainingConfigSchema
__main__.FNOTrainingGlobalSchema = FNOTrainingGlobalSchema
__main__.TrainingDataConfig = TrainingDataConfig
__main__.TrainingLossConfig = TrainingLossConfig
__main__.TrainingOptimizerConfig = TrainingOptimizerConfig
__main__.TrainingCheckpointConfig = TrainingCheckpointConfig
__main__.TrainingVisualizerConfig = TrainingVisualizerConfig



def compute_cvar_risk(mean: np.ndarray, std: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """Gaussian CVaR: μ + σ · φ(Φ⁻¹(α)) / (1-α)"""
    pdf = stats.norm.pdf(stats.norm.ppf(alpha))
    risk = mean + std * pdf / (1 - alpha)
    return np.clip(risk, 0.0, 1.0).astype(np.float32)


def forecast_horizon(model, seed_frames, t_start, horizon, seq_len_ref, device, mode="sample"):
    """
    Batched autoregressive forecasting of the FNO model over the planning horizon.
    """
    B_batch, h_ctx, H_dim, W_dim = seed_frames.shape
    ctx = seed_frames.clone()
    t_offset = t_start
    h_pred = model.cfg.h_pred
    ref = max(seq_len_ref - 1, 1)
    
    preds = []
    while len(preds) < horizon:
        t_abs = torch.arange(t_offset, t_offset + h_ctx, device=device, dtype=torch.float32)
        times = (t_abs / ref).unsqueeze(0).expand(B_batch, -1)  # (B_batch, h_ctx)
        
        with torch.no_grad():
            # FNO forward outputs List[Normal] of length h_pred, each shape (B_batch, H_dim, W_dim, 1)
            dists = model(ctx, times)
            
        new_frames_for_ctx = []
        eps = torch.randn(B_batch, 1, 1, 1, device=device)
        for d in dists:
            if len(preds) >= horizon:
                break
                
            mu = d.mean
            sigma = d.stddev
            
            if mode == "mean":
                sampled = mu
            elif mode == "sample":
                sampled = mu + sigma * eps
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
            preds.append({
                "mean": mu.squeeze(-1),       # (B_batch, H_dim, W_dim)
                "std": sigma.squeeze(-1),      # (B_batch, H_dim, W_dim)
                "sample": sampled.squeeze(-1)   # (B_batch, H_dim, W_dim)
            })
            new_frames_for_ctx.append(sampled)
            
        # Slide context window
        n_slide = len(new_frames_for_ctx)
        new_stack = torch.cat([f.permute(0, 3, 1, 2) for f in new_frames_for_ctx], dim=1)  # (B_batch, n_slide, H_dim, W_dim)
        ctx = torch.cat([ctx, new_stack], dim=1)[:, -h_ctx:]
        t_offset += n_slide
        
    return preds


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Persistent Prediction Proof-of-Concept")
    parser.add_argument("--no_gif", action="store_true", help="Disable saving dynamic plot as a GIF")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()
    save_gif = not args.no_gif

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # 1. Load FNO Checkpoint
    checkpoint_path = "/home/emunoz/dev/safe-nav-smoke/outputs/training/fno/2026-06-17/00-36-09/checkpoints/last.ckpt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    model_hp = hp.get("model", hp) if isinstance(hp, dict) else {}
    
    # Establish default fallback config
    fallback_cfg = {
        "h_ctx": 5,
        "h_pred": 5,
        "modes_t": 2,
        "modes_h": 8,
        "modes_w": 8,
        "width": 32,
        "n_layers": 4,
        "use_grid": True,
        "use_time": True,
        "min_std": 1e-4,
        "seq_len_ref": 30
    }
    for k, v in fallback_cfg.items():
        if k not in model_hp:
            model_hp[k] = v
            
    valid_keys = set(FNOConfig.model_fields.keys())
    clean_model_hp = {k: v for k, v in model_hp.items() if k in valid_keys}
    fno_cfg = FNOConfig(**clean_model_hp)
    
    model = FNO(fno_cfg)
    
    # Load state dict robustly
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("FNO Model loaded successfully.")

    # 2. Setup Benchmark Scenario Setup
    num_agents = 6
    x_size = 30.0
    y_size = 20.0
    dt = 0.1
    max_steps = 200
    collision_radius = 0.4
    goal_radius = 0.6
    d_safe = 1.6
    r_sense = 8.0
    
    # Decoupled belief ensemble and cost parameters
    half_width = 5.0    # Square FOV half-width
    cost_alpha = 1.0    # Aleatoric uncertainty cost coefficient (risk aversion)
    cost_beta = 0.5     # Epistemic uncertainty cost coefficient (exploration incentive)
    
    initial_positions = [
        [3.0, 3.0],   # agent_0 (left bottom)
        [3.0, 10.0],  # agent_1 (left middle)
        [3.0, 17.0],  # agent_2 (left top)
        [27.0, 3.0],  # agent_3 (right bottom)
        [27.0, 10.0], # agent_4 (right middle)
        [27.0, 17.0], # agent_5 (right top)
    ]
    
    goal_locations = [
        [27.0, 17.0], # agent_0 target
        [27.0, 10.0], # agent_1 target
        [27.0, 3.0],  # agent_2 target
        [3.0, 17.0],  # agent_3 target
        [3.0, 10.0],  # agent_4 target
        [3.0, 3.0],   # agent_5 target
    ]
    
    initial_headings = []
    for start, goal in zip(initial_positions, goal_locations):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        initial_headings.append(np.arctan2(dy, dx))
        
    tailored_blobs = {
        "case_1": {
            "x_pos": [10.0, 10.0, 10.0, 20.0, 20.0],
            "y_pos": [3.0, 10.0, 17.0, 6.0, 14.0],
        },
        "case_2": {
            "x_pos": [20.0, 20.0, 20.0, 10.0, 10.0],
            "y_pos": [3.0, 10.0, 17.0, 6.0, 14.0],
        },
        "case_3": {
            "x_pos": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "y_pos": [3.0, 8.0, 13.0, 7.0, 12.0, 17.0],
        },
        "case_4": {
            "x_pos": [20.0, 20.0, 20.0, 10.0, 10.0, 10.0],
            "y_pos": [3.0, 8.0, 13.0, 7.0, 12.0, 17.0],
        },
        "case_5": {
            "x_pos": [8.0, 8.0, 15.0, 22.0, 22.0],
            "y_pos": [4.0, 16.0, 10.0, 4.0, 16.0],
        },
        "case_6": {
            "x_pos": [8.0, 15.0, 15.0, 15.0, 22.0],
            "y_pos": [10.0, 4.0, 10.0, 16.0, 10.0],
        },
    }

    # Select random case blobs config
    case_idx = np.random.randint(1, len(tailored_blobs) + 1)
    case_blobs = tailored_blobs[f"case_{case_idx}"]
    num_blobs = len(case_blobs["x_pos"])
    blobs_cfg = []
    for i in range(num_blobs):
        spread_rate = np.random.uniform(1.0, 3.0)
        blobs_cfg.append({
            "x": case_blobs["x_pos"][i],
            "y": case_blobs["y_pos"][i],
            "intensity": 1.0,
            "spread": spread_rate
        })

    # 3. Environment and Multi-Agent Controller Setup
    env_cfg = {
        "world_x_size": x_size,
        "world_y_size": y_size,
        "max_steps": max_steps,
        "clock": dt,
        "render": "none",
        "playback_path": None,
        "num_agents": num_agents,
        "collision_radius": collision_radius,
        "terminate_on_collision": False,
        "collision_penalty": -10.0,
        "goal_radius": goal_radius,
        "goal_locations": goal_locations,
        "sensor": {
            "type": "global"
        },
        "smoke": {
            "resolution": 0.2,
            "blobs": blobs_cfg
        },
        "robot": {
            "type": "dubins2d",
            "dt": dt,
            "action_dim": 2,
            "state_dim": 3,
            "action_max": [5.0, 4.0],
            "action_min": [0.5, -4.0],
            "state_max": [x_size, y_size, 6.28],
            "state_min": [0.0, 0.0, 0.0],
        },
    }
    
    robot_params = RobotParams(
        action_dim=2,
        state_dim=3,
        action_max=[5.0, 4.0],
        action_min=[0.5, -4.0],
        state_max=[x_size, y_size, 6.28],
        state_min=[0.0, 0.0, 0.0],
        name="dubins2d",
        dt=dt,
    )
    
    env = SmokeEnv(cfg=env_cfg, robot_params=robot_params)
    
    mppi_params = MPPICtrlParams(num_samples=120, horizon=14, lambda_=1.2)
    controller = MultiDualGuardCBFCtrl(
        num_agents=num_agents,
        robot_params=robot_params,
        robot_type="dubins2d",
        goal_thresh=goal_radius,
        device=device,
        mppi_params=mppi_params,
        dt=dt,
        r_sense=r_sense,
        d_safe=d_safe,
        k1=1.5,
        k2=1.5,
    )
    goals_dict = {f"agent_{i}": np.array(goal_locations[i]) for i in range(num_agents)}
    controller.set_goals(goals_dict)

    # 4. Setup Grid and Coordinates
    global_sensor_params = GlobalSensorParams(
        world_x_size=x_size,
        world_y_size=y_size,
        density_reading_per_unit_length=5.0
    )
    global_sensor = GlobalSensor(global_sensor_params)
    coords_global = global_sensor.grid_pairs_positions
    H_grid, W_grid = SmokeForecastWrapper._infer_grid_shape(coords_global)
    print(f"Global Grid: {H_grid} x {W_grid}")
    
    dx_cell = x_size / W_grid
    dy_cell = y_size / H_grid

    # Run episodes
    for ep in range(args.episodes):
        print(f"\n--- Starting Episode {ep + 1}/{args.episodes} ---")
        
        initial_state_dict = {}
        for i in range(num_agents):
            initial_state_dict[f"agent_{i}"] = {
                "location": np.array(initial_positions[i], dtype=np.float32),
                "angle": np.array([initial_headings[i]], dtype=np.float32),
            }
            
        obs, _ = env.reset(initial_state=initial_state_dict, seed=ep)
        
        # 15 Context rollouts: shape (15, h_ctx, H_grid, W_grid)
        h_ctx = fno_cfg.h_ctx
        rollout_contexts = torch.rand(15, h_ctx, H_grid, W_grid, device=device) * 0.1
        
        # Collage patch initial observations at t=0 for all h_ctx frames using square dense patch
        agent_locs = np.array([obs[f"agent_{i}"]["location"] for i in range(num_agents)])
        dx = np.abs(coords_global[:, np.newaxis, 0] - agent_locs[np.newaxis, :, 0])
        dy = np.abs(coords_global[:, np.newaxis, 1] - agent_locs[np.newaxis, :, 1])
        in_fov_flat = np.any((dx <= half_width) & (dy <= half_width), axis=1)
        in_fov_grid = in_fov_flat.reshape(H_grid, W_grid)

        gt_smoke_flat = obs["agent_0"]["smoke_density"].squeeze(-1)
        gt_smoke_grid = gt_smoke_flat.reshape(H_grid, W_grid)

        for h in range(h_ctx):
            rollout_contexts[:, h, in_fov_grid] = torch.tensor(gt_smoke_grid[in_fov_grid], dtype=torch.float32, device=device)

        # Plot setup (headless)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("FNO Multi-Agent Persistent Prediction Dashboard", fontsize=14, fontweight="bold")
        
        im_gt = None
        im_mean = None
        im_aleatoric = None
        im_epistemic = None
        
        agent_paths = {f"agent_{i}": [] for i in range(num_agents)}
        gif_frames = []
        
        finished = False
        t = 0
        
        while not finished and t < max_steps:
            # A. Autoregressive forecast of the planning horizon (15 steps)
            preds = forecast_horizon(
                model=model,
                seed_frames=rollout_contexts,
                t_start=t,
                horizon=fno_cfg.h_pred * 3,  # 15 steps
                seq_len_ref=fno_cfg.seq_len_ref,
                device=device,
                mode="mean"
            )
            
            # A2. Forecast of the ensemble mean context to get the predicted aleatoric uncertainty (sigma_PFNO)
            mean_context = rollout_contexts.mean(dim=0, keepdim=True)  # (1, h_ctx, H_grid, W_grid)
            mean_preds = forecast_horizon(
                model=model,
                seed_frames=mean_context,
                t_start=t,
                horizon=fno_cfg.h_pred * 3,
                seq_len_ref=fno_cfg.seq_len_ref,
                device=device,
                mode="mean"
            )
            
            # B. Build uncertainty maps and cost maps
            maps_deque = deque(maxlen=mppi_params.horizon)
            
            # Generate FOV mask for the current step using square dense patch
            agent_locs = np.array([obs[f"agent_{i}"]["location"] for i in range(num_agents)])
            dx = np.abs(coords_global[:, np.newaxis, 0] - agent_locs[np.newaxis, :, 0])
            dy = np.abs(coords_global[:, np.newaxis, 1] - agent_locs[np.newaxis, :, 1])
            in_fov_flat = np.any((dx <= half_width) & (dy <= half_width), axis=1)
            in_fov = in_fov_flat.reshape(H_grid, W_grid)
                    
            # Extract maps for visualization (at the first future prediction step h=0)
            mean_vis = preds[0]['sample'].mean(dim=0).cpu().numpy()
            std_aleatoric_vis = mean_preds[0]['std'].cpu().numpy().squeeze(0)
            std_epistemic_vis = preds[0]['sample'].std(dim=0).cpu().numpy()
            
            # Combine uncertainties and build risk deque
            for h in range(mppi_params.horizon):
                pred_step = min(h, len(preds) - 1)
                mean_h = preds[pred_step]['sample'].mean(dim=0).cpu().numpy()
                std_aleatoric_h = mean_preds[pred_step]['std'].cpu().numpy().squeeze(0)
                std_epistemic_h = preds[pred_step]['sample'].std(dim=0).cpu().numpy()
                
                # Decoupled cost formulation: Mean + alpha * std_aleatoric - beta * std_epistemic
                cost_h = mean_h + cost_alpha * std_aleatoric_h - cost_beta * std_epistemic_h
                maps_deque.append((coords_global, cost_h.ravel()))
                
            # C. Plan & Command Multi-Agents
            controller.set_maps(maps_deque)
            commands_dict = controller.get_commands(obs)
            step_actions = {k: cmd.detach().cpu().numpy() for k, cmd in commands_dict.items()}
            
            # D. Simulation Step
            next_obs, reward, terminated, truncated, _ = env.step(step_actions)
            
            # E. Update Context and collage-patch observations
            pred_next_frame = preds[0]["sample"]  # (15, H_grid, W_grid)
            rollout_contexts = torch.cat([rollout_contexts[:, 1:], pred_next_frame.unsqueeze(1)], dim=1)
            
            # Patch new observations into the latest frame using square dense patch
            agent_locs = np.array([next_obs[f"agent_{i}"]["location"] for i in range(num_agents)])
            dx = np.abs(coords_global[:, np.newaxis, 0] - agent_locs[np.newaxis, :, 0])
            dy = np.abs(coords_global[:, np.newaxis, 1] - agent_locs[np.newaxis, :, 1])
            in_fov_flat = np.any((dx <= half_width) & (dy <= half_width), axis=1)
            in_fov_grid = in_fov_flat.reshape(H_grid, W_grid)

            gt_smoke_flat = next_obs["agent_0"]["smoke_density"].squeeze(-1)
            gt_smoke_grid = gt_smoke_flat.reshape(H_grid, W_grid)

            rollout_contexts[:, -1, in_fov_grid] = torch.tensor(gt_smoke_grid[in_fov_grid], dtype=torch.float32, device=device)
            rollout_contexts = torch.clamp(rollout_contexts, 0.0, 1.0)
            
            # Track positions
            for i in range(num_agents):
                agent_paths[f"agent_{i}"].append(obs[f"agent_{i}"]["location"].copy())
                
            # F. Visualization
            if t % 2 == 0:
                # Get Ground Truth
                gt_smoke_flat = env.smoke_simulator.get_smoke_density(coords_global)
                gt_smoke_map = gt_smoke_flat.reshape(H_grid, W_grid)
                
                # Clear subplots
                for ax in axes.ravel():
                    ax.cla()
                    
                extent = [0, x_size, 0, y_size]
                
                # 1. Ground Truth
                ax0 = axes[0, 0]
                sc0 = ax0.imshow(gt_smoke_map, origin="lower", extent=extent, cmap="Greys", vmin=0.0, vmax=1.0)
                ax0.set_title(f"Ground Truth + Drones (Step {t})")
                ax0.set_xlabel("X (m)")
                ax0.set_ylabel("Y (m)")
                
                colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
                for i in range(num_agents):
                    pos = obs[f"agent_{i}"]["location"]
                    goal = goal_locations[i]
                    ax0.scatter(goal[0], goal[1], color=colors_list[i], marker="*", s=150, zorder=5)
                    
                    # Trajectory path
                    path_pts = np.array(agent_paths[f"agent_{i}"])
                    if len(path_pts) > 0:
                        ax0.plot(path_pts[:, 0], path_pts[:, 1], color=colors_list[i], linewidth=1.5, alpha=0.8)
                    ax0.scatter(pos[0], pos[1], color=colors_list[i], s=80, edgecolors='black', zorder=6)
                    
                    # FOV boundary (square patch of size 2 * half_width)
                    rect = plt.Rectangle((pos[0] - half_width, pos[1] - half_width), 2 * half_width, 2 * half_width, fill=False, edgecolor=colors_list[i], linestyle="--", linewidth=1.0, alpha=0.6)
                    ax0.add_patch(rect)
                    
                # 2. Predicted Mean
                ax1 = axes[0, 1]
                sc1 = ax1.imshow(mean_vis, origin="lower", extent=extent, cmap="rainbow", vmin=0.0, vmax=1.0)
                ax1.set_title("Predicted Global Mean Smoke Map")
                ax1.set_xlabel("X (m)")
                ax1.set_ylabel("Y (m)")
                
                # 3. Aleatoric Uncertainty
                ax2 = axes[1, 0]
                sc2 = ax2.imshow(std_aleatoric_vis, origin="lower", extent=extent, cmap="hot", vmin=0.0, vmax=0.5)
                ax2.set_title("Aleatoric Uncertainty (Model predicted Std)")
                ax2.set_xlabel("X (m)")
                ax2.set_ylabel("Y (m)")
                
                # 4. Epistemic Uncertainty
                ax3 = axes[1, 1]
                # We mask the epistemic uncertainty inside FOV to emphasize it is 0 there
                epistemic_masked = np.where(in_fov, 0.0, std_epistemic_vis)
                sc3 = ax3.imshow(epistemic_masked, origin="lower", extent=extent, cmap="coolwarm", vmin=0.0, vmax=0.5)
                ax3.set_title("Epistemic Uncertainty (Rollout Disagreement)")
                ax3.set_xlabel("X (m)")
                ax3.set_ylabel("Y (m)")
                
                if im_gt is None:
                    fig.colorbar(sc0, ax=ax0, fraction=0.046, pad=0.04)
                    fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
                    fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
                    fig.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
                    im_gt, im_mean, im_aleatoric, im_epistemic = sc0, sc1, sc2, sc3
                    
                plt.tight_layout()
                
                if save_gif:
                    import io
                    import imageio.v2 as imageio
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    gif_frames.append(imageio.imread(buf))

            obs = next_obs
            
            # Check termination
            all_term = all(terminated.values()) if isinstance(terminated, dict) else terminated
            any_trunc = any(truncated.values()) if isinstance(truncated, dict) else truncated
            if all_term or any_trunc:
                finished = True
                
            t += 1
            
        print(f"Episode completed in {t} steps.")
        
        if save_gif and len(gif_frames) > 0:
            os.makedirs("data/videos", exist_ok=True)
            gif_path = "data/videos/run_persistent_prediction.gif"
            print(f"Saving visualization GIF to {gif_path}...")
            imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
            print("GIF saved.")
            
        plt.close(fig)


if __name__ == "__main__":
    main()
