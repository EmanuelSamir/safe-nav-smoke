import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import time
import argparse

import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.getcwd())

from src.models.model_free.rnp import RNP, RNPConfig
from src.models.model_free.rnp_multistep import RNPMultistep
from src.models.shared.observations import Obs, slice_obs

log = logging.getLogger(__name__)

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    try:
        raw_data = np.load(data_path)
        if 'smoke_data' not in raw_data:
             raise KeyError("smoke_data not found in npz")
        smoke_data = raw_data['smoke_data'] # (E, T, H, W)
        x_size = float(raw_data.get('x_size', 50.0))
        y_size = float(raw_data.get('y_size', 50.0))
        res_val = float(raw_data.get('resolution', 1.0))
        return smoke_data, x_size, y_size, res_val
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def get_dense_obs(ep_data, t_step, device, x_size, res_val, y_size, context=True):
    frame = ep_data[t_step:t_step+1] # (1, H, W)
    T, H_grid, W_grid = frame.shape

    y_idxs = np.arange(H_grid)
    x_idxs = np.arange(W_grid)
    y_idxs, x_idxs = np.meshgrid(y_idxs, x_idxs, indexing='ij')
    y_idxs = y_idxs.flatten()
    x_idxs = x_idxs.flatten()
    
    t_range = np.zeros(len(x_idxs), dtype=int) # Since first dim is 1 (batch size)
    if context:
        vals = frame[t_range, y_idxs, x_idxs] 
    else:
        vals = np.zeros_like(x_idxs)
    
    xs_norm = 2.0 * (x_idxs * res_val) / (x_size) - 1.0
    ys_norm = 2.0 * (y_idxs * res_val) / (y_size) - 1.0
    
    # xs shape to be (B=1, T=1, P, 1)
    xs = torch.from_numpy(xs_norm).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device)
    ys = torch.from_numpy(ys_norm).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device)
    vals_t = torch.from_numpy(vals).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device)
    
    return Obs(xs=xs, ys=ys, values=vals_t)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='saved_rollouts')
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=None)
    parser.add_argument('--stride', type=int, default=3, help='Stride for rolling forecast')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of predictive samples to save')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_model(name, cls, path):
        print(f"Loading {name} from {path}...")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = ckpt['hyper_parameters']
        if isinstance(cfg_dict, dict):
            cfg = RNPConfig(**cfg_dict)
        else:
            cfg = cfg_dict
            
        if cls == RNPMultistep:
             model = cls(cfg, forecast_horizon=5)
        else:
             model = cls(cfg)
             
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        return model, cfg

    if args.model_type == 'rnp':
        model, cfg = load_model('RNP', RNP, args.ckpt)
    elif args.model_type == 'multistep':
        model, cfg = load_model('Multistep', RNPMultistep, args.ckpt)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    smoke_data, x_size, y_size, res_val = load_data(args.data_path)
    _, H_grid, W_grid = smoke_data[0].shape
    
    if args.num_episodes:
        min_ep = min((args.num_episodes, len(smoke_data)))
        indices = np.random.choice(len(smoke_data), min_ep, replace=False)
    else:
        indices = range(len(smoke_data))
    
    for ep_idx in tqdm(indices, desc="Episodes"):
        ep_data = smoke_data[ep_idx]
        T_total = ep_data.shape[0]
        
        t_start = 10
        t_end = T_total - args.horizon - 1
        
        if t_end <= t_start: 
            print(f"Skipping episode {ep_idx} (Too short)")
            continue
            
        stride = args.stride
        time_steps = list(range(t_start, t_end, stride))
        
        # Save array structure — no full gt_data to keep files small
        save_dict = {
            'ep_idx': ep_idx,
            'time_steps': np.array(time_steps),
        }
        
        # Initialize state for this episode
        running_state = model.init_state(1, device)

        # We step through time sequentially 
        for t_current in tqdm(range(t_end), desc=f"Ep {ep_idx} (Stepping)", leave=False):
            t0 = time.time()
            ctx_obs = get_dense_obs(ep_data, t_current, device, x_size, res_val, y_size, context=True)
            print(f"  [Time] Context: {time.time()-t0:.3f}s")
            
            # 1. Update running states with the real ground truth at t_current
            with torch.no_grad():
                # Target is None because we only want to update the recurrent state
                out = model(running_state, context_obs=ctx_obs, target_obs=None)
                running_state = out.state
                     
            # 2. If it's time to evaluate / save rollout based on stride
            if t_current >= t_start and (t_current - t_start) % stride == 0:
                gt_frames_horizon = ep_data[t_current+1 : t_current+1+args.horizon]
                save_dict[f't_{t_current}_gt_horizon'] = gt_frames_horizon.astype(np.float16)
                
                query_obs = get_dense_obs(ep_data, t_current, device, x_size, res_val, y_size, context=False)
                
                t0_model = time.time()
                with torch.no_grad():
                    def clone_state(state):
                        if state is None: return None
                        # state is List[Tuple[Tensor, Tensor]], one (h, c) per ConvLSTM layer
                        return [(layer[0].clone(), layer[1].clone()) for layer in state]
                    
                    state_clone = clone_state(running_state)
                    
                    preds = model.autoregressive_forecast(
                        state=state_clone,
                        context_obs=ctx_obs,
                        target_obs=query_obs,
                        horizon=args.horizon,
                        num_samples=args.num_samples
                    )
                    
                latency_ms = ((time.time() - t0_model) * 1000.0) / args.num_samples
                
                # Save samples + ensemble mean/std, all in float16
                sample_imgs, mean_imgs, std_imgs = [], [], []
                for step_preds in preds:
                    # After delayed batch expansion: (num_samples, 1, P, 1)
                    s_vals   = step_preds['sample'].values.squeeze(-1).squeeze(1)  # (num_samples, P)
                    m_vals   = step_preds['mean'].values.squeeze(-1).squeeze(1)   # (num_samples, P) ensemble mean
                    std_vals = step_preds['std'].values.squeeze(-1).squeeze(1)   # (num_samples, P) ensemble std
                    sample_imgs.append(s_vals.view(args.num_samples, H_grid, W_grid).detach().cpu().to(torch.float16).numpy())
                    mean_imgs.append(m_vals.view(H_grid, W_grid).detach().cpu().to(torch.float16).numpy())
                    std_imgs.append(std_vals.view(H_grid, W_grid).detach().cpu().to(torch.float16).numpy())
                
                # (horizon, num_samples, H, W) for samples; (horizon, H, W) for mean/std
                sample_imgs = np.stack(sample_imgs, axis=0)
                mean_imgs   = np.stack(mean_imgs, axis=0)
                std_imgs    = np.stack(std_imgs, axis=0)
                
                save_dict[f't_{t_current}_{args.model_type}_sample'] = sample_imgs
                save_dict[f't_{t_current}_{args.model_type}_mean'] = mean_imgs
                save_dict[f't_{t_current}_{args.model_type}_std'] = std_imgs
                save_dict[f't_{t_current}_{args.model_type}_latency'] = latency_ms                
        out_file = out_dir / f"ep_{ep_idx}_rollouts.npz"
        np.savez_compressed(out_file, **save_dict)
        
    print(f"All rollouts saved in {out_dir}")

if __name__ == "__main__":
    main()
