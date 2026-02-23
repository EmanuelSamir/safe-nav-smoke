import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import time
import argparse

# Add src to path
sys.path.append(os.getcwd())

from src.models.model_free.rnp import RNP, RNPConfig
from src.models.model_free.rnp_multistep import RNPMultistep
from src.models.shared.observations import Obs, slice_obs

log = logging.getLogger(__name__)

def get_dense_query_obs(B, device, x_size, y_size, res_val, H_grid, W_grid):
    xs_idx, ys_idx = np.meshgrid(np.arange(W_grid), np.arange(H_grid))
    xs_flat = xs_idx.flatten()
    ys_flat = ys_idx.flatten()
    
    xs_norm = (xs_flat * res_val) / x_size
    ys_norm = (ys_flat * res_val) / y_size
    
    xs_t = torch.from_numpy(xs_norm).float().to(device)
    ys_t = torch.from_numpy(ys_norm).float().to(device)
    
    xs_t = xs_t.view(1, 1, -1, 1).expand(B, -1, -1, -1)
    ys_t = ys_t.view(1, 1, -1, 1).expand(B, -1, -1, -1)
    
    P = xs_t.shape[2]
    
    return Obs(
        xs=xs_t,
        ys=ys_t,
        values=torch.zeros(B, 1, P, 1, device=device),
        mask=None,
        ts=None
    )


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

def get_context_obs(ep_data, t_end, device, x_size, res_val, y_size):
    frames = ep_data[:t_end+1] 
    T, H_grid, W_grid = frames.shape
    num_points = int(H_grid * W_grid * 0.5)
    
    y_idxs = np.random.randint(0, H_grid, size=(T, num_points))
    x_idxs = np.random.randint(0, W_grid, size=(T, num_points))
    
    t_range = np.arange(T)[:, None]
    vals = frames[t_range, y_idxs, x_idxs] 
    
    xs_norm = (x_idxs * res_val) / x_size
    ys_norm = (y_idxs * res_val) / y_size
    
    xs = torch.from_numpy(xs_norm).float().unsqueeze(0).unsqueeze(-1).to(device)
    ys = torch.from_numpy(ys_norm).float().unsqueeze(0).unsqueeze(-1).to(device)
    vals_t = torch.from_numpy(vals).float().unsqueeze(0).unsqueeze(-1).to(device)
    
    return Obs(xs=xs, ys=ys, values=vals_t)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_rnp', type=str, required=True)
    parser.add_argument('--ckpt_multi', type=str, required=True)
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='saved_rollouts')
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=10)
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

    models = {}
    configs = {}
    models['rnp'], configs['rnp'] = load_model('RNP', RNP, args.ckpt_rnp)
    models['multistep'], configs['multistep'] = load_model('Multistep', RNPMultistep, args.ckpt_multi)

    smoke_data, x_size, y_size, res_val = load_data(args.data_path)
    _, H_grid, W_grid = smoke_data[0].shape
    
    metric_keys = ['rnp', 'multistep']
    indices = np.linspace(0, len(smoke_data)-1, args.num_episodes, dtype=int)
    
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
        
        # Save array structure with gt_data and predictions
        save_dict = {
            'ep_idx': ep_idx,
            'time_steps': np.array(time_steps),
            'gt_data': ep_data,
        }
        
        for t_current in tqdm(time_steps, desc=f"Ep {ep_idx} (Stride {stride})", leave=False):
            t0 = time.time()
            ctx_obs = get_context_obs(ep_data, t_current, device, x_size, res_val, y_size)
            print(f"  [Time] Context: {time.time()-t0:.3f}s")
            
            gt_frames_horizon = ep_data[t_current+1 : t_current+1+args.horizon]
            save_dict[f't_{t_current}_gt_horizon'] = gt_frames_horizon
            
            query_obs = get_dense_query_obs(1, device, x_size, y_size, res_val, H_grid, W_grid)
            
            for m_name in metric_keys:
                model = models[m_name]
                
                t0_model = time.time()
                with torch.no_grad():
                    preds = model.autoregressive_forecast(
                        state=None,
                        context_obs=ctx_obs,
                        target_obs=query_obs,
                        horizon=args.horizon,
                        num_samples=args.num_samples
                    )
                latency_ms = ((time.time() - t0_model) * 1000.0) / args.num_samples
                
                sample_imgs, mean_imgs, std_imgs = [], [], []
                
                for step_preds in preds:
                    s_vals = step_preds['sample'].values.squeeze(1).squeeze(-1)
                    m_vals = step_preds['mean'].values.squeeze(1).squeeze(-1)
                    std_vals = step_preds['std'].values.squeeze(1).squeeze(-1)
                    
                    sample_imgs.append(s_vals.view(args.num_samples, H_grid, W_grid).detach().cpu().numpy())
                    mean_imgs.append(m_vals.view(args.num_samples, H_grid, W_grid).detach().cpu().numpy())
                    std_imgs.append(std_vals.view(args.num_samples, H_grid, W_grid).detach().cpu().numpy())
                
                sample_imgs = np.stack(sample_imgs, axis=1)
                mean_imgs = np.stack(mean_imgs, axis=1)
                std_imgs = np.stack(std_imgs, axis=1)
                
                save_dict[f't_{t_current}_{m_name}_sample'] = sample_imgs
                save_dict[f't_{t_current}_{m_name}_mean'] = mean_imgs
                save_dict[f't_{t_current}_{m_name}_std'] = std_imgs
                save_dict[f't_{t_current}_{m_name}_latency'] = latency_ms

        out_file = out_dir / f"ep_{ep_idx}_rollouts.npz"
        np.savez_compressed(out_file, **save_dict)
        
    print(f"All rollouts saved in {out_dir}")

if __name__ == "__main__":
    main()
