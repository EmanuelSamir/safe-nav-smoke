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
from src.models.model_free.rnp_residual import RNPResidual
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

def autoregressive_rollout(model, context_obs, horizon, H_grid, W_grid, x_size, y_size, res_val, model_type='rnp', num_samples=1):
    start_time = time.time()
    
    B = num_samples
    device = context_obs.xs.device
    
    ctx = Obs(
        xs=context_obs.xs.expand(B, -1, -1, -1),
        ys=context_obs.ys.expand(B, -1, -1, -1),
        values=context_obs.values.expand(B, -1, -1, -1),
        mask=context_obs.mask,
        ts=context_obs.ts
    )
    
    state = model.init_state(batch_size=B, device=device)
    
    T_ctx = ctx.xs.shape[1]
    
    for t in range(T_ctx):
        r_step = model.encoder(slice_obs(ctx, t, t+1))
        r_step_sq = r_step.squeeze(1)
        _, state = model.forecaster(r_step_sq, state)
        
    query_obs = get_dense_query_obs(B, device, x_size, y_size, res_val, H_grid, W_grid)
    
    current_input = slice_obs(ctx, T_ctx-1, T_ctx) 
    
    trajectories = [] 
    
    steps_generated = 0
    
    while steps_generated < horizon:
        r_step = model.encoder(current_input)
        r_step_sq = r_step.squeeze(1)
        
        r_next, state = model.forecaster(r_step_sq, state)
        r_next_expanded = r_next.unsqueeze(1)
        
        if model_type == 'multistep':
            dists = model.decoder(r_next_expanded, query_obs)
            chunk_samples = []
            for dist in dists:
                if num_samples > 1:
                    s = dist.sample()
                else:
                    s = dist.mean
                chunk_samples.append(s)
                
            trajectories.extend(chunk_samples)
            steps_generated += len(chunk_samples)
            
            if steps_generated < horizon:
                for pred_vals in chunk_samples:
                     obs_in = Obs(xs=query_obs.xs, ys=query_obs.ys, values=pred_vals)
                     r_s = model.encoder(obs_in)
                     r_s_sq = r_s.squeeze(1)
                     _, state = model.forecaster(r_s_sq, state)
                
                current_input = Obs(xs=query_obs.xs, ys=query_obs.ys, values=chunk_samples[-1])

        else:
            if 'residual' in model_type:
                dists = model.decoder(r_next_expanded, r_step, query_obs, current_input)
                dist_tp1 = dists[1]
                
                if model_type == 'residual_delta':
                    if steps_generated == 0:
                        if num_samples > 1:
                            s = dist_tp1.sample()
                        else:
                            s = dist_tp1.mean
                    else:
                        comps = getattr(dist_tp1, 'components', None)
                        if comps is None:
                            raise RuntimeError("Residual model did not return components!")
                            
                        delta_mu = comps['delta_mu']
                        delta_sigma = comps['delta_sigma']
                        
                        dist_delta = torch.distributions.Normal(delta_mu, delta_sigma)
                        
                        if num_samples > 1:
                            delta = dist_delta.sample()
                        else:
                            delta = delta_mu
                            
                        s = current_input.values + delta
                else:
                    if num_samples > 1:
                        s = dist_tp1.sample()
                    else:
                        s = dist_tp1.mean
            else:
                dist = model.decoder(r_next_expanded, query_obs)
                if num_samples > 1:
                    s = dist.sample()
                else:
                    s = dist.mean
            
            trajectories.append(s)
            steps_generated += 1
            
            current_input = Obs(xs=query_obs.xs, ys=query_obs.ys, values=s)
            
    trajectories = trajectories[:horizon]
    
    stack = torch.stack([t.squeeze(1) for t in trajectories], dim=1) 
    
    # Reshape to (num_samples, horizon, H, W)
    stack_imgs = stack.squeeze(-1).view(num_samples, horizon, H_grid, W_grid).detach().cpu().numpy()
    
    end_time = time.time()
    latency_ms = ((end_time - start_time) * 1000.0) / num_samples
    
    return stack_imgs, latency_ms

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
    parser.add_argument('--ckpt_res', type=str, required=True)
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
    models['residual'], configs['residual'] = load_model('Residual', RNPResidual, args.ckpt_res)
    models['multistep'], configs['multistep'] = load_model('Multistep', RNPMultistep, args.ckpt_multi)

    smoke_data, x_size, y_size, res_val = load_data(args.data_path)
    _, H_grid, W_grid = smoke_data[0].shape
    
    metric_keys = ['rnp', 'residual', 'residual_delta', 'multistep']
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
            
            for m_name in metric_keys:
                if m_name == 'multistep':
                    model = models['multistep']
                    model_type = 'multistep'
                elif m_name == 'residual':
                    model = models['residual']
                    model_type = 'residual'
                elif m_name == 'residual_delta':
                    model = models['residual']
                    model_type = 'residual_delta'
                else:
                    model = models['rnp']
                    model_type = 'rnp'
                
                preds_samples, lat = autoregressive_rollout(
                    model, ctx_obs, args.horizon, H_grid, W_grid, x_size, y_size, res_val, 
                    model_type=model_type, num_samples=args.num_samples
                )
                
                # Predictions are of shape (num_samples, horizon, H, W)
                save_dict[f't_{t_current}_{m_name}_preds'] = preds_samples
                save_dict[f't_{t_current}_{m_name}_latency'] = lat

        out_file = out_dir / f"ep_{ep_idx}_rollouts.npz"
        np.savez_compressed(out_file, **save_dict)
        
    print(f"All rollouts saved in {out_dir}")

if __name__ == "__main__":
    main()
