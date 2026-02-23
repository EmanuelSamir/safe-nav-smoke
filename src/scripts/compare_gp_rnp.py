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
from src.models.gaussian_process import GaussianProcess, OnlineKernel

log = logging.getLogger(__name__)

class GPWrapper:
    def __init__(self, dt):
        # We initialize the GP with the online true pattern. 
        self.gp = GaussianProcess(online=True, online_kernel=OnlineKernel.Matern)
        self.dt = dt
        self.t_current = 0
        
    def reset(self):
        self.gp = GaussianProcess(online=True, online_kernel=OnlineKernel.Matern)
        self.t_current = 0
        
    def update(self, xs_np, ys_np, vals_np, t_idx):
        self.t_current = t_idx
        # GP uses real scales since its kernel lengthscale is configured for it (~2.0)
        ts_np = np.full_like(xs_np, float(t_idx * self.dt))
        # Concatenate features into X buffer: (P, 3) where columns are [x, y, t]
        X = np.stack([xs_np.flatten(), ys_np.flatten(), ts_np.flatten()], axis=1)
        self.gp.track_data(X, vals_np.flatten())
        self.gp.update()
        
    def predict(self, xs_query_np, ys_query_np, t_idx):
        ts_query_np = np.full_like(xs_query_np, float(t_idx * self.dt))
        X_query = np.stack([xs_query_np.flatten(), ys_query_np.flatten(), ts_query_np.flatten()], axis=1)
        mean, std = self.gp.predict(X_query)
        return mean, std


def get_context_obs(ep_data, t_end, device, x_size, res_val, y_size, context_points=200):
    """ Builds Context Points for RNP and GP. Limits points heavily per frame to avoid GP OOM. """
    frames = ep_data[:t_end+1] 
    T, H_grid, W_grid = frames.shape
    
    # We sample 'context_points' randomly over the map per each time step
    y_idxs = np.random.randint(0, H_grid, size=(T, context_points))
    x_idxs = np.random.randint(0, W_grid, size=(T, context_points))
    
    t_range = np.arange(T)[:, None]
    vals = frames[t_range, y_idxs, x_idxs] 
    
    # Absolute Physical Coordinates
    xs_real = x_idxs * res_val
    ys_real = y_idxs * res_val
    
    # Normalized for RNP
    xs_norm = xs_real / x_size
    ys_norm = ys_real / y_size
    
    xs_t = torch.from_numpy(xs_norm).float().unsqueeze(0).unsqueeze(-1).to(device)
    ys_t = torch.from_numpy(ys_norm).float().unsqueeze(0).unsqueeze(-1).to(device)
    vals_t = torch.from_numpy(vals).float().unsqueeze(0).unsqueeze(-1).to(device)
    
    # Export real numpy coords for GP wrapper
    numpy_data = {
        'xs': xs_real, # (T, P)
        'ys': ys_real,
        'vals': vals
    }
    
    return Obs(xs=xs_t, ys=ys_t, values=vals_t), numpy_data


def get_dense_query_obs(B, device, x_size, y_size, res_val, H_grid, W_grid, downsample_factor=2):
    """ Build targets. Allows downsampling on evaluating since GP is very slow per dense point """
    y_range = np.arange(0, H_grid, downsample_factor)
    x_range = np.arange(0, W_grid, downsample_factor)
    
    xs_idx, ys_idx = np.meshgrid(x_range, y_range)
    xs_flat = xs_idx.flatten()
    ys_flat = ys_idx.flatten()
    
    xs_real = xs_flat * res_val
    ys_real = ys_flat * res_val
    
    xs_norm = xs_real / x_size
    ys_norm = ys_real / y_size
    
    xs_t = torch.from_numpy(xs_norm).float().to(device)
    ys_t = torch.from_numpy(ys_norm).float().to(device)
    
    xs_t = xs_t.view(1, 1, -1, 1).expand(B, -1, -1, -1)
    ys_t = ys_t.view(1, 1, -1, 1).expand(B, -1, -1, -1)
    
    P = xs_t.shape[2]
    
    query_obs = Obs(
        xs=xs_t,
        ys=ys_t,
        values=torch.zeros(B, 1, P, 1, device=device),
        mask=None,
        ts=None
    )
    
    numpy_query = {
        'xs': xs_real,
        'ys': ys_real,
        'shape': (len(y_range), len(x_range))
    }
    
    return query_obs, numpy_query


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
        dt = float(raw_data.get('dt', 0.1))
        return smoke_data, x_size, y_size, res_val, dt
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_rnp', type=str, required=True)
    parser.add_argument('--ckpt_multi', type=str, required=True)
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='saved_comparisons')
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--num_episodes', type=int, default=10, help='Max episodes to evaluate')
    parser.add_argument('--stride', type=int, default=10, help='Stride for rolling forecast')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of predictive samples to save for RNP')
    parser.add_argument('--downsample_eval', type=int, default=2, help='Downsample scale factor for dense query grid (Speeding up GP)')
    parser.add_argument('--context_points', type=int, default=100, help='Points per frame to supply model (to avoid hanging the GP M2 matrix)')

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
    models['rnp'], _ = load_model('RNP', RNP, args.ckpt_rnp)
    models['multistep'], _ = load_model('Multistep', RNPMultistep, args.ckpt_multi)

    smoke_data, x_size, y_size, res_val, dt = load_data(args.data_path)
    _, H_grid, W_grid = smoke_data[0].shape
    
    metric_keys = ['gp', 'rnp', 'multistep']
    num_eval_episodes = min(args.num_episodes, len(smoke_data))
    indices = np.linspace(0, len(smoke_data)-1, num_eval_episodes, dtype=int)
    
    for ep_idx in tqdm(indices, desc="Episodes"):
        ep_data = smoke_data[ep_idx]
        T_total = ep_data.shape[0]
        
        # Start looking context fairly early
        t_start = 5
        # Do not roll for the entire trajectory because GP memory fills up and inference scales cubically
        # Only take a few strides at the start
        t_end = min(20, T_total - args.horizon - 1) 
        
        if t_end <= t_start: 
            print(f"Skipping episode {ep_idx} (Too short)")
            continue
            
        stride = args.stride
        time_steps = list(range(t_start, t_end, stride))
        
        save_dict = {
            'ep_idx': ep_idx,
            'time_steps': np.array(time_steps),
            'gt_data': ep_data, # Ensure we save the original dense gt data
        }
        
        for t_current in tqdm(time_steps, desc=f"Ep {ep_idx} (Strides)", leave=False):
            # Fetch data formats
            ctx_obs, numpy_ctx = get_context_obs(ep_data, t_current, device, x_size, res_val, y_size, context_points=args.context_points)
            
            gt_frames_horizon = ep_data[t_current+1 : t_current+1+args.horizon]
            save_dict[f't_{t_current}_gt_horizon'] = gt_frames_horizon
            
            # Obtain the downsampled grid targets to speed up GP inference queries mapping the horizon
            query_obs, numpy_query = get_dense_query_obs(1, device, x_size, y_size, res_val, H_grid, W_grid, downsample_factor=args.downsample_eval)
            H_out, W_out = numpy_query['shape']
            save_dict['eval_shape'] = np.array([H_out, W_out])
            
            # --- 1) EVAL GP ---
            print(f"\n  Evaluating GP at t={t_current} (Updating step-by-step)...")
            gp = GPWrapper(dt=dt)
            t0_gp = time.time()
            # Feed context observations one chronologically
            for t_step in range(t_current + 1):
                xs_t = numpy_ctx['xs'][t_step]
                ys_t = numpy_ctx['ys'][t_step]
                vals_t = numpy_ctx['vals'][t_step]
                gp.update(xs_t, ys_t, vals_t, t_idx=t_step)
            
            # Predict Horizon frames
            gp_mean_imgs = []
            gp_std_imgs = []
            for h in range(1, args.horizon + 1):
                m, s = gp.predict(numpy_query['xs'], numpy_query['ys'], t_idx=t_current+h)
                gp_mean_imgs.append(m.reshape(H_out, W_out))
                gp_std_imgs.append(s.reshape(H_out, W_out))
                
            latency_gp = ((time.time() - t0_gp) * 1000.0) 
            save_dict[f't_{t_current}_gp_mean'] = np.stack(gp_mean_imgs, axis=0)
            save_dict[f't_{t_current}_gp_std'] = np.stack(gp_std_imgs, axis=0)
            save_dict[f't_{t_current}_gp_latency'] = latency_gp
            
            # --- 2) EVAL RNP MODELS ---
            for m_name in ['rnp', 'multistep']:
                print(f"  Evaluating {m_name} at t={t_current}...")
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
                    
                    sample_imgs.append(s_vals.view(args.num_samples, H_out, W_out).detach().cpu().numpy())
                    mean_imgs.append(m_vals.view(args.num_samples, H_out, W_out).detach().cpu().numpy())
                    std_imgs.append(std_vals.view(args.num_samples, H_out, W_out).detach().cpu().numpy())
                
                sample_imgs = np.stack(sample_imgs, axis=1) # (N_Samples, Horizon, H, W)
                mean_imgs = np.stack(mean_imgs, axis=1)
                std_imgs = np.stack(std_imgs, axis=1)
                
                # Squeeze the first channel just like gp
                save_dict[f't_{t_current}_{m_name}_sample'] = sample_imgs
                save_dict[f't_{t_current}_{m_name}_mean'] = mean_imgs
                save_dict[f't_{t_current}_{m_name}_std'] = std_imgs
                save_dict[f't_{t_current}_{m_name}_latency'] = latency_ms

        out_file = out_dir / f"ep_{ep_idx}_gp_comparison.npz"
        np.savez_compressed(out_file, **save_dict)
        
    print(f"\nAll comparison grids generated successfully in {out_dir}")

if __name__ == "__main__":
    main()
