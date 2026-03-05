import sys
import os
import math
import time
sys.path.append(os.getcwd())

import random
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.pfno import PFNO, PFNOConfig
from src.models.shared.datasets import SequentialDataset, dense_sequential_collate_fn
from src.models.shared.schedulers import LinearBetaScheduler

log = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, loss, cfg_dict, path):
    torch.save({
        'epoch':                 epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
        'loss':                  loss,
        'hyper_parameters':      cfg_dict,
    }, path)

def make_times(t_offset: int, h_ctx: int, seq_len: int,
               device, B: int) -> torch.Tensor:
    t = torch.arange(t_offset, t_offset + h_ctx, device=device, dtype=torch.float32)
    t = t / max(seq_len - 1, 1)
    return t.unsqueeze(0).expand(B, -1)

def beta_nll_loss(dist, gt, beta=0.5):
    """
    Computes Beta-NLL loss.
    dist: Normal distribution object
    gt: Ground truth tensor
    Return mean Beta-NLL.
    """
    var = dist.variance
    
    # Standard NLL: (mu - gt)^2 / (2*var) + 0.5 * log(var)
    # Scaled by (var^beta) -- actually pytorch var**beta
    nll = 0.5 * (((dist.mean - gt) ** 2) / var + torch.log(var))
    
    # We detach the variance multiplier to prevent gradients through beta scaling
    # Or keep it if we want it to affect variance learning. Usually beta-NLL explicitly uses the stopped gradient for weighting.
    weight = (var.detach() ** beta)
    
    loss = nll * weight
    return loss.mean()

def log_vis(model, loader, cfg, H, W, device, writer, epoch,
            rollout_steps=(1, 5, 10, 15)):
    model.eval()
    ctx_obs, trg_obs, _ = next(iter(loader))
    ctx_obs = ctx_obs.to(device)

    T      = ctx_obs.xs.shape[1]
    h_ctx  = cfg.training.model.h_ctx
    seq_len = cfg.training.data.sequence_length
    cases  = min(2, ctx_obs.xs.shape[0])
    max_step = max(rollout_steps)

    n_cols = len(rollout_steps) * 3
    fig, axes = plt.subplots(cases, n_cols, figsize=(5 * n_cols, 4 * cases))
    if cases == 1:
        axes = axes[None, :]

    frames_all = ctx_obs.values[:, :, :, 0].view(ctx_obs.xs.shape[0], T, H, W)

    with torch.no_grad():
        for b_idx in range(cases):
            # seed shape needs to be (1, h_ctx, H, W) for standard, or (1, 2, h_ctx, H, W) for pfno
            # fno3d autoregressive_forecast handles adding 0s stddev if seeded with 4d
            seed  = frames_all[b_idx:b_idx+1, :h_ctx]    
            preds = model.autoregressive_forecast(
                seed, seed_t_start=0, horizon=max_step, num_samples=1)

            for j, step in enumerate(rollout_steps):
                idx     = step - 1
                if idx >= len(preds): break
                mu_img  = preds[idx]['mean'][0].astype('float32')
                std_img = preds[idx]['std'][0].astype('float32')

                abs_t   = h_ctx + idx
                gt_img  = (frames_all[b_idx, abs_t].cpu().numpy()
                           if abs_t < T else np.zeros((H, W), dtype=np.float32))
                err     = np.abs(gt_img - mu_img).mean()
                col     = j * 3

                ax = axes[b_idx, col]
                sc = ax.imshow(gt_img, vmin=0, vmax=1, cmap='rainbow', origin='lower')
                ax.set_title(f"GT +{step} b={b_idx}")
                plt.colorbar(sc, ax=ax)

                ax = axes[b_idx, col+1]
                sc2 = ax.imshow(mu_img, vmin=0, vmax=1, cmap='rainbow', origin='lower')
                ax.set_title(f"μ +{step}  err={err:.3f}")
                plt.colorbar(sc2, ax=ax)

                ax = axes[b_idx, col+2]
                sc3 = ax.imshow(std_img, vmin=0, vmax=1, cmap='hot', origin='lower')
                ax.set_title(f"σ +{step}  μσ={std_img.mean():.3f}")
                plt.colorbar(sc3, ax=ax)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_t = torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1)
    writer.add_image("Val/Rollout", img_t, epoch)
    plt.close()

@hydra.main(version_base=None,
            config_path="../../configs/training",
            config_name="pfno_train")
def train(cfg: DictConfig):
    print(f"Training PFNO-3D — {cfg.training.experiment_name}")
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    log_dir    = os.path.join(output_dir, "logs")
    ckpt_dir   = os.path.join(output_dir, "checkpoints")
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    writer = SummaryWriter(log_dir=log_dir)

    # Data
    try:
        root_dir = Path(hydra.utils.get_original_cwd())
    except Exception:
        root_dir = Path(os.getcwd())

    data_path = root_dir / cfg.training.data.data_path
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    h_ctx   = cfg.training.model.h_ctx
    m_pred  = cfg.training.model.m_pred
    seq_len = cfg.training.data.sequence_length
    max_ep  = cfg.training.data.get("max_samples", None)

    curr_cfg = cfg.training.scheduler
    horizon_start = curr_cfg.get("horizon_start_step", 1)
    horizon_end = curr_cfg.get("horizon_end_step", 1)
    max_forecast_horizon = horizon_end * m_pred

    assert seq_len > h_ctx + max_forecast_horizon, \
        f"sequence_length ({seq_len}) must be > h_ctx+max_horiz ({h_ctx+max_forecast_horizon})"

    train_ds = SequentialDataset(
        data_path=str(data_path), sequence_length=seq_len,
        forecast_horizon=max_forecast_horizon, mode='train',
        train_split=cfg.training.data.train_split,
        max_episodes=max_ep, dense=True)
    val_ds = SequentialDataset(
        data_path=str(data_path), sequence_length=seq_len,
        forecast_horizon=max_forecast_horizon, mode='val',
        train_split=cfg.training.data.train_split,
        max_episodes=max_ep, dense=True)

    H, W = train_ds.H, train_ds.W
    print(f"Grid {H}×{W}  h_ctx={h_ctx}  m_pred={m_pred}  "
          f"max_H={max_forecast_horizon}  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.data.batch_size,
        shuffle=True, collate_fn=dense_sequential_collate_fn,
        num_workers=cfg.training.data.num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.data.batch_size,
        shuffle=False, collate_fn=dense_sequential_collate_fn,
        num_workers=cfg.training.data.num_workers)

    # Model
    mc = cfg.training.model
    pfno_cfg = PFNOConfig(
        h_ctx     = h_ctx,
        m_pred    = m_pred,
        modes_t   = mc.get("modes_t",  4),
        modes_h   = mc.get("modes_h",  8),
        modes_w   = mc.get("modes_w",  8),
        width     = mc.get("width",   32),
        n_layers  = mc.get("n_layers", 4),
        use_grid  = mc.get("use_grid", True),
        use_time  = mc.get("use_time", True),
        seq_len_ref = seq_len,
        min_std   = mc.get("min_std", 1e-4),
    )
    model = PFNO(pfno_cfg).to(device)
    print(f"PFNO Params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    opt_cfg   = cfg.training.optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt_cfg.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt_cfg.max_epochs, eta_min=opt_cfg.min_lr)

    # Curriculum Scheduler for Autoregressive Rolling
    horizon_scheduler = LinearBetaScheduler(
        beta_start=horizon_start,
        beta_end=horizon_end,
        num_steps=curr_cfg.get("epochs", 100)
    )

    beta_loss = cfg.training.get("loss", {}).get("beta", 0.5)

    def criterion(dist, gt):
        return beta_nll_loss(dist, gt, beta=beta_loss)

    best_val = float('inf')

    # To pass 2 channels [mu, sigma] initially, we start with 0 std
    def make_pfno_ctx(frames_ctx):
        B_size, h_c, h_s, w_s = frames_ctx.shape
        zeros_std = torch.zeros_like(frames_ctx)
        return torch.stack([frames_ctx, zeros_std], dim=1)

    for epoch in range(opt_cfg.max_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        # Número de pasos AUTOREGRESIVOS para este epoch
        h_autoreg = int(round(horizon_scheduler.update(epoch)))
        H_total = h_autoreg * m_pred
        
        t_data_acc = 0.0
        t_prep_acc = 0.0
        t_forward_acc = 0.0
        t_backward_acc = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (H: {H_total} | h={h_autoreg} × m={m_pred})")
        
        t_batch_start = time.time()
        for batch in pbar:
            t_data_acc += (time.time() - t_batch_start)
            
            t_prep_start = time.time()
            ctx_obs, trg_obs, _ = batch
            ctx_obs = ctx_obs.to(device)
            trg_obs = trg_obs.to(device)

            T      = ctx_obs.xs.shape[1]
            B_size = ctx_obs.xs.shape[0]

            frames  = ctx_obs.values[:, :, :, 0].view(B_size, T, H, W)
            targets = trg_obs.values.view(B_size, T, H, W, max_forecast_horizon)

            n_win = T - h_ctx - H_total + 1
            if n_win <= 0:
                continue

            optimizer.zero_grad()
            batch_loss = 0.0
            
            t_prep_acc += (time.time() - t_prep_start)
            
            t_forward_start = time.time()
            
            for t in range(h_ctx - 1, T - H_total):
                ctx_w = frames[:, t - h_ctx + 1 : t + 1]    # (B, h_ctx, H, W)
                ctx_pfno = make_pfno_ctx(ctx_w)             # (B, 2, h_ctx, H, W)
                times_base  = make_times(t - h_ctx + 1, h_ctx, seq_len, device, B_size)

                step_loss_acc = 0.0
                curr_ctx = ctx_pfno
                curr_times = times_base

                # --- BUCLE AUTOREGRESIVO (h veces) DONDE CADA PASE ADIVINA m DELTAS ---
                for autoreg_step in range(h_autoreg):
                    t_f0 = time.time()
                    dists = model(curr_ctx, curr_times) # Retorna lista de 'm_pred' Normales

                    # Calculamos NLL de cada delta del forward actual
                    for i_m, d in enumerate(dists):
                        abs_h = autoreg_step * m_pred + i_m
                        gt_h = targets[:, t, :, :, abs_h].unsqueeze(-1)
                        
                        nll = 0.5 * (((d.mean - gt_h) ** 2) / d.variance + torch.log(d.variance))
                        weight = d.variance.detach() ** beta_loss
                        
                        step_loss = (nll * weight).mean() / (H_total * n_win)
                        step_loss_acc = step_loss_acc + step_loss
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_forward_acc += (time.time() - t_f0)

                    if autoreg_step < h_autoreg - 1:
                        new_frames = []
                        # Variedad en la forma de entrenar (probabilidad 50% de usar GT puro o Rollout predicho)
                        use_gt_context = random.random() < 0.5
                        
                        if use_gt_context:
                            for i_m in range(m_pred):
                                abs_h = autoreg_step * m_pred + i_m
                                gt_frame = targets[:, t, :, :, abs_h]
                                zeros_std = torch.zeros_like(gt_frame)
                                new_frames.append(torch.stack([gt_frame, zeros_std], dim=1))
                        else:
                            for d in dists:
                                new_frames.append(torch.stack([d.mean[..., 0], d.stddev[..., 0]], dim=1).detach())
                        
                        # Añadimos bloque de frames 'm' y desplazamos conservando últimos 'h_ctx'
                        new_stack = torch.stack(new_frames, dim=2)
                        curr_ctx = torch.cat([curr_ctx, new_stack], dim=2)[:, :, -h_ctx:]
                        curr_times = curr_times + (m_pred / max(seq_len - 1, 1))

                # Medir backward independiente de todo el árbol H_total (1 pase por t)
                t_b0 = time.time()
                step_loss_acc.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_backward_acc += (time.time() - t_b0)
                
                batch_loss += step_loss_acc.item() * n_win

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            optimizer.step()

            avg = batch_loss / n_win
            train_loss += avg
            pbar.set_postfix({'bnll': f"{avg:.4f}"})
            
            t_batch_start = time.time()

        avg_train = train_loss / len(train_loader)
        lr_scheduler.step()
        
        print(f"\n--- DEBUG TIME EPOCH {epoch} ---")
        print(f"Data Loading: {t_data_acc:.2f}s")
        print(f"Tensor Prep:  {t_prep_acc:.2f}s")
        print(f"Forward Only: {t_forward_acc:.2f}s")
        print(f"Backward Only:{t_backward_acc:.2f}s")
        print(f"  Model Breakdown (accumulated calls x{model.time_metrics['forward_calls']}):")
        for k, v in model.time_metrics.items():
            if k != 'forward_calls':
                print(f"    - {k}: {v:.2f}s")
        model.reset_time_metrics()
        print(f"Total Train:  {(time.time() - epoch_start_time):.2f}s")
        print(f"----------------------------------\n")

        writer.add_scalar("Train/BetaNLL", avg_train, epoch)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Train/Horizon_Total", H_total, epoch)

        # Validation (always evaluate full horizon)
        val_start_time = time.time()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ctx_obs, trg_obs, _ = batch
                ctx_obs = ctx_obs.to(device)
                trg_obs = trg_obs.to(device)

                T      = ctx_obs.xs.shape[1]
                B_size = ctx_obs.xs.shape[0]
                frames  = ctx_obs.values[:, :, :, 0].view(B_size, T, H, W)
                targets = trg_obs.values.view(B_size, T, H, W, max_forecast_horizon)

                H_total_val = horizon_end * m_pred
                n_win = T - h_ctx - H_total_val + 1
                if n_win <= 0:
                    continue

                batch_nll = 0.0
                for t in range(h_ctx - 1, T - H_total_val):
                    ctx_w = frames[:, t - h_ctx + 1 : t + 1]
                    ctx_pfno = make_pfno_ctx(ctx_w)
                    times_base = make_times(t - h_ctx + 1, h_ctx, seq_len, device, B_size)

                    curr_ctx = ctx_pfno
                    curr_times = times_base

                    for autoreg_step in range(horizon_end):
                        dists = model(curr_ctx, curr_times)

                        for i_m, d in enumerate(dists):
                            abs_h = autoreg_step * m_pred + i_m
                            gt_h = targets[:, t, :, :, abs_h].unsqueeze(-1)
                            
                            batch_nll += -d.log_prob(gt_h).mean().item()

                        if autoreg_step < horizon_end - 1:
                            new_frames = []
                            for d in dists:
                                new_frames.append(torch.stack([d.mean[..., 0], d.stddev[..., 0]], dim=1).detach())
                            
                            new_stack = torch.stack(new_frames, dim=2)
                            curr_ctx = torch.cat([curr_ctx, new_stack], dim=2)[:, :, -h_ctx:]
                            curr_times = curr_times + (m_pred / max(seq_len - 1, 1))

                val_loss += batch_nll / (H_total_val * n_win)

        avg_val = val_loss / len(val_loader)
        writer.add_scalar("Val/NLL", avg_val, epoch)
        
        val_total_time = time.time() - val_start_time
        print(f"Total Val time: {val_total_time:.2f}s")
        log.info(f"Epoch {epoch}: Train={avg_train:.4f}  Val={avg_val:.4f}")

        if epoch % cfg.training.visualizer.visualize_every == 0:
            log_vis(model, val_loader, cfg, H, W, device, writer, epoch)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        save_checkpoint(model, optimizer, epoch, avg_val, cfg_dict,
                        os.path.join(ckpt_dir, "last_model.pt"))
        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(model, optimizer, epoch, avg_val, cfg_dict,
                            os.path.join(ckpt_dir, "best_model.pt"))

    writer.close()


if __name__ == "__main__":
    train()
