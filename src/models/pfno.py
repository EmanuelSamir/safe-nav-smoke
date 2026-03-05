import math
from dataclasses import dataclass
from typing import List
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.models.fno_3d import SpectralConv3d

@dataclass
class PFNOConfig:
    # Context / prediction
    h_ctx:  int = 10          
    m_pred: int = 5           

    # 3D spectral modes
    modes_t: int = 4          
    modes_h: int = 8          
    modes_w: int = 8          

    # Network width and depth
    width:    int = 32        
    n_layers: int = 4         

    # Features
    use_grid: bool = True     
    use_time: bool = True     

    # Normalisation
    seq_len_ref: int = 25     
    min_std:    float = 1e-4


class PFNO(nn.Module):
    """
    Probabilistic FNO-3D (PFNO) Network.
    
    ========================================================================================
    ¿CÓMO FUNCIONA LA PREDICCIÓN (RECONSTRUCCIÓN DIRECTA)?
    
    1. EXTRACCIÓN Y PREDICCIÓN:
       El backbone usa convoluciones espectrales en 3D que miran todo el contexto (h_ctx).
       Lanza un pase hacia adelante y proyecta el resultado a 2*m_pred canales.
       Cada par de (mu, sigma) representa directamente la RECONSTRUCCIÓN COMPLETA 
       del frame correspondiente en el futuro. Ya no se aprenden deltas o saltos.
    
    2. DESPLIEGUE EN ROLLOUT:
       Dado que el modelo predice `m_pred` pasos como frames directos consecutivos,
       para predecir un horizonte H (ej. h_pred = 20), se avanza el contexto deslizando
       la ventana temporal.
    ========================================================================================
    """

    def __init__(self, cfg: PFNOConfig):
        super().__init__()
        self.cfg = cfg

        # PFNO uses [mu, sigma] (2 channels) + time (1, if use_time) → C_in
        self.c_in  = 2 + (1 if cfg.use_time else 0)
        self.grid_ch = 2 if cfg.use_grid else 0
        self.c_post  = cfg.width + self.grid_ch

        self.lift = nn.Conv3d(self.c_in, cfg.width, kernel_size=1)

        self.spec_convs = nn.ModuleList([
            SpectralConv3d(cfg.width, cfg.width,
                           cfg.modes_t, cfg.modes_h, cfg.modes_w)
            for _ in range(cfg.n_layers)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv3d(cfg.width, cfg.width, kernel_size=1)
            for _ in range(cfg.n_layers)
        ])

        self.temporal_agg = nn.Conv3d(cfg.width, cfg.width, kernel_size=(cfg.h_ctx, 1, 1))

        self.fc1 = nn.Linear(self.c_post, 128)
        self.fc2 = nn.Linear(128, 2 * cfg.m_pred) # Mu and Sigma for FULL horizon
        
        # Timing metrics for debug
        self.time_metrics = {
            'build_feat': 0.0,
            'lift': 0.0,
            'spec_layers': 0.0,
            'temporal_agg': 0.0,
            'mlp': 0.0,
            'autoreg_post': 0.0,
            'forward_calls': 0
        }

    def reset_time_metrics(self):
        for k in self.time_metrics:
            self.time_metrics[k] = 0.0

    def _build_feat_volume(self, frames: torch.Tensor, times: torch.Tensor | None) -> torch.Tensor:
        if frames.dim() == 4:
            feat = frames.unsqueeze(1)
        elif frames.dim() == 5:
            feat = frames
        else:
            raise ValueError(f"Unexpected frames shape: {frames.shape}")

        if self.cfg.use_time:
            if times is None:
                B, T = frames.shape[0], frames.shape[-3]
                times = torch.linspace(0, 1, T, device=frames.device).unsqueeze(0).expand(B, -1)

            B, C, T, H, W = feat.shape
            t_feat = times.view(B, 1, T, 1, 1).expand(-1, -1, -1, H, W)
            feat = torch.cat([feat, t_feat.float()], dim=1)

        return feat

    def _build_grid(self, H: int, W: int, device) -> torch.Tensor:
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([gx, gy], dim=0).unsqueeze(0)

    def forward(self, frames: torch.Tensor, times: torch.Tensor | None = None) -> List[Normal]:
        self.time_metrics['forward_calls'] += 1
        t0 = time.time()
        
        B = frames.shape[0]
        H, W = frames.shape[-2], frames.shape[-1]

        feat = self._build_feat_volume(frames, times)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        self.time_metrics['build_feat'] += (t1 - t0)

        x = self.lift(feat)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.time()
        self.time_metrics['lift'] += (t2 - t1)

        for spec, skip in zip(self.spec_convs, self.skip_convs):
            x = F.gelu(spec(x) + skip(x))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t3 = time.time()
        self.time_metrics['spec_layers'] += (t3 - t2)

        x = self.temporal_agg(x)
        x = x.squeeze(2)

        if self.cfg.use_grid:
            grid = self._build_grid(H, W, frames.device).expand(B, -1, -1, -1)
            x = torch.cat([x, grid], dim=1)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t4 = time.time()
        self.time_metrics['temporal_agg'] += (t4 - t3)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        out = self.fc2(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t5 = time.time()
        self.time_metrics['mlp'] += (t5 - t4)

        dists = []
        
        # --- PREDICCIÓN DE TODO EL HORIZONTE (RECONSTRUCCIÓN COMPLETA DIRECTA) ---
        for h in range(self.cfg.m_pred):
            # La red aprende a reconstruir directamente la media y la varianza
            # Ya no se suma como un delta a los frames pasados
            mu_next = out[..., 2*h : 2*h+1]
            
            # Predecimos la desviación estándar directamente (asegurando positividad y estabilidad)
            sigma_next = F.softplus(out[..., 2*h+1 : 2*h+2]) + self.cfg.min_std
            
            dists.append(Normal(mu_next, sigma_next))
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.time_metrics['autoreg_post'] += (time.time() - t5)
        
        return dists

    def autoregressive_forecast(
        self,
        seed_frames:  torch.Tensor,
        seed_t_start: int = 0,
        horizon:      int = 15,
        num_samples:  int = 10,
    ) -> List[dict]:
        if seed_frames.dim() == 3:
            seed_frames = seed_frames.unsqueeze(0)
            
        if seed_frames.dim() == 4:
            zeros_std = torch.zeros_like(seed_frames)
            seed_frames = torch.stack([seed_frames, zeros_std], dim=1)

        h_ctx  = self.cfg.h_ctx
        ref    = max(self.cfg.seq_len_ref - 1, 1)
        device = seed_frames.device

        ctx = seed_frames.expand(num_samples, *([-1]*(seed_frames.dim() - 1))).clone()
        t_offset = seed_t_start
        preds    = []

        while len(preds) < horizon:
            t_abs  = torch.arange(t_offset, t_offset + h_ctx,
                                   device=device, dtype=torch.float32)
            times  = (t_abs / ref).unsqueeze(0).expand(num_samples, -1)

            with torch.no_grad():
                dists = self.forward(ctx, times)

            new_frames_for_ctx = []
            for d in dists:
                if len(preds) >= horizon:
                    break

                sampled = d.sample()
                sample_np = sampled[..., 0].cpu().to(torch.float16).numpy()
                mu_np     = d.mean[..., 0].cpu().to(torch.float16).numpy()
                std_np    = d.stddev[..., 0].cpu().to(torch.float16).numpy()

                preds.append({'sample': sample_np, 'mean': mu_np, 'std': std_np})
                
                # Para PFNO, el estado que pasa como "contexto" temporal autoregresivo es [Media, Desviación]
                new_frame = torch.stack([d.mean[..., 0], d.stddev[..., 0]], dim=1)
                new_frames_for_ctx.append(new_frame)

            n_slide   = len(new_frames_for_ctx)
            new_stack = torch.stack(new_frames_for_ctx, dim=2)
            ctx = torch.cat([ctx[:, :, n_slide:], new_stack], dim=2)

            t_offset += n_slide

        return preds
