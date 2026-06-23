import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import torch
from scipy.spatial import KDTree

log = logging.getLogger(__name__)


@dataclass
class SmokeForecastWrapperConfig:
    model_type: str
    checkpoint: Optional[str]
    ensemble_size: int = 15
    process_noise_std: float = 0.05
    cvar_alpha: float = 0.95
    device: str = "cpu"
    h_ctx: int = 5
    x_size: float = 30.0
    y_size: float = 30.0


def _cvar(mean: np.ndarray, std: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """Gaussian CVaR: μ + σ · φ(Φ⁻¹(α)) / (1-α)"""
    if alpha >= 1.0 - 1e-6:
        # Fallback to avoid division by zero
        return mean + 3.0 * std
    pdf = stats.norm.pdf(stats.norm.ppf(alpha))
    return mean + std * pdf / (1 - alpha)


class SmokeForecastWrapper:
    """Unified forecast wrapper that manages an ensemble of smoke forecast contexts,
    updating them with local/sparse observations and predicting CVaR risk maps.

    Parameters
    ----------
    cfg : SmokeForecastWrapperConfig
        Configuration object containing model type, checkpoint, ensemble size, etc.
    """

    def __init__(self, cfg: SmokeForecastWrapperConfig):
        self.cfg = cfg
        self.model_type = cfg.model_type.lower()

        self.device = torch.device(cfg.device)
        self.h_ctx = cfg.h_ctx

        # Load FNO or ConvLSTM model
        self.model = self._load_model(cfg.checkpoint)

        # Stateful ensemble contexts: shape (ensemble_size, h_ctx, H, W)
        self.rollout_contexts: Optional[torch.Tensor] = None
        # Siguiente frame predicho para correr el buffer en el siguiente update
        self._next_predicted_frame: Optional[torch.Tensor] = None
        self._step_counter = 0

        # Global coordinates cache for KDTree mapping
        self._coords_global_sorted: Optional[np.ndarray] = None
        self._sort_idx: Optional[np.ndarray] = None
        self._inv_sort: Optional[np.ndarray] = None

        log.info(
            f"SmokeForecastWrapper ready — model={self.model_type}, "
            f"ensemble_size={cfg.ensemble_size}, device={cfg.device}"
        )

    def _load_model(self, checkpoint: Optional[str]) -> torch.nn.Module:
        if checkpoint is None:
            raise ValueError(f"model_type='{self.model_type}' requires a checkpoint path.")

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        hp = ckpt.get("hyper_parameters", {})

        if isinstance(hp, dict) and "training" in hp:
            model_hp = hp["training"]["model"]
        else:
            model_hp = hp.get("model", hp) if isinstance(hp, dict) else {}

        if self.model_type == "fno":
            from models.fno import FNO, FNOConfig

            valid = set(FNOConfig.model_fields.keys())
            cfg = FNOConfig(**{k: v for k, v in model_hp.items() if k in valid})
            self.h_ctx = cfg.h_ctx  # override from checkpoint
            model = FNO(cfg)

            # Robustly load state dict
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            model.to(self.device).eval()
            return model

        if self.model_type == "conv_lstm":
            from models.conv_lstm import ConvLSTMConfig, ConvLSTMModel

            valid = set(ConvLSTMConfig.model_fields.keys())
            cfg = ConvLSTMConfig(**{k: v for k, v in model_hp.items() if k in valid})
            self.h_ctx = cfg.h_ctx  # override from checkpoint
            model = ConvLSTMModel(cfg)

            # Robustly load state dict
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            model.to(self.device).eval()
            return model

        raise ValueError(f"Unknown model_type: '{self.model_type}'. Choose from: fno, conv_lstm")

    def update(
        self,
        smoke_frame: np.ndarray,
        coords: np.ndarray,
        t: float,
        coords_global: Optional[np.ndarray] = None,
    ):
        """Ingests new local/sparse observations and updates the ensemble contexts.
        Uses a KDTree to map observation coordinates to global grid cell coordinates.

        Parameters
        ----------
        smoke_frame   : (N,) float32 — observed smoke density readings (can be sparse or full)
        coords        : (N, 2) float32 — coordinates of the observed readings
        t             : float — current time step (seconds)
        coords_global : Optional[(H*W, 2)] float32 — full global grid coordinates.
                        If not provided, falls back to using `coords`.
        """
        # If coords_global is not provided, fallback to coords (valid for GlobalSensor)
        if coords_global is None:
            coords_global = coords

        H_grid, W_grid = self._infer_grid_shape(coords_global)

        # Cache global coordinates sorting indices (row-major mapping)
        if self._coords_global_sorted is None:
            self._sort_idx = np.lexsort((coords_global[:, 0], coords_global[:, 1]))
            self._inv_sort = np.argsort(self._sort_idx)
            self._coords_global_sorted = coords_global[self._sort_idx]

        # 1. Initialize rollout_contexts on the very first step (t=0 or first call)
        if self.rollout_contexts is None:
            # Initialize ensemble with high diversity [0, 0.8]
            self.rollout_contexts = (
                torch.rand(self.cfg.ensemble_size, self.h_ctx, H_grid, W_grid, device=self.device)
                * 0.8
            )
            self._step_counter = 0

            # Find matching global grid indices for observed coordinates
            tree = KDTree(self._coords_global_sorted)
            _, indices = tree.query(coords)
            grid_y = indices // W_grid
            grid_x = indices % W_grid

            obs_values_torch = torch.tensor(smoke_frame, dtype=torch.float32, device=self.device)
            # Flatten to 1D
            if obs_values_torch.dim() > 1:
                obs_values_torch = obs_values_torch.squeeze(-1)

            # Patch all h_ctx frames in the history window with the initial observation
            for h in range(self.h_ctx):
                self.rollout_contexts[:, h, grid_y, grid_x] = obs_values_torch

        # 2. Shift and update context for subsequent steps (t > 0)
        else:
            self._step_counter += 1
            if self._next_predicted_frame is not None:
                # Roll history window and append the latest predicted step
                self.rollout_contexts = torch.cat(
                    [self.rollout_contexts[:, 1:], self._next_predicted_frame.unsqueeze(1)], dim=1
                )
            else:
                # Fallback: repeat the last frame if no prediction is cached
                self.rollout_contexts = torch.cat(
                    [self.rollout_contexts[:, 1:], self.rollout_contexts[:, -1:]], dim=1
                )

            # Find global grid cell indices for current observations
            tree = KDTree(self._coords_global_sorted)
            _, indices = tree.query(coords)
            grid_y = indices // W_grid
            grid_x = indices % W_grid

            obs_values_torch = torch.tensor(smoke_frame, dtype=torch.float32, device=self.device)
            if obs_values_torch.dim() > 1:
                obs_values_torch = obs_values_torch.squeeze(-1)

            # Direct Masked Injection: Hard-patch observed region in the latest frame
            self.rollout_contexts[:, -1, grid_y, grid_x] = obs_values_torch

            # Inject independent process noise to unobserved regions to represent growing uncertainty
            observed_mask = torch.zeros((H_grid, W_grid), dtype=torch.bool, device=self.device)
            observed_mask[grid_y, grid_x] = True

            noise = (
                torch.randn(self.cfg.ensemble_size, H_grid, W_grid, device=self.device)
                * self.cfg.process_noise_std
            )
            self.rollout_contexts[:, -1, ~observed_mask] += noise[:, ~observed_mask]

            # Keep values physically bound to [0.0, 1.0]
            self.rollout_contexts = torch.clamp(self.rollout_contexts, 0.0, 1.0)

    def predict_risk_maps(
        self,
        smoke_frame: np.ndarray,
        coords: np.ndarray,
        t: float,
        horizon: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Predicts `horizon` future risk maps (CVaR).
        Runs autoregressive forecasting on the ensemble and calculates risk maps.

        Parameters
        ----------
        smoke_frame : (N,) float32 — current observation (same as passed to update)
        coords      : (N, 2) float32 — coordinates for the observation
        t           : float — current time step (seconds)
        horizon     : int — planning horizon steps

        Returns:
        -------
        List of (coords, cvar_flat) tuples, length == horizon.
        """
        # Lazily initialize if update hasn't been called yet
        if self.rollout_contexts is None:
            self.update(smoke_frame, coords, t)

        H_grid, W_grid = self._infer_grid_shape(coords)

        # Run ensemble autoregressive forecast
        # Returns List[dict], each containing "sample", "mean", "std"
        preds = self._forecast_horizon(horizon, mode="mean")

        # Cache the first prediction step to use for context shifting in the next update
        self._next_predicted_frame = torch.tensor(preds[0]["sample"], device=self.device)

        results = []
        for h in range(horizon):
            pred_step = min(h, len(preds) - 1)

            # samples shape: (ensemble_size, H, W)
            samples = preds[pred_step]["sample"]
            # model stds shape: (ensemble_size, H, W)
            stds = preds[pred_step]["std"]

            # Compute ensemble mean prediction
            mean_h = samples.mean(axis=0)

            # Epistemic uncertainty: variance of forecasts across ensemble members
            var_epistemic_h = samples.var(axis=0)

            # Aleatoric uncertainty: mean of the model-predicted variances
            var_aleatoric_h = (stds**2).mean(axis=0)

            # Combine uncertainties
            std_total_h = np.sqrt(var_aleatoric_h + var_epistemic_h)

            # Compute Gaussian CVaR risk map
            cvar_grid = _cvar(mean_h, std_total_h, self.cfg.cvar_alpha)
            cvar_grid = np.clip(cvar_grid, 0.0, 1.0).astype(np.float32)

            # Flatten row-major and restore caller's coordinate order
            cvar_flat = cvar_grid.ravel()[self._inv_sort]
            results.append((coords, cvar_flat))

        return results

    def _forecast_horizon(self, horizon: int, mode: str = "mean") -> List[dict]:
        """Runs autoregressive forecasting over the ensemble of rollout_contexts.

        Returns List[dict] of length horizon. Each dict contains:
            - "sample": np.ndarray of shape (ensemble_size, H, W)
            - "mean": np.ndarray of shape (ensemble_size, H, W)
            - "std": np.ndarray of shape (ensemble_size, H, W)
        """
        B_batch, h_ctx, H_dim, W_dim = self.rollout_contexts.shape
        ctx = self.rollout_contexts.clone()
        t_offset = self._step_counter

        # Check sequence length references from config
        seq_len_ref = getattr(self.model.cfg, "sequence_length", 30) or 30
        ref = max(seq_len_ref - 1, 1)

        preds = []
        while len(preds) < horizon:
            t_abs = torch.arange(
                t_offset, t_offset + h_ctx, device=self.device, dtype=torch.float32
            )
            times = (t_abs / ref).unsqueeze(0).expand(B_batch, -1)  # (ensemble_size, h_ctx)

            with torch.no_grad():
                # Forward pass outputs List[Normal] of length h_pred, each shape (M, H, W, 1)
                dists = self.model(ctx, times)

            new_frames_for_ctx = []
            eps = torch.randn(B_batch, 1, 1, 1, device=self.device)
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

                preds.append(
                    {
                        "mean": mu.squeeze(-1).cpu().numpy(),
                        "std": sigma.squeeze(-1).cpu().numpy(),
                        "sample": sampled.squeeze(-1).cpu().numpy(),
                    }
                )
                new_frames_for_ctx.append(sampled)

            # Slide context window
            n_slide = len(new_frames_for_ctx)
            new_stack = torch.cat(
                [f.permute(0, 3, 1, 2) for f in new_frames_for_ctx], dim=1
            )  # (M, n_slide, H, W)
            ctx = torch.cat([ctx, new_stack], dim=1)[:, -h_ctx:]
            t_offset += n_slide

        return preds

    def flatten_and_restore(self, grid_tensor: torch.Tensor) -> np.ndarray:
        """Helper to flatten spatial predictions and restore the caller's original coordinate order.

        Parameters
        ----------
        grid_tensor : torch.Tensor of shape (..., H, W)

        Returns:
        -------
        np.ndarray of shape (..., H*W) in the original coordinate layout
        """
        # Convert to numpy
        grid_np = grid_tensor.detach().cpu().numpy()
        # Flatten the last two spatial dimensions
        shape_prefix = grid_np.shape[:-2]
        flat_np = grid_np.reshape(shape_prefix + (-1,))
        # Restore the original coordinate layout using the cached inverse sort index
        if self._inv_sort is None:
            raise ValueError("Coordinates mapping not initialized. Call update() first.")
        return flat_np[..., self._inv_sort]

    @staticmethod
    def _infer_grid_shape(coords: np.ndarray) -> Tuple[int, int]:
        """Attempt to recover (H, W) from a flat coordinate array (H*W, 2).
        Works when coords are laid out row-major (meshgrid indexing='ij').
        Falls back to a square approximation.
        """
        P = coords.shape[0]
        unique_y = np.unique(np.round(coords[:, 1], decimals=3))
        unique_x = np.unique(np.round(coords[:, 0], decimals=3))
        H = len(unique_y)
        W = len(unique_x)
        if H * W == P:
            return H, W
        sq = int(round(P**0.5))
        return sq, sq
