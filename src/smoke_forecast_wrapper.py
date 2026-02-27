"""
SmokeForecastWrapper — unified interface for all smoke forecast models.

Supported model types (cfg.experiment.model_type):
  gp              : GaussianProcess (online, no checkpoint)
  rnp             : RNP  (checkpoint .pt, stateful ConvLSTM)
  rnp_multistep   : RNPMultistep (checkpoint .pt, stateful ConvLSTM)
  fno             : FNO2d (checkpoint .pt, stateless)
  fno_3d          : FNO3d (checkpoint .pt, stateless, needs h_ctx frames)

Public API
----------
wrapper.update(obs_frame, coords, t)
    Ingest the latest map reading.  Must be called every step before predict.

predicted_maps = wrapper.predict_risk_maps(smoke_frame, coords, t, horizon)
    Returns List[(coords, cvar_flat)] length == horizon.
    'coords' is the same np.ndarray passed in (H*W, 2).
    'cvar_flat' is a np.ndarray of shape (H*W,) with CVaR risk values in [0, 1].
    The list is ready to be passed to MPPIControlDyn.set_maps().
"""

import logging
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import scipy.stats as stats
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline CVaR helper  (same formula as RiskMapBuilder.cvar_map)
# ---------------------------------------------------------------------------

def _cvar(mean: np.ndarray, std: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """Gaussian CVaR: μ + w·β·σ · φ(Φ⁻¹(α)) / (1-α)  where w = exp(-γ·β·σ)."""
    return mean + std * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)

# ---------------------------------------------------------------------------
# Dense-grid Obs builder  (mirrors save_rollouts.get_dense_obs)
# ---------------------------------------------------------------------------

def _build_obs(smoke_frame_flat: np.ndarray,
               coords: np.ndarray,
               x_size: float,
               y_size: float,
               device: torch.device):
    """
    Build a normalised Obs for a single time-step from a flat smoke reading.

    Parameters
    ----------
    smoke_frame_flat : (H*W,) float32
    coords           : (H*W, 2) float32  — world-space (x, y)
    x_size, y_size   : world dimensions
    device           : torch device
    """
    from src.models.shared.observations import Obs

    P = smoke_frame_flat.shape[0]

    xs_norm = 2.0 * coords[:, 0] / x_size - 1.0   # [-1, 1]
    ys_norm = 2.0 * coords[:, 1] / y_size - 1.0

    xs  = torch.tensor(xs_norm, dtype=torch.float32).view(1, 1, P, 1).to(device)
    ys  = torch.tensor(ys_norm, dtype=torch.float32).view(1, 1, P, 1).to(device)
    vals = torch.tensor(smoke_frame_flat, dtype=torch.float32).view(1, 1, P, 1).to(device)

    return Obs(xs=xs, ys=ys, values=vals, mask=None, ts=None)


# ---------------------------------------------------------------------------
# SmokeForecastWrapper
# ---------------------------------------------------------------------------

class SmokeForecastWrapper:
    """
    Unified forecast wrapper:  model.update() + model.predict_risk_maps()

    Parameters
    ----------
    model_type   : str   one of {gp, rnp, rnp_multistep, fno, fno_3d}
    checkpoint   : str | None   path to .pt file (not needed for gp)
    x_size       : float  world x dimension
    y_size       : float  world y dimension
    cvar_alpha   : float  CVaR confidence level
    gamma        : float  CVaR exponential weight parameter
    beta         : float  CVaR std scaling parameter
    device       : str    'cpu' or 'cuda'
    h_ctx        : int    context history length for FNO3d (ignored for others)
    """

    def __init__(
        self,
        model_type: str,
        x_size: float,
        y_size: float,
        checkpoint: Optional[str] = None,
        cvar_alpha: float = 0.95,
        gamma: float = 0.75,
        beta: float = 0.20,
        device: str = "cpu",
        h_ctx: int = 5,
    ):
        self.model_type  = model_type.lower()
        self.x_size      = x_size
        self.y_size      = y_size
        self.cvar_alpha  = cvar_alpha
        self.gamma       = gamma
        self.beta        = beta
        self.device      = torch.device(device)
        self.h_ctx       = h_ctx

        # Internal state
        self._rnp_state  = None          # ConvLSTM hidden state (RNP / RNPMultistep)
        self._fno_ctx    = deque(maxlen=h_ctx)   # rolling frame buffer for FNO3d
        self._last_frame: Optional[np.ndarray] = None   # latest flat smoke map

        # Load model
        self.model = self._load_model(checkpoint)
        log.info(f"SmokeForecastWrapper ready — model={model_type}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint: Optional[str]):
        if self.model_type == "gp":
            from src.models.gaussian_process import GaussianProcess
            gp = GaussianProcess(online=True)
            return gp

        if checkpoint is None:
            raise ValueError(f"model_type='{self.model_type}' requires a checkpoint path.")

        import torch
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        hp   = ckpt.get("hyper_parameters", {})

        # Unwrap nested hydra dict if needed
        if isinstance(hp, dict) and "training" in hp:
            model_hp = hp["training"]["model"]
        else:
            model_hp = hp if isinstance(hp, dict) else {}

        if self.model_type in ("rnp", "rnp_multistep"):
            from src.models.model_free.rnp import RNP, RNPConfig
            from src.models.model_free.rnp_multistep import RNPMultistep
            import dataclasses
            valid = {f.name for f in dataclasses.fields(RNPConfig)}
            cfg = RNPConfig(**{k: v for k, v in model_hp.items() if k in valid})
            if self.model_type == "rnp_multistep":
                fh = model_hp.get("forecast_horizon", 5)
                model = RNPMultistep(cfg, forecast_horizon=fh)
            else:
                model = RNP(cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            return model

        if self.model_type == "fno":
            from src.models.model_free.fno import FNO2d, FNOConfig
            import dataclasses
            valid = {f.name for f in dataclasses.fields(FNOConfig)}
            cfg = FNOConfig(**{k: v for k, v in model_hp.items() if k in valid})
            model = FNO2d(cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            return model

        if self.model_type == "fno_3d":
            from src.models.model_free.fno_3d import FNO3d, FNO3dConfig
            import dataclasses
            valid = {f.name for f in dataclasses.fields(FNO3dConfig)}
            cfg = FNO3dConfig(**{k: v for k, v in model_hp.items() if k in valid})
            self.h_ctx = cfg.h_ctx  # override from checkpoint
            self._fno_ctx = deque(maxlen=self.h_ctx)
            model = FNO3d(cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            return model

        raise ValueError(f"Unknown model_type: '{self.model_type}'. "
                         "Choose from: gp, rnp, rnp_multistep, fno, fno_3d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, smoke_frame: np.ndarray, coords: np.ndarray, t: float):
        """
        Ingest one observation.  Call this EVERY step before predict_risk_maps.

        Parameters
        ----------
        smoke_frame : (H*W,) float32  — flattened smoke density values
        coords      : (H*W, 2) float32 — world (x, y) of each reading
        t           : float  — current time (seconds)
        """
        self._last_frame = smoke_frame.astype(np.float32)

        if self.model_type == "gp":
            # Build (x, y, t) input — shape (N, 3)
            t_col   = np.full((coords.shape[0], 1), t, dtype=np.float32)
            X_input = np.concatenate([coords, t_col], axis=-1)            # (N, 3)
            y_input = smoke_frame.astype(np.float32).reshape(-1, 1)       # (N, 1) — must match X dim
            self.model.track_data(X_input, y_input)
            self.model.update()
            return

        if self.model_type in ("fno_3d",):
            # Add 2-D frame to rolling context buffer
            H_grid = int(round(self.y_size / 1.0))  # approximate; real shape inferred at predict
            W_grid = int(round(self.x_size / 1.0))
            # Use the raw flat frame; will be reshaped in predict
            self._fno_ctx.append(self._last_frame.copy())
            return

        if self.model_type in ("rnp", "rnp_multistep"):
            # Update hidden state with current observation
            obs = _build_obs(smoke_frame, coords, self.x_size, self.y_size, self.device)
            with torch.no_grad():
                if self._rnp_state is None:
                    self._rnp_state = self.model.init_state(1, self.device)
                out = self.model(self._rnp_state, obs, target_obs=None)
                self._rnp_state = out.state
            return

        # FNO (stateless): nothing to update, just remember last frame
        # (self._last_frame already set above)

    def predict_risk_maps(
        self,
        smoke_frame: np.ndarray,
        coords: np.ndarray,
        t: float,
        horizon: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Predict `horizon` future risk maps.

        Parameters
        ----------
        smoke_frame : (H*W,) float32  — current observation (same as passed to update)
        coords      : (H*W, 2) float32 — world (x, y) for each grid cell
        t           : float  — current time (seconds)
        horizon     : int  — planning horizon steps

        Returns
        -------
        List of (coords, cvar_flat) tuples, length == horizon.
        cvar_flat is (H*W,) float32 with CVaR risk values.
        """
        if self.model_type == "gp":
            return self._predict_gp(coords, t, horizon)
        if self.model_type in ("rnp", "rnp_multistep"):
            return self._predict_rnp(smoke_frame, coords, horizon)
        if self.model_type == "fno":
            return self._predict_fno(smoke_frame, coords, horizon)
        if self.model_type == "fno_3d":
            return self._predict_fno3d(smoke_frame, coords, horizon)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    # ------------------------------------------------------------------
    # Per-model prediction implementations
    # ------------------------------------------------------------------

    def _cvar_risk(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        risk = _cvar(mean, std, self.cvar_alpha)
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    # ---- GP -----------------------------------------------------------

    def _predict_gp(self, coords: np.ndarray, t: float, horizon: int
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        results = []
        for h in range(1, horizon + 1):
            t_future = t + h  # one time unit ahead per step (approximate)
            t_col   = np.full((coords.shape[0], 1), t_future, dtype=np.float32)
            X_query = np.concatenate([coords, t_col], axis=-1)
            mean, std = self.model.predict(X_query)
            mean = mean.squeeze().astype(np.float32)
            std  = std.squeeze().astype(np.float32)
            results.append((coords, self._cvar_risk(mean, std)))
        return results

    # ---- RNP / RNPMultistep ------------------------------------------

    def _predict_rnp(self, smoke_frame: np.ndarray, coords: np.ndarray, horizon: int
                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
        ctx_obs   = _build_obs(smoke_frame, coords, self.x_size, self.y_size, self.device)
        query_obs = _build_obs(np.zeros_like(smoke_frame), coords,
                               self.x_size, self.y_size, self.device)

        # Clone state so we do not advance the running state during planning
        state_clone = None
        if self._rnp_state is not None:
            state_clone = [(l[0].clone(), l[1].clone()) for l in self._rnp_state]

        with torch.no_grad():
            preds = self.model.autoregressive_forecast(
                state=state_clone,
                context_obs=ctx_obs,
                target_obs=query_obs,
                horizon=horizon,
                num_samples=1,
            )

        P = smoke_frame.shape[0]
        results = []
        for step_pred in preds:
            # mean / std shapes: (1, 1, P, 1)
            mean = step_pred["mean"].values.squeeze().detach().cpu().numpy().astype(np.float32)
            std  = step_pred["std"].values.squeeze().detach().cpu().numpy().astype(np.float32)
            # Defensive reshape in case of shape mismatch
            mean = mean.reshape(P)
            std  = std.reshape(P)
            results.append((coords, self._cvar_risk(mean, std)))
        return results

    # ---- FNO2d (stateless) -------------------------------------------

    def _predict_fno(self, smoke_frame: np.ndarray, coords: np.ndarray, horizon: int
                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
        P = smoke_frame.shape[0]
        # Infer grid shape from sensor density —  coords are (H*W, 2)
        # The FNO expects a 2D grid, so we try to recover H, W
        H_grid, W_grid = self._infer_grid_shape(coords)

        frame_2d = smoke_frame.reshape(H_grid, W_grid).astype(np.float32)
        seed = torch.tensor(frame_2d, dtype=torch.float32, device=self.device)
        seed = seed.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

        with torch.no_grad():
            raw_preds = self.model.autoregressive_forecast(
                seed_frame=seed,
                horizon=horizon,
                num_samples=1,
            )

        results = []
        for p in raw_preds:
            mean = p["mean"].reshape(P).astype(np.float32)
            std  = p["std"].reshape(P).astype(np.float32)
            results.append((coords, self._cvar_risk(mean, std)))
        return results

    # ---- FNO3d -------------------------------------------------------

    def _predict_fno3d(self, smoke_frame: np.ndarray, coords: np.ndarray, horizon: int
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
        P = smoke_frame.shape[0]
        H_grid, W_grid = self._infer_grid_shape(coords)

        h_ctx = self.model.cfg.h_ctx

        # Build context window: pad with zeros if not enough history
        frames_available = list(self._fno_ctx)
        # The latest frame is appended in update(); if update() was not called yet
        # for this step, fall back to smoke_frame
        if len(frames_available) == 0:
            frames_available = [smoke_frame]
        pad_len = max(0, h_ctx - len(frames_available))
        padding = [np.zeros_like(smoke_frame)] * pad_len
        window  = padding + frames_available[-h_ctx:]

        seed_frames = np.stack([f.reshape(H_grid, W_grid) for f in window], axis=0)  # (h_ctx, H, W)
        seed = torch.tensor(seed_frames, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, h_ctx, H, W)

        with torch.no_grad():
            raw_preds = self.model.autoregressive_forecast(
                seed_frames=seed,
                seed_t_start=0,
                horizon=horizon,
                num_samples=1,
            )

        results = []
        for p in raw_preds:
            mean = p["mean"].reshape(P).astype(np.float32)
            std  = p["std"].reshape(P).astype(np.float32)
            results.append((coords, self._cvar_risk(mean, std)))
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_grid_shape(coords: np.ndarray) -> Tuple[int, int]:
        """
        Attempt to recover (H, W) from a flat coordinate array (H*W, 2).
        Works when coords are laid out row-major (meshgrid indexing='ij').
        Falls back to a square approximation.
        """
        P = coords.shape[0]
        # Count unique y values (rows) and x values (cols)
        unique_y = np.unique(np.round(coords[:, 1], decimals=3))
        unique_x = np.unique(np.round(coords[:, 0], decimals=3))
        H = len(unique_y)
        W = len(unique_x)
        if H * W == P:
            return H, W
        # Fallback: square
        sq = int(round(P ** 0.5))
        return sq, sq
