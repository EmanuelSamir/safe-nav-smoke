import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import torch

log = logging.getLogger(__name__)


@dataclass
class SmokeForecastWrapperConfig:
    model_type: str
    x_size: float
    y_size: float
    checkpoint: Optional[str]
    cvar_alpha: float
    gamma: float
    beta: float
    device: str
    h_ctx: int


def _cvar(mean: np.ndarray, std: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """Gaussian CVaR: μ + σ · φ(Φ⁻¹(α)) / (1-α)"""
    return mean + std * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)


class SmokeForecastWrapper:
    """Unified forecast wrapper:  model.update() + model.predict_risk_maps()

    Parameters
    ----------
    model_type   : str   one of {fno, pfno, conv_lstm}
    checkpoint   : str | None   path to .pt file
    x_size       : float  world x dimension
    y_size       : float  world y dimension
    cvar_alpha   : float  CVaR confidence level
    gamma        : float  CVaR exponential weight parameter
    beta         : float  CVaR std scaling parameter
    device       : str    'cpu' or 'cuda'
    h_ctx        : int    context history length
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
        self.model_type = model_type.lower()
        self.x_size = x_size
        self.y_size = y_size
        self.cvar_alpha = cvar_alpha
        self.gamma = gamma
        self.beta = beta
        self.device = torch.device(device)
        self.h_ctx = h_ctx

        # Internal state
        self._ctx_frames = deque(maxlen=h_ctx)  # rolling frame buffer
        self._last_frame: Optional[np.ndarray] = None  # latest flat smoke map

        # Load model
        self.model = self._load_model(checkpoint)
        log.info(f"SmokeForecastWrapper ready — model={model_type}")

    def _load_model(self, checkpoint: Optional[str]):
        if checkpoint is None:
            raise ValueError(f"model_type='{self.model_type}' requires a checkpoint path.")

        import torch

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        hp = ckpt.get("hyper_parameters", {})

        if isinstance(hp, dict) and "training" in hp:
            model_hp = hp["training"]["model"]
        else:
            # train_fno.py saves t_cfg.model_dump() which has a "model" sub-key
            model_hp = hp.get("model", hp) if isinstance(hp, dict) else {}

        if self.model_type in ("fno", "pfno"):
            from models.fno import FNO, FNOConfig

            valid = set(FNOConfig.model_fields.keys())
            cfg = FNOConfig(**{k: v for k, v in model_hp.items() if k in valid})
            # Override is_pfno if model_type is pfno
            if self.model_type == "pfno":
                cfg.is_pfno = True
            self.h_ctx = cfg.h_ctx  # override from checkpoint
            self._ctx_frames = deque(maxlen=self.h_ctx)
            model = FNO(cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            return model

        if self.model_type == "conv_lstm":
            from models.conv_lstm import ConvLSTMConfig, ConvLSTMModel

            valid = set(ConvLSTMConfig.model_fields.keys())
            cfg = ConvLSTMConfig(**{k: v for k, v in model_hp.items() if k in valid})
            self.h_ctx = cfg.h_ctx  # override from checkpoint
            self._ctx_frames = deque(maxlen=self.h_ctx)
            model = ConvLSTMModel(cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            return model

        raise ValueError(
            f"Unknown model_type: '{self.model_type}'. Choose from: fno, pfno, conv_lstm"
        )

    def update(self, smoke_frame: np.ndarray, coords: np.ndarray, t: float):
        """Ingest one observation.  Call this EVERY step before predict_risk_maps.

        Parameters
        ----------
        smoke_frame : (H*W,) float32  — flattened smoke density values
        coords      : (H*W, 2) float32 — world (x, y) of each reading
        t           : float  — current time (seconds)
        """
        self._last_frame = smoke_frame.astype(np.float32)

        if self.model_type in ("fno", "conv_lstm", "pfno"):
            # Add 2-D frame to rolling context buffer
            self._ctx_frames.append(self._last_frame.copy())
            return

    def predict_risk_maps(
        self,
        smoke_frame: np.ndarray,
        coords: np.ndarray,
        t: float,
        horizon: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Predict `horizon` future risk maps.

        Parameters
        ----------
        smoke_frame : (H*W,) float32  — current observation (same as passed to update)
        coords      : (H*W, 2) float32 — world (x, y) for each grid cell
        t           : float  — current time (seconds)
        horizon     : int  — planning horizon steps

        Returns:
        -------
        List of (coords, cvar_flat) tuples, length == horizon.
        cvar_flat is (H*W,) float32 with CVaR risk values.
        """
        if self.model_type in ("fno", "conv_lstm", "pfno"):
            return self._predict_autoregressive(smoke_frame, coords, horizon)

        raise ValueError(f"Unknown model_type: {self.model_type}")

    def _cvar_risk(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        risk = _cvar(mean, std, self.cvar_alpha)
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def _predict_autoregressive(
        self, smoke_frame: np.ndarray, coords: np.ndarray, horizon: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        P = smoke_frame.shape[0]
        H_grid, W_grid = self._infer_grid_shape(coords)

        h_ctx = self.model.cfg.h_ctx

        # ------------------------------------------------------------------ #
        # Normalise flat ordering to row-major (y ascending → x ascending).  #
        # Training convention: episode[t].ravel() with numpy default C order, #
        # i.e. y (row) varies slowly, x (col) varies fast — indexing='ij'.   #
        # The caller may deliver coords/frames in any order, so we sort once  #
        # and invert after prediction to restore the caller's original order. #
        # ------------------------------------------------------------------ #
        sort_idx = np.lexsort((coords[:, 0], coords[:, 1]))  # primary sort: y, secondary: x
        inv_sort = np.argsort(sort_idx)

        # Build context window: pad with zeros if not enough history
        frames_available = list(self._ctx_frames)

        if len(frames_available) == 0:
            frames_available = [smoke_frame]
        pad_len = max(0, h_ctx - len(frames_available))
        padding = [np.zeros_like(smoke_frame)] * pad_len
        window = padding + frames_available[-h_ctx:]

        # Reorder each frame to row-major before reshape → (h_ctx, H, W)
        seed_frames = np.stack([f[sort_idx].reshape(H_grid, W_grid) for f in window], axis=0)
        seed = torch.tensor(seed_frames, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )  # (1, h_ctx, H, W)

        with torch.no_grad():
            raw_preds = self.model.autoregressive_forecast(
                seed_frames=seed,
                seed_t_start=0,
                horizon=horizon,
                num_samples=1,
                mode="mean",
            )

        results = []
        for p in raw_preds:
            # p["mean"]: (1, H, W) — flatten row-major, then restore caller's coord order
            mean = p["mean"].reshape(P)[inv_sort].astype(np.float32)
            std = p["std"].reshape(P)[inv_sort].astype(np.float32)
            results.append((coords, self._cvar_risk(mean, std)))
        return results

    @staticmethod
    def _infer_grid_shape(coords: np.ndarray) -> Tuple[int, int]:
        """Attempt to recover (H, W) from a flat coordinate array (H*W, 2).
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
        sq = int(round(P**0.5))
        return sq, sq
