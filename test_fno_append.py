import torch
import torch.nn as nn
from src.models.fno_3d import FNO3dConfig, FNO3d

cfg = FNO3dConfig(
    h_ctx=10, h_pred=15,  # h_pred > h_ctx !
    modes_t=4, modes_h=6, modes_w=6,
    width=16, n_layers=4,
    use_grid=True, use_time=True,
)
model = FNO3d(cfg)
seed = torch.randn(1, cfg.h_ctx, 20, 30)
try:
    preds = model.autoregressive_forecast(seed, horizon=30, num_samples=5)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
