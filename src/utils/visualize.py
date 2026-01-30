import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms

def log_pinn_fields(model, dataloader, device, writer, epoch, name="Val", num_samples=5):
    """
    Genera 3 imágenes separadas en TensorBoard:
    1. Reconstrucción Contexto (Context Last)
    2. Forecasting Inmediato (Forecast +1)
    3. Forecasting a largo plazo (Forecast Last)
    Cada imagen contiene num_samples casos.
    """
    model.eval()
    dataset = dataloader.dataset
    H, W = dataset.H, dataset.W
    res, dt = dataset.res, dataset.dt
    
    # Grid de consulta completa
    x_range, y_range = np.arange(W) * res, np.arange(H) * res
    X, Y = np.meshgrid(x_range, y_range)
    query_xs_base = torch.from_numpy(X.flatten()).float().to(device).unsqueeze(0)
    query_ys_base = torch.from_numpy(Y.flatten()).float().to(device).unsqueeze(0)
    
    # Tomar muestras aleatorias del dataset completo
    dataset_len = len(dataset)
    num_samples = min(num_samples, dataset_len)
    indices = np.random.randint(0, dataset_len, num_samples)
    samples = [dataset[i] for i in indices]
    
    from src.models.model_based.dataset import pinn_collate_fn
    batch = pinn_collate_fn(samples)
    
    # Despaquetar batch
    if len(batch) == 6:
        context, target, total, inflow_map, ep_indices, t0_list = batch
    else:
        context, target, total, inflow_map, ep_indices = batch
        t0_list = torch.zeros_like(ep_indices).float()

    num_avail = context.xs.shape[0]
    num_samples = min(num_samples, num_avail)
    
    from src.models.model_based.utils import ObsPINN

    # Definir los tipos de snapshots
    snapshot_configs = [
        ("Context_Reconstruction", "ts_last_ctx"), # Usará el último frame del contexto
        ("Forecast_Next", "ts_first_trg"),         # Usará el primer frame del target
        ("Forecast_End", "ts_last_trg")            # Usará el último frame del target
    ]

    for tag_suffix, time_type in snapshot_configs:
        fig, axes = plt.subplots(num_samples, 7, figsize=(35, 5 * num_samples))
        if num_samples == 1: axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            ep_idx = ep_indices[i].item()
            t0 = t0_list[i].item() # Offset de tiempo absoluto
            
            # Determinar el timestamp relativo (lo que la red entiende)
            if time_type == "ts_last_ctx":
                t_val = context.ts[i, -1].item()
                t_label = "Context Last"
            elif time_type == "ts_first_trg":
                t_val = target.ts[i, 0].item()
                t_label = "Forecast +1"
            else:
                t_val = target.ts[i, -1].item()
                t_label = "Forecast Last"

            # 1. Obtener GT usando tiempo absoluto (t_relativo + t_offset)
            t_abs = t_val + t0
            t_idx_original = int(round(t_abs / dt))
            gt_full = dataset.smoke_data[ep_idx, t_idx_original]

            # 2. Inferencia PINN
            query_ts = torch.full_like(query_xs_base, t_val)
            query = ObsPINN(xs=query_xs_base, ys=query_ys_base, ts=query_ts)
            ctx_sample = ObsPINN(
                xs=context.xs[i:i+1].to(device), ys=context.ys[i:i+1].to(device), 
                ts=context.ts[i:i+1].to(device), values=context.values[i:i+1].to(device)
            )
            
            with torch.no_grad():
                output = model(ctx_sample, query)
                
            # 3. Extraer campos
            pred_s = output.smoke_dist.loc.cpu().numpy().reshape(H, W)
            u = output.u.cpu().numpy().reshape(H, W) if (hasattr(output, 'u') and output.u is not None) else None
            v = output.v.cpu().numpy().reshape(H, W) if (hasattr(output, 'v') and output.v is not None) else None
            fu = output.fu.cpu().numpy().reshape(H, W) if (hasattr(output, 'fu') and output.fu is not None) else None
            fv = output.fv.cpu().numpy().reshape(H, W) if (hasattr(output, 'fv') and output.fv is not None) else None
            error = np.abs(gt_full - pred_s)

            cols = [
                (gt_full, "GT", "plasma", [0, 1]),
                (pred_s, "Pred", "plasma", [0, 1]),
                (error, "Error", "inferno", [0, 0.5])
            ]
            if u is not None: cols.append((u, "U", "RdBu_r", [None, None]))
            if v is not None: cols.append((v, "V", "RdBu_r", [None, None]))
            if fu is not None: cols.append((fu, "FU", "RdBu_r", [None, None]))
            if fv is not None: cols.append((fv, "FV", "RdBu_r", [None, None]))

            for col_idx, (data, title, cmap, clim) in enumerate(cols):
                ax = axes[i, col_idx]
                im = ax.imshow(data, origin='lower', extent=[0, dataset.W*res, 0, dataset.H*res], cmap=cmap)
                if clim[0] is not None: im.set_clim(clim[0], clim[1])
                
                if i == 0: ax.set_title(f"{title}\n{t_label} (t={t_val:.1f}s)")
                else: ax.set_title(f"Case {i+1} (t={t_val:.1f}s)")
                
                ax.set_xlabel("x (m)")
                if col_idx == 0: ax.set_ylabel("y (m)")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        writer.add_image(f"{name}/{tag_suffix}", transforms.ToTensor()(img), epoch)
        plt.close()

def log_physics_params(loss_fn, writer, epoch):
    """Logs learnable physics parameters to TensorBoard with better grouping."""
    writer.add_scalar("Phys/D_diffusion", loss_fn.D.item(), epoch)
    # También loggeamos los gradientes para ver si están llegando
    if hasattr(loss_fn.log_D, 'grad') and loss_fn.log_D.grad is not None:
        writer.add_scalar("Grads/log_D", loss_fn.log_D.grad.item(), epoch)