import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLossBase(nn.Module):
    """Clase base con utilidades para cálculo de gradientes en PINNs."""
    def first_derivatives(self, y, w):
        """
        Calculates dy/dw where w is a vector [x, y, t] (N, 3).
        Returns y_x, y_y, y_t.
        """
        if not y.requires_grad:
            return torch.zeros_like(w[:, 0:1]), torch.zeros_like(w[:, 0:1]), torch.zeros_like(w[:, 0:1])

        grad = torch.autograd.grad(
            y, w,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if grad is None:
            return torch.zeros_like(w[:, 0:1]), torch.zeros_like(w[:, 0:1]), torch.zeros_like(w[:, 0:1])
            
        # grad is (N, 3) -> [dy/dx, dy/dy, dy/dt]
        return grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]

    def gradients(self, y, x):
         """Legacy single gradient (not consistent if x is slice). Avoid using."""
         grad = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]
         return grad


def print_graph_trace(var):
    print("--- COMPUTATION GRAPH TRACE ---")
    fn = var.grad_fn
    while fn is not None:
        print(f" -> {fn}")
        if hasattr(fn, 'next_functions'):
            # Solo seguimos el primer camino por brevedad
            if len(fn.next_functions) > 0 and fn.next_functions[0][0] is not None:
                fn = fn.next_functions[0][0]
            else:
                break
        else:
            break
    print("-------------------------------")


class BlindDiscoveryLoss(PINNLossBase):
    """
    Física de Alta Fidelidad: Navier-Stokes + Transporte.
    Infiere difusión, decaimiento y viscosidad, además de una fuente Q.
    """
    def __init__(self):
        super().__init__()
        # Parámetros Globales Aprendibles
        # Ajustados a los valores del simulador: 
        # smoke_diffusion_rate ~ 0.3 -> kappa ~ 0.18 => log(0.18) ~ -1.7
        # decay_rate = 1.0 => log(1.0) = 0.0
        self.log_D = nn.Parameter(torch.tensor(-1.7))   

    @property
    def D(self): return torch.exp(self.log_D)

    def forward(self, pred_tensor, coords_tensor, latent_z=None):
        stats = {}
        if not pred_tensor.requires_grad:
            return torch.tensor(0.0, device=pred_tensor.device), stats
            
        # print(f"[DEBUG 4] Loss pred_tensor: grad_fn={pred_tensor.grad_fn}")
        # print(f"[DEBUG 5] Loss coords_tensor: requires_grad={coords_tensor.requires_grad}, is_leaf={coords_tensor.is_leaf}")

        # Extraemos salidas
        u = pred_tensor[:, 0:1]
        v = pred_tensor[:, 1:2]
        s = pred_tensor[:, 2:3]
        fu = pred_tensor[:, 3:4]
        fv = pred_tensor[:, 4:5]

        # Derivadas de primer orden
        # Derivadas de primer orden
        u_x, u_y, u_t = self.first_derivatives(u, coords_tensor)
        v_x, v_y, v_t = self.first_derivatives(v, coords_tensor)
        s_x, s_y, s_t = self.first_derivatives(s, coords_tensor)

        s_xx, _, _ = self.first_derivatives(s_x, coords_tensor)
        _, s_yy, _ = self.first_derivatives(s_y, coords_tensor)

        # 1. Navier-Stokes Residuals
        res_u = u_t + (u*u_x + v*u_y) - fu
        res_v = v_t + (u*v_x + v*v_y) - fv
        res_div = u_x + v_y
        
        # 3. Transporte (Advección-Difusión-Reacción)
        advection = (u*s_x + v*s_y)
        diffusion = self.D*(s_xx + s_yy)
        res_s = s_t + advection - diffusion
        
        loss_u_v = torch.mean(res_u**2 + res_v**2)
        loss_div = torch.mean(res_div**2)
        loss_s = torch.mean(res_s**2)
        
        loss_pde = loss_u_v + loss_div + loss_s

        # Stats para debugging
        stats['loss_u_v'] = loss_u_v.item()
        stats['loss_div'] = loss_div.item()
        stats['loss_s'] = loss_s.item()
        stats['res_u_mean'] = res_u.abs().mean().item()
        stats['res_v_mean'] = res_v.abs().mean().item()
        stats['res_s_mean'] = res_s.abs().mean().item()
        stats['res_div_mean'] = res_div.abs().mean().item()
        stats['advection_mean'] = advection.abs().mean().item()
        stats['diffusion_mean'] = diffusion.abs().mean().item()
        stats['laplacian_s_mean'] = (s_xx + s_yy).abs().mean().item()
        stats['s_t_mean'] = s_t.abs().mean().item()

        return loss_pde, stats


