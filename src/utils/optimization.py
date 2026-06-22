import torch


def solve_qp_batch_pytorch(
    u_nom: torch.Tensor,
    R_diag: torch.Tensor,
    A: torch.Tensor,
    C: torch.Tensor,
    u_min: torch.Tensor,
    u_max: torch.Tensor,
    max_iters: int = 20,
    rho: float = 10.0,
) -> torch.Tensor:
    """Solves a batch of QPs in parallel using PyTorch vectorization.

    For N=2 (Dubins), it runs a mathematically exact closed-form analytical solver
    in a single step (zero iterations). For other dimensions, it falls back to ADMM.

    min 1/2 (u - u_nom)^T R (u - u_nom)
    s.t. A u + C >= 0, u_min <= u <= u_max
    """
    # Handle 1D input tensors (single QP call) seamlessly by unsqueezing to batch size 1
    is_1d = u_nom.ndim == 1
    if is_1d:
        u_nom = u_nom.unsqueeze(0)
        A = A.unsqueeze(0)
        C = C.unsqueeze(0) if C.ndim == 0 else C.reshape(1)

    K, N = u_nom.shape
    device = u_nom.device
    dtype = u_nom.dtype

    if N == 2:
        # High-performance closed-form analytical batch solver for 2D control spaces (e.g. Dubins)
        # Bypasses expensive iterative ADMM loops entirely (zero iterations, mathematically exact)
        R_v, R_w = R_diag[0], R_diag[1]
        v_nom, w_nom = u_nom[:, 0], u_nom[:, 1]
        v_min, w_min = u_min[0], u_min[1]
        v_max, w_max = u_max[0], u_max[1]
        A_v, A_w = A[:, 0], A[:, 1]

        # 1. Fast path: check safe mask under nominal control
        safe_mask = A_v * v_nom + A_w * w_nom + C >= 0
        v_safe = torch.clamp(v_nom, v_min, v_max)
        w_safe = torch.clamp(w_nom, w_min, w_max)

        # 2. Unsafe path: constraint is active A_v * v + A_w * w + C = 0
        eps = 1e-9

        # Case A: A_w is non-zero (steering has control authority)
        A_w_sgn = torch.where(A_w >= 0, torch.clamp(A_w, min=eps), torch.clamp(A_w, max=-eps))
        alpha = -A_v / A_w_sgn
        beta = -C / A_w_sgn

        # Unconstrained quadratic minimizer along the active line
        denom = R_v + R_w * alpha**2
        num = R_v * v_nom - R_w * alpha * (beta - w_nom)
        v_unconstrained = num / denom

        # Intersect bounds: v in [v_min, v_max] and w(v) in [w_min, w_max]
        val1 = (w_min - beta) / torch.where(
            torch.abs(alpha) >= eps, alpha, torch.ones_like(alpha) * eps
        )
        val2 = (w_max - beta) / torch.where(
            torch.abs(alpha) >= eps, alpha, torch.ones_like(alpha) * eps
        )
        v_w_min = torch.minimum(val1, val2)
        v_w_max = torch.maximum(val1, val2)

        # Handle alpha close to zero to prevent restricting v
        alpha_zero = torch.abs(alpha) < eps
        v_w_min = torch.where(alpha_zero, v_min, v_w_min)
        v_w_max = torch.where(alpha_zero, v_max, v_w_max)

        v_feas_min = torch.maximum(v_min, v_w_min)
        v_feas_max = torch.minimum(v_max, v_w_max)

        # Detect physical infeasibility (no intersection between safe set and motor box)
        infeas = v_feas_min > v_feas_max
        v_feas_min_clamped = torch.where(infeas, v_min, v_feas_min)
        v_feas_max_clamped = torch.where(infeas, v_max, v_feas_max)

        v_active = torch.clamp(v_unconstrained, min=v_feas_min_clamped, max=v_feas_max_clamped)
        v_active = torch.where(infeas, v_min, v_active)  # Fallback to full braking
        w_active = alpha * v_active + beta
        w_active = torch.where(
            infeas, torch.clamp(w_nom, w_min, w_max), w_active
        )  # Fallback steering

        # Case B: A_w is zero (steering has zero control authority on the barrier)
        A_v_sgn = torch.where(A_v >= 0, torch.clamp(A_v, min=eps), torch.clamp(A_v, max=-eps))
        v_bound = -C / A_v_sgn

        v_bound_min = torch.where(A_v > 0, v_bound, v_min)
        v_bound_max = torch.where(A_v < 0, v_bound, v_max)

        v_zero_feas_min = torch.maximum(v_min, v_bound_min)
        v_zero_feas_max = torch.minimum(v_max, v_bound_max)

        infeas_zero = (v_zero_feas_min > v_zero_feas_max) | ((torch.abs(A_v) < eps) & (C < 0))
        v_zero_feas_min_clamped = torch.where(infeas_zero, v_min, v_zero_feas_min)
        v_zero_feas_max_clamped = torch.where(infeas_zero, v_max, v_zero_feas_max)

        v_zero = torch.clamp(v_nom, min=v_zero_feas_min_clamped, max=v_zero_feas_max_clamped)
        v_zero = torch.where(infeas_zero, v_min, v_zero)
        w_zero = torch.clamp(w_nom, w_min, w_max)

        # Merge cases based on A_w being close to zero
        is_aw_zero = torch.abs(A_w) < eps
        v_unsafe = torch.where(is_aw_zero, v_zero, v_active)
        w_unsafe = torch.where(is_aw_zero, w_zero, w_active)

        # Merge safe and unsafe items
        v_final = torch.where(safe_mask, v_safe, v_unsafe)
        w_final = torch.where(safe_mask, w_safe, w_unsafe)

        u = torch.stack([v_final, w_final], dim=1)

    else:
        # Fallback to general iterative ADMM solver for higher control dimensions
        u = u_nom.clone()
        z = torch.clamp(torch.sum(A * u, dim=1) + C, min=0.0)  # slack variable of shape (K,)
        y = torch.zeros(K, device=device, dtype=dtype)  # dual variable of shape (K,)

        # Precompute ADMM system matrix inversion components
        R_mat = torch.diag(R_diag).unsqueeze(0).repeat(K, 1, 1)
        A_uns = A.unsqueeze(-1)  # (K, N, 1)
        A_A_T = torch.bmm(A_uns, A_uns.transpose(1, 2))  # (K, N, N)
        M = R_mat + rho * A_A_T  # (K, N, N)

        for _ in range(max_iters):
            # 1. Update u: solve M_k * u_k = RHS_k
            temp = rho * (z - C - y)  # (K,)
            rhs = R_diag * u_nom + A * temp.unsqueeze(-1)  # (K, N)

            # Batch linear solver
            u = torch.linalg.solve(M, rhs)  # (K, N)
            u = torch.clamp(u, u_min, u_max)

            # 2. Update slack variable z
            A_u = torch.sum(A * u, dim=1)
            z = torch.clamp(A_u + C + y, min=0.0)

            # 3. Update dual variable y
            y = y + A_u + C - z

    if is_1d:
        u = u.squeeze(0)

    return u
