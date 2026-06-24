"""Hamilton-Jacobi (HJ) Reachability safety filters for multi-agent navigation.

Two filters are provided:
* **LRF filter** — least-restrictive controller for dynamic neighbours, operating
  in the *relative* Dubins frame defined by an ``hj_reachability.Grid``.
* **Static filter** — least-restrictive controller for static obstacles, operating
  in the *absolute* state-space frame.

All computation is done in JAX (``jnp``). NumPy is used **only** at the
torch ↔ jax boundary (input conversion and final output).

State layout (per agent): [x, y, theta] — position in metres, heading in radians.
Neighbour tensors share the same layout.
"""

from dataclasses import dataclass
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Each element of the list (or rows of the tensor) has shape (3,): [x, y, theta].
Neighbors = Union[list[Union[torch.Tensor, np.ndarray]], torch.Tensor, np.ndarray]

# Gradient of a value function as returned by jnp.gradient.
ValuesGrad = list  # list[jax.Array], one array per spatial dimension


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class HJFilterParams:
    """Parameters for HJ Reachability / LRF projection.

    Attributes:
        d_safe:       Minimum centre-to-centre clearance distance (m).
        safe_margin:  Value-function threshold below which a state is considered
                      unsafe (typically 0.0 for the zero-sublevel set).
        r_sense:      Sensing radius — neighbours beyond this are ignored (m).
        dt:           Simulation time step (s).
        action_min:   Lower control bounds, shape ``(2,)``.
        action_max:   Upper control bounds, shape ``(2,)``.
    """

    d_safe: float
    safe_margin: float
    r_sense: float
    dt: float
    action_min: torch.Tensor
    action_max: torch.Tensor
    control_type: str = "smooth"  # "smooth" (QP projection) or "bang_bang" (optimal avoidance)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HJFilter:
    """HJ Reachability least-restrictive safety filter backed by JAX.

    Value functions and their gradients live as JAX arrays; all computation
    is done in ``jnp``. The only ``np`` calls are at the torch/jax boundary.

    Args:
        params: HJ filter hyper-parameters.
    """

    def __init__(self, params: HJFilterParams) -> None:
        self.params = params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lrf_filter(
        self,
        state: torch.Tensor,
        u_nominal: torch.Tensor,
        neighbors: Neighbors,
        grid,
        relative_values: Union[jax.Array, np.ndarray, torch.Tensor],
        relative_values_grad: ValuesGrad,
        t: int = 0,
    ) -> torch.Tensor:
        """Vectorized HJI Dubins least-restrictive controller for dynamic neighbours.

        All computation is done purely in PyTorch to avoid massive PyTorch <-> JAX
        memory transfer overheads during the MPPI hot loop. Uses `torch.round` for
        nearest-index lookup over the full `(K × N)` batch of relative states.

        Args:
            state:                Ego states, shape ``(K, 3)`` — columns: x, y, theta.
            u_nominal:            Nominal controls, shape ``(K, 2)`` — columns: v, omega.
            neighbors:            Neighbour states as a list of ``(3,)`` arrays/tensors
                                  **or** an array of shape ``(N, 3)``.
            grid:                 ``hj_reachability.Grid`` for the relative frame.
            relative_values:      HJ value function — shape ``(Gx, Gy, Gtheta)``.
            relative_values_grad: ``jnp.gradient(relative_values)`` — one array per dim.
            t:                    Look-ahead step for neighbour position prediction.

        Returns:
            Safe controls, shape ``(K, 2)``.
        """
        if self._no_neighbors(neighbors):
            return u_nominal

        dtype = state.dtype
        device = state.device

        K = state.shape[0]

        # ── Fast PyTorch operations ──────────────────────────────────────────
        neighbors_t = self._to_torch(neighbors, device, dtype)  # (N, 3)
        neighbors_t = self._predict_neighbors_torch(neighbors_t, t)  # (N, 3)
        N = neighbors_t.shape[0]

        theta = state[:, 2:3]  # (K, 1)
        theta_j = neighbors_t[:, 2]  # (N,)

        dx = neighbors_t[:, 0] - state[:, 0:1]  # (K, N)
        dy = neighbors_t[:, 1] - state[:, 1:2]

        xr = torch.cos(theta) * dx + torch.sin(theta) * dy  # (K, N)
        yr = -torch.sin(theta) * dx + torch.cos(theta) * dy
        thr = (theta_j - theta) % (2.0 * torch.pi)  # (K, N)

        # ── Fast Grid Lookup in PyTorch ──────────────────────────────────────
        lo = torch.tensor(grid.domain.lo, device=device, dtype=dtype)
        hi = torch.tensor(grid.domain.hi, device=device, dtype=dtype)
        cells = torch.tensor(grid.shape, device=device, dtype=torch.long)
        dx_grid = (hi - lo) / cells

        # Flatten to (K*N, 3)
        states_rel_flat = torch.stack([xr.reshape(-1), yr.reshape(-1), thr.reshape(-1)], dim=-1)

        idx = torch.round((states_rel_flat - lo) / dx_grid).long()

        # Safely wrap periodic theta dimension and clip x/y dimensions
        i0 = torch.clamp(idx[:, 0], 0, cells[0] - 1)
        i1 = torch.clamp(idx[:, 1], 0, cells[1] - 1)
        i2 = idx[:, 2] % cells[2]

        # Convert JAX arrays to PyTorch tensors (cached in memory if possible)
        v_grid = self._as_torch(relative_values, device)
        gx_grid = self._as_torch(relative_values_grad[0], device)
        gy_grid = self._as_torch(relative_values_grad[1], device)
        gt_grid = self._as_torch(relative_values_grad[2], device)

        v_kn = v_grid[i0, i1, i2].reshape(K, N)
        gx_kn = gx_grid[i0, i1, i2].reshape(K, N)
        gy_kn = gy_grid[i0, i1, i2].reshape(K, N)
        gt_kn = gt_grid[i0, i1, i2].reshape(K, N)

        # Select critical (most dangerous) neighbour per ego agent
        crit_idx = torch.argmin(v_kn, dim=1)  # (K,)
        batch_idx = torch.arange(K, device=device)

        v_min = v_kn[batch_idx, crit_idx]  # (K,)
        xr_crit = xr[batch_idx, crit_idx]
        yr_crit = yr[batch_idx, crit_idx]
        gx_crit = gx_kn[batch_idx, crit_idx]
        gy_crit = gy_kn[batch_idx, crit_idx]
        gt_crit = gt_kn[batch_idx, crit_idx]

        # Hamiltonian linearisation: V_dot = A_v * v + A_w * omega >= 0
        A_v = -gx_crit
        A_w = yr_crit * gx_crit - xr_crit * gy_crit - gt_crit

        unsafe_mask = v_min <= self.params.safe_margin
        u_safe = self._project(
            u_nominal,
            A_v,
            A_w,
            unsafe_mask,
            control_type=self.params.control_type,
            action_min=self.params.action_min,
            action_max=self.params.action_max,
        )

        return u_safe

    def static_filter(
        self,
        state: torch.Tensor,
        u_nominal: torch.Tensor,
        static_values: Union[jax.Array, np.ndarray, torch.Tensor],
        static_values_grad: ValuesGrad,
        domain: np.ndarray,
        domain_cells: np.ndarray,
        t: int = 0,
    ) -> torch.Tensor:
        """Least-restrictive static obstacle avoidance filter in pure PyTorch.

        Args:
            state:              Ego states, shape ``(K, 3)``.
            u_nominal:          Nominal controls, shape ``(K, 2)``.
            static_values:      HJ value function — shape ``(Gx, Gy, Gtheta)``.
            static_values_grad: Gradients — one array per dim.
            domain:             Grid bounds, shape ``(2, 3)`` — rows: lo, hi.
            domain_cells:       Number of grid cells per dimension, shape ``(3,)``.
            t:                  Unused; reserved for future time-varying obstacles.

        Returns:
            Safe controls, shape ``(K, 2)``.
        """
        dtype = state.dtype
        device = state.device

        lo = torch.tensor(domain[0], device=device, dtype=dtype)
        hi = torch.tensor(domain[1], device=device, dtype=dtype)
        cells = torch.tensor(domain_cells, device=device, dtype=torch.long)

        dx_grid = (hi - lo) / cells
        idx = torch.round((state - lo) / dx_grid).long()
        idx = torch.clamp(idx, 0, cells - 1)

        i0, i1, i2 = idx[:, 0], idx[:, 1], idx[:, 2]

        v_grid = self._as_torch(static_values, device)
        gx_grid = self._as_torch(static_values_grad[0], device)
        gy_grid = self._as_torch(static_values_grad[1], device)
        gt_grid = self._as_torch(static_values_grad[2], device)

        v_val = v_grid[i0, i1, i2]
        grad_x = gx_grid[i0, i1, i2]
        grad_y = gy_grid[i0, i1, i2]
        grad_th = gt_grid[i0, i1, i2]

        theta = state[:, 2]
        A_v = torch.cos(theta) * grad_x + torch.sin(theta) * grad_y
        A_w = grad_th

        unsafe_mask = v_val <= self.params.safe_margin
        return self._project(
            u_nominal,
            A_v,
            A_w,
            unsafe_mask,
            control_type=self.params.control_type,
            action_min=self.params.action_min,
            action_max=self.params.action_max,
        )

    # ------------------------------------------------------------------
    # Shared QP projection kernel — pure PyTorch
    # ------------------------------------------------------------------

    @staticmethod
    def _project(
        u_nom: torch.Tensor,
        A_v: torch.Tensor,
        A_w: torch.Tensor,
        unsafe_mask: torch.Tensor,
        control_type: str = "smooth",
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project nominal controls onto the safe half-space ``A · u >= 0``.

        Args:
            u_nom:        Nominal controls, shape ``(K, 2)``.
            A_v:          Hamiltonian coefficient for linear velocity.
            A_w:          Hamiltonian coefficient for angular velocity.
            unsafe_mask:  Boolean mask of unsafe states, shape ``(K,)``.
            control_type: "smooth" (QP projection) or "bang_bang" (optimal avoidance).
            action_min:   Lower control bounds (required for bang_bang).
            action_max:   Upper control bounds (required for bang_bang).
        """
        if control_type == "bang_bang":
            assert action_min is not None and action_max is not None
            v_star = torch.where(A_v > 0, action_max[0], action_min[0])
            w_star = torch.where(A_w > 0, action_max[1], action_min[1])
            u_bang = torch.stack([v_star, w_star], dim=1)
            # Override completely with optimal avoidance if unsafe
            return torch.where(unsafe_mask.unsqueeze(-1), u_bang, u_nom)

        A_norm_sq = torch.clamp(A_v**2 + A_w**2, min=1e-6)
        A_u = A_v * u_nom[:, 0] + A_w * u_nom[:, 1]
        proj_scale = torch.where(unsafe_mask & (-A_u > 0), -A_u / A_norm_sq, torch.zeros_like(A_u))

        u_safe = u_nom.clone()
        u_safe[:, 0] += proj_scale * A_v
        u_safe[:, 1] += proj_scale * A_w
        return u_safe

    # ------------------------------------------------------------------
    # Neighbour helpers — pure PyTorch
    # ------------------------------------------------------------------

    def _predict_neighbors_torch(self, neighbors_t: torch.Tensor, t: int) -> torch.Tensor:
        """Predict neighbour positions at look-ahead step *t* using pure PyTorch.

        Constant-heading, constant-speed model at ``v_nominal = 3.0 m/s``.

        Args:
            neighbors_t: Neighbour states, PyTorch tensor shape ``(N, 3)``.
            t:           Look-ahead step.

        Returns:
            Predicted neighbour states, shape ``(N, 3)``.
        """
        # TODO: Replace v_nominal with per-neighbour estimates when available.
        v_nominal = 3.0  # m/s
        dt_ahead = t * self.params.dt

        theta_j = neighbors_t[:, 2]
        dx = v_nominal * torch.cos(theta_j) * dt_ahead
        dy = v_nominal * torch.sin(theta_j) * dt_ahead

        out = neighbors_t.clone()
        out[:, 0] += dx
        out[:, 1] += dy
        return out

    @staticmethod
    def _to_torch(neighbors: Neighbors, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Convert any neighbour format to a PyTorch tensor of shape ``(N, 3)``."""
        if isinstance(neighbors, torch.Tensor):
            return neighbors.to(device=device, dtype=dtype)
        if isinstance(neighbors, np.ndarray):
            return torch.tensor(neighbors, device=device, dtype=dtype)
        if hasattr(neighbors, "device"):  # jax.Array
            return torch.tensor(np.asarray(neighbors), device=device, dtype=dtype)

        arrays = [
            n if torch.is_tensor(n) else torch.tensor(np.asarray(n), device=device, dtype=dtype)
            for n in neighbors
        ]
        if len(arrays) == 0:
            return torch.empty((0, 3), device=device, dtype=dtype)
        return torch.stack(arrays).to(device=device, dtype=dtype)

    @staticmethod
    def _as_torch(
        array: Union[jax.Array, np.ndarray, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """Helper to cast value functions to Torch and send to device once."""
        if isinstance(array, torch.Tensor):
            return array.to(device)
        return torch.tensor(np.asarray(array), device=device)

    @staticmethod
    def _no_neighbors(neighbors: Neighbors) -> bool:
        if isinstance(neighbors, (torch.Tensor, np.ndarray, jax.Array)):
            return neighbors.shape[0] == 0
        return len(neighbors) == 0


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (backwards-compatible)
# ---------------------------------------------------------------------------


def hj_lrf_filter(
    state: torch.Tensor,
    u_nominal: torch.Tensor,
    neighbors: Neighbors,
    grid,
    relative_values: jax.Array,
    relative_values_grad: ValuesGrad,
    params: HJFilterParams,
    t: int = 0,
) -> torch.Tensor:
    """Module-level wrapper around :meth:`HJFilter.lrf_filter`."""
    return HJFilter(params).lrf_filter(
        state, u_nominal, neighbors, grid, relative_values, relative_values_grad, t
    )


def hj_static_filter(
    state: torch.Tensor,
    u_nominal: torch.Tensor,
    static_values: jax.Array,
    static_values_grad: ValuesGrad,
    domain: np.ndarray,
    domain_cells: np.ndarray,
    params: HJFilterParams,
    t: int = 0,
) -> torch.Tensor:
    """Module-level wrapper around :meth:`HJFilter.static_filter`."""
    return HJFilter(params).static_filter(
        state, u_nominal, static_values, static_values_grad, domain, domain_cells, t
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid_shape = (40, 40, 63)
    lo = jnp.array([-10.0, -10.0, 0.0])
    hi = jnp.array([10.0, 10.0, 2 * jnp.pi])
    dx = (hi - lo) / jnp.array(grid_shape)

    class DummyGrid:
        """Minimal stand-in for hj_reachability.Grid."""

        _lo, _dx = lo, dx
        _cells = jnp.array(grid_shape) - 1

        def nearest_index(self, state):
            return jnp.clip(
                jnp.round((state - self._lo) / self._dx).astype(jnp.int32),
                0,
                self._cells,
            )

    grid = DummyGrid()
    relative_values = jnp.zeros(grid_shape)
    relative_values_grad = list(jnp.gradient(relative_values))

    state = torch.tensor([[0.0, 0.0, 0.0]])
    u_nominal = torch.tensor([[1.0, 0.0]])
    neighbors = [torch.tensor([0.5, 0.0, 0.0])]
    params = HJFilterParams(
        d_safe=1.0,
        safe_margin=0.0,
        r_sense=5.0,
        dt=0.1,
        action_min=torch.tensor([-2.0, -2.0]),
        action_max=torch.tensor([2.0, 2.0]),
    )

    hj = HJFilter(params)
    u_safe = hj.lrf_filter(state, u_nominal, neighbors, grid, relative_values, relative_values_grad)
    print("Test passed. HJ Safe Action:", u_safe)
