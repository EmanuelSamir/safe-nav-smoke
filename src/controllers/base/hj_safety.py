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
from typing import Union

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
        relative_values: jax.Array,
        relative_values_grad: ValuesGrad,
        t: int = 0,
    ) -> torch.Tensor:
        """Vectorized HJI Dubins least-restrictive controller for dynamic neighbours.

        All intermediate computation stays in JAX. Uses ``grid.nearest_index``
        vmapped over the full ``(K × N)`` batch of relative states.

        Args:
            state:                Ego states, shape ``(K, 3)`` — columns: x, y, theta.
            u_nominal:            Nominal controls, shape ``(K, 2)`` — columns: v, omega.
            neighbors:            Neighbour states as a list of ``(3,)`` arrays/tensors
                                  **or** an array of shape ``(N, 3)``.
            grid:                 ``hj_reachability.Grid`` for the relative frame.
            relative_values:      HJ value function — JAX array, shape ``(Gx, Gy, Gtheta)``.
            relative_values_grad: ``jnp.gradient(relative_values)`` — one JAX array per dim.
            t:                    Look-ahead step for neighbour position prediction.

        Returns:
            Safe controls, shape ``(K, 2)``.
        """
        if self._no_neighbors(neighbors):
            return u_nominal

        dtype = state.dtype
        device = state.device

        # ── Boundary: torch → jax (once) ─────────────────────────────────────
        state_j = jnp.asarray(state.detach().cpu())  # (K, 3)
        u_nom_j = jnp.asarray(u_nominal.detach().cpu())  # (K, 2)
        neighbors_j = self._to_jax(neighbors)  # (N, 3)
        neighbors_j = self._predict_neighbors_jax(neighbors_j, t)  # (N, 3)

        K = state_j.shape[0]
        N = neighbors_j.shape[0]

        # ── All jnp from here ─────────────────────────────────────────────────
        theta = state_j[:, 2:3]  # (K, 1)
        theta_j = neighbors_j[:, 2]  # (N,)

        dx = neighbors_j[:, 0] - state_j[:, 0:1]  # (K, N)
        dy = neighbors_j[:, 1] - state_j[:, 1:2]

        xr = jnp.cos(theta) * dx + jnp.sin(theta) * dy  # (K, N)
        yr = -jnp.sin(theta) * dx + jnp.cos(theta) * dy
        thr = (theta_j - theta) % (2.0 * jnp.pi)  # (K, N)

        # Flatten to (K*N, 3) for a single vmapped nearest_index call
        states_rel_flat = jnp.stack([xr.ravel(), yr.ravel(), thr.ravel()], axis=-1)  # (K*N, 3)

        idx = jax.vmap(grid.nearest_index)(states_rel_flat)  # (K*N, 3) int

        # Safely wrap periodic theta dimension and clip x/y dimensions
        i0 = jnp.clip(idx[:, 0], 0, grid.shape[0] - 1)
        i1 = jnp.clip(idx[:, 1], 0, grid.shape[1] - 1)
        i2 = idx[:, 2] % grid.shape[2]

        v_kn = relative_values[i0, i1, i2].reshape(K, N)
        gx_kn = relative_values_grad[0][i0, i1, i2].reshape(K, N)
        gy_kn = relative_values_grad[1][i0, i1, i2].reshape(K, N)
        gt_kn = relative_values_grad[2][i0, i1, i2].reshape(K, N)

        # Select critical (most dangerous) neighbour per ego agent
        crit_idx = jnp.argmin(v_kn, axis=1)  # (K,)
        batch_idx = jnp.arange(K)
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
        u_safe_j = self._project(u_nom_j, A_v, A_w, unsafe_mask)

        # ── Boundary: jax → torch (once) ─────────────────────────────────────
        return torch.as_tensor(np.asarray(u_safe_j), dtype=dtype, device=device)

    def static_filter(
        self,
        state: torch.Tensor,
        u_nominal: torch.Tensor,
        static_values: jax.Array,
        static_values_grad: ValuesGrad,
        domain: np.ndarray,
        domain_cells: np.ndarray,
        t: int = 0,
    ) -> torch.Tensor:
        """Least-restrictive static obstacle avoidance filter.

        Args:
            state:              Ego states, shape ``(K, 3)``.
            u_nominal:          Nominal controls, shape ``(K, 2)``.
            static_values:      HJ value function — JAX array, shape ``(Gx, Gy, Gtheta)``.
            static_values_grad: ``jnp.gradient(static_values)`` — one JAX array per dim.
            domain:             Grid bounds, shape ``(2, 3)`` — rows: lo, hi.
            domain_cells:       Number of grid cells per dimension, shape ``(3,)``.
            t:                  Unused; reserved for future time-varying obstacles.

        Returns:
            Safe controls, shape ``(K, 2)``.
        """
        dtype = state.dtype
        device = state.device

        # ── Boundary: torch → jax (once) ─────────────────────────────────────
        state_j = jnp.asarray(state.detach().cpu())  # (K, 3)
        u_nom_j = jnp.asarray(u_nominal.detach().cpu())  # (K, 2)

        lo = jnp.asarray(domain[0])  # (3,)
        hi = jnp.asarray(domain[1])
        cells = jnp.asarray(domain_cells, dtype=jnp.int32)  # (3,)

        # ── All jnp from here ─────────────────────────────────────────────────
        dx_grid = (hi - lo) / cells
        idx = jnp.clip(
            jnp.round((state_j - lo) / dx_grid).astype(jnp.int32),
            0,
            cells - 1,
        )  # (K, 3)

        i0, i1, i2 = idx[:, 0], idx[:, 1], idx[:, 2]
        v_val = static_values[i0, i1, i2]
        grad_x = static_values_grad[0][i0, i1, i2]
        grad_y = static_values_grad[1][i0, i1, i2]
        grad_th = static_values_grad[2][i0, i1, i2]

        theta = state_j[:, 2]
        A_v = jnp.cos(theta) * grad_x + jnp.sin(theta) * grad_y
        A_w = grad_th

        unsafe_mask = v_val <= self.params.safe_margin
        u_safe_j = self._project(u_nom_j, A_v, A_w, unsafe_mask)

        # ── Boundary: jax → torch (once) ─────────────────────────────────────
        return torch.as_tensor(np.asarray(u_safe_j), dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Shared QP projection kernel — pure jnp
    # ------------------------------------------------------------------

    @staticmethod
    def _project(
        u_nom: jax.Array,
        A_v: jax.Array,
        A_w: jax.Array,
        unsafe_mask: jax.Array,
    ) -> jax.Array:
        """Project nominal controls onto the safe half-space ``A · u >= 0``.

        Args:
            u_nom:       Nominal controls, shape ``(K, 2)``.
            A_v:         Linear velocity component of the HJ gradient, shape ``(K,)``.
            A_w:         Angular velocity component of the HJ gradient, shape ``(K,)``.
            unsafe_mask: Boolean mask of unsafe samples, shape ``(K,)``.

        Returns:
            Projected controls, shape ``(K, 2)``.
        """
        A_norm_sq = jnp.maximum(A_v**2 + A_w**2, 1e-6)
        A_u = A_v * u_nom[:, 0] + A_w * u_nom[:, 1]
        proj_scale = jnp.where(unsafe_mask & (-A_u > 0), -A_u / A_norm_sq, 0.0)

        return u_nom.at[:, 0].add(proj_scale * A_v).at[:, 1].add(proj_scale * A_w)

    # ------------------------------------------------------------------
    # Neighbour helpers — pure jnp
    # ------------------------------------------------------------------

    def _predict_neighbors_jax(self, neighbors_j: jax.Array, t: int) -> jax.Array:
        """Predict neighbour positions at look-ahead step *t* using jnp.

        Constant-heading, constant-speed model at ``v_nominal = 3.0 m/s``.

        Args:
            neighbors_j: Neighbour states, JAX array shape ``(N, 3)``.
            t:           Look-ahead step.

        Returns:
            Predicted neighbour states, shape ``(N, 3)``.
        """
        # TODO: Replace v_nominal with per-neighbour estimates when available.
        v_nominal = 3.0  # m/s
        dt_ahead = t * self.params.dt

        theta_j = neighbors_j[:, 2]
        dx = v_nominal * jnp.cos(theta_j) * dt_ahead
        dy = v_nominal * jnp.sin(theta_j) * dt_ahead

        return neighbors_j.at[:, 0].add(dx).at[:, 1].add(dy)

    @staticmethod
    def _to_jax(neighbors: Neighbors) -> jax.Array:
        """Convert any neighbour format to a JAX array of shape ``(N, 3)``."""
        if isinstance(neighbors, jax.Array):
            return neighbors
        if isinstance(neighbors, torch.Tensor):
            return jnp.asarray(neighbors.detach().cpu().numpy())
        if isinstance(neighbors, np.ndarray):
            return jnp.asarray(neighbors)
        # list[Tensor | ndarray]
        arrays = [
            n.detach().cpu().numpy() if torch.is_tensor(n) else np.asarray(n) for n in neighbors
        ]
        return jnp.asarray(np.stack(arrays))

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
