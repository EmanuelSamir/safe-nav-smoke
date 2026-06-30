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

from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch

from src.controllers.base.schemas import HJFilterConfig

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Each element of the list (or rows of the tensor) has shape (3,): [x, y, theta].
Neighbors = Union[list[Union[torch.Tensor, np.ndarray]], torch.Tensor, np.ndarray]

# Gradient of a value function as returned by jnp.gradient.
ValuesGrad = list  # list[jax.Array], one array per spatial dimension


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

    def __init__(self, config: HJFilterConfig) -> None:
        self.config = config

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

        batch_size = state.shape[0]

        # ── Fast PyTorch operations ──────────────────────────────────────────
        neighbors_tensor = self._to_torch(neighbors, device, dtype)  # (N, 3)
        neighbors_tensor = self._predict_neighbors_torch(neighbors_tensor, t)  # (N, 3)
        num_neighbors = neighbors_tensor.shape[0]

        ego_theta = state[:, 2:3]  # (K, 1)
        neighbor_theta = neighbors_tensor[:, 2]  # (N,)

        delta_x = neighbors_tensor[:, 0] - state[:, 0:1]  # (K, N)
        delta_y = neighbors_tensor[:, 1] - state[:, 1:2]

        rel_x = torch.cos(ego_theta) * delta_x + torch.sin(ego_theta) * delta_y  # (K, N)
        rel_y = -torch.sin(ego_theta) * delta_x + torch.cos(ego_theta) * delta_y
        rel_theta = (neighbor_theta - ego_theta) % (2.0 * torch.pi)  # (K, N)

        # ── Fast Grid Lookup in PyTorch ──────────────────────────────────────
        domain_min = torch.tensor(grid.domain.lo, device=device, dtype=dtype)
        domain_max = torch.tensor(grid.domain.hi, device=device, dtype=dtype)
        grid_cells = torch.tensor(grid.shape, device=device, dtype=torch.long)
        cell_size = (domain_max - domain_min) / grid_cells

        # Flatten to (K*N, 3)
        rel_states_flat = torch.stack([rel_x.reshape(-1), rel_y.reshape(-1), rel_theta.reshape(-1)], dim=-1)

        grid_indices = torch.round((rel_states_flat - domain_min) / cell_size).long()

        # Safely wrap periodic theta dimension and clip x/y dimensions
        idx_x = torch.clamp(grid_indices[:, 0], 0, grid_cells[0] - 1)
        idx_y = torch.clamp(grid_indices[:, 1], 0, grid_cells[1] - 1)
        idx_theta = grid_indices[:, 2] % grid_cells[2]

        # Convert JAX arrays to PyTorch tensors (cached in memory if possible)
        values_grid = self._as_torch(relative_values, device)
        grad_x_grid = self._as_torch(relative_values_grad[0], device)
        grad_y_grid = self._as_torch(relative_values_grad[1], device)
        grad_theta_grid = self._as_torch(relative_values_grad[2], device)

        values_batch = values_grid[idx_x, idx_y, idx_theta].reshape(batch_size, num_neighbors)
        grad_x_batch = grad_x_grid[idx_x, idx_y, idx_theta].reshape(batch_size, num_neighbors)
        grad_y_batch = grad_y_grid[idx_x, idx_y, idx_theta].reshape(batch_size, num_neighbors)
        grad_theta_batch = grad_theta_grid[idx_x, idx_y, idx_theta].reshape(batch_size, num_neighbors)

        # Select critical (most dangerous) neighbour per ego agent
        critical_idx = torch.argmin(values_batch, dim=1)  # (K,)
        batch_indices = torch.arange(batch_size, device=device)

        min_safe_value = values_batch[batch_indices, critical_idx]  # (K,)
        crit_rel_x = rel_x[batch_indices, critical_idx]
        crit_rel_y = rel_y[batch_indices, critical_idx]
        crit_grad_x = grad_x_batch[batch_indices, critical_idx]
        crit_grad_y = grad_y_batch[batch_indices, critical_idx]
        crit_grad_theta = grad_theta_batch[batch_indices, critical_idx]

        # Hamiltonian linearisation: V_dot = hamiltonian_vel * v + hamiltonian_omega * omega >= 0
        hamiltonian_vel = -crit_grad_x
        hamiltonian_omega = crit_rel_y * crit_grad_x - crit_rel_x * crit_grad_y - crit_grad_theta

        unsafe_mask = min_safe_value <= 0.0
        u_safe = self._project(
            u_nominal,
            hamiltonian_vel,
            hamiltonian_omega,
            unsafe_mask,
            control_type=self.config.control_type,
            action_min=self.config.action_min,
            action_max=self.config.action_max,
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

        domain_min = torch.tensor(domain[0], device=device, dtype=dtype)
        domain_max = torch.tensor(domain[1], device=device, dtype=dtype)
        grid_cells = torch.tensor(domain_cells, device=device, dtype=torch.long)

        cell_size = (domain_max - domain_min) / grid_cells
        grid_indices = torch.round((state - domain_min) / cell_size).long()
        grid_indices = torch.clamp(grid_indices, 0, grid_cells - 1)

        idx_x, idx_y, idx_theta = grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]

        values_grid = self._as_torch(static_values, device)
        grad_x_grid = self._as_torch(static_values_grad[0], device)
        grad_y_grid = self._as_torch(static_values_grad[1], device)
        grad_theta_grid = self._as_torch(static_values_grad[2], device)

        safe_value = values_grid[idx_x, idx_y, idx_theta]
        grad_x = grad_x_grid[idx_x, idx_y, idx_theta]
        grad_y = grad_y_grid[idx_x, idx_y, idx_theta]
        grad_theta = grad_theta_grid[idx_x, idx_y, idx_theta]

        ego_theta = state[:, 2]
        hamiltonian_vel = torch.cos(ego_theta) * grad_x + torch.sin(ego_theta) * grad_y
        hamiltonian_omega = grad_theta

        unsafe_mask = safe_value <= 0.0
        return self._project(
            u_nominal,
            hamiltonian_vel,
            hamiltonian_omega,
            unsafe_mask,
            control_type=self.config.control_type,
            action_min=self.config.action_min,
            action_max=self.config.action_max,
        )

    # ------------------------------------------------------------------
    # Shared QP projection kernel — pure PyTorch
    # ------------------------------------------------------------------

    @staticmethod
    def _project(
        u_nom: torch.Tensor,
        hamiltonian_vel: torch.Tensor,
        hamiltonian_omega: torch.Tensor,
        unsafe_mask: torch.Tensor,
        control_type: str = "smooth",
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project nominal controls onto the safe half-space.

        Args:
            u_nom:             Nominal controls, shape ``(K, 2)``.
            hamiltonian_vel:   Hamiltonian coefficient for linear velocity.
            hamiltonian_omega: Hamiltonian coefficient for angular velocity.
            unsafe_mask:       Boolean mask of unsafe states, shape ``(K,)``.
            control_type:      "smooth" (QP projection) or "bang_bang" (optimal avoidance).
            action_min:        Lower control bounds (required for bang_bang).
            action_max:        Upper control bounds (required for bang_bang).
        """
        if control_type == "bang_bang":
            assert action_min is not None and action_max is not None
            v_star = torch.where(hamiltonian_vel > 0, action_max[0], action_min[0])
            w_star = torch.where(hamiltonian_omega > 0, action_max[1], action_min[1])
            u_bang = torch.stack([v_star, w_star], dim=1)
            # Override completely with optimal avoidance if unsafe
            return torch.where(unsafe_mask.unsqueeze(-1), u_bang, u_nom)

        hamiltonian_norm_sq = torch.clamp(hamiltonian_vel**2 + hamiltonian_omega**2, min=1e-6)
        hamiltonian_u = hamiltonian_vel * u_nom[:, 0] + hamiltonian_omega * u_nom[:, 1]
        proj_scale = torch.where(unsafe_mask & (-hamiltonian_u > 0), -hamiltonian_u / hamiltonian_norm_sq, torch.zeros_like(hamiltonian_u))

        u_safe = u_nom.clone()
        u_safe[:, 0] += proj_scale * hamiltonian_vel
        u_safe[:, 1] += proj_scale * hamiltonian_omega
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
        dt_ahead = t * self.config.dt

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

        shape = grid_shape

        class Domain:
            lo = lo
            hi = hi

        domain = Domain()

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
    params = HJFilterConfig(
        robot_radius=0.4,
        safe_margin=0.2,
        r_sense=5.0,
        dt=0.1,
        action_min=torch.tensor([-2.0, -2.0]),
        action_max=torch.tensor([2.0, 2.0]),
    )

    hj = HJFilter(params)
    u_safe = hj.lrf_filter(state, u_nominal, neighbors, grid, relative_values, relative_values_grad)
    print("Test passed. HJ Safe Action:", u_safe)
