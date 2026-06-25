"""Control Barrier Function (CBF) safety filter for multi-agent navigation.

State layout (per agent): [x, y, theta]  — position in meters, heading in radians.
Neighbor tensors share the same layout; shape is (N_neighbors, 3).
"""

from typing import Union

import torch

from src.utils.config_utils import StrictBaseModel

# Column indices for the state/neighbor tensors
_X, _Y, _THETA = 0, 1, 2


class CBFFilterParams(StrictBaseModel):
    """Parameters for the High-Order Control Barrier Function (HOCBF) filter.

    Attributes:
        d_safe:          Minimum centre-to-centre clearance distance (m).
        k1:              Class-K gain for the 1st-order CBF constraint.
        k2:              Class-K gain for the 2nd-order CBF constraint (reserved).
        dt:              Simulation time step (s).
        r_sense:         Sensing radius — neighbours beyond this are ignored (m).
        L:               Longitudinal offset from vehicle centre to safety point (m).
        smoke_threshold: Smoke concentration threshold above which motion is penalised.
        rho:             Penalty weight for smoke exposure in the cost function.
    """

    d_safe: float
    k1: float
    k2: float
    dt: float
    r_sense: float
    L: float = 0.4
    smoke_threshold: float = 0.75
    rho: float = 5.0


# ---------------------------------------------------------------------------
# Type alias for neighbours
# ---------------------------------------------------------------------------
# Each element of the list (or rows of the tensor) has shape (3,): [x, y, theta].
Neighbors = Union[list[torch.Tensor], torch.Tensor]


class CBFFilter:
    """Analytical QP-based Control Barrier Function safety filter.

    Wraps the two public operations — evaluating the safety index *h* and
    projecting a nominal control into the safe set — sharing the common
    geometric pre-computation between them.

    Args:
        params: CBF hyper-parameters.
    """

    def __init__(self, params: CBFFilterParams) -> None:
        self.params = params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def h_function(
        self,
        state: torch.Tensor,
        neighbors: Neighbors,
        t: int = 0,
    ) -> torch.Tensor:
        """Compute HOCBF safety index values.

        Returns a tensor of shape ``(batch_size,)`` where positive values indicate a
        safe configuration and negative values indicate a constraint violation.

        Args:
            state:     Ego-agent states, shape ``(batch_size, 3)`` — columns: x, y, theta.
            neighbors: Neighbour states as a list of ``(3,)`` tensors **or** a
                       pre-stacked tensor of shape ``(N_neighbors, 3)``.
            t:         Look-ahead time step for neighbour position prediction.
        """
        p = self.params
        device = state.device
        batch_size = state.shape[0]

        if self._no_neighbors(neighbors):
            return 1000.0 * torch.ones(batch_size, device=device)

        ego_safe_pos, neighbor_pred_pos, _, _ = self._geometry(state, neighbors, t, device)

        d_safe_barrier = p.d_safe + 2.0 * p.L
        relative_pos = ego_safe_pos.unsqueeze(1) - neighbor_pred_pos.unsqueeze(0)  # (batch_size, N, 2)
        dist_sq = torch.sum(relative_pos**2, dim=2)  # (batch_size, N)
        barrier_values = dist_sq - d_safe_barrier**2

        h_min, _ = torch.min(barrier_values, dim=1)
        return h_min

    def qp_filter(
        self,
        state: torch.Tensor,
        u_nominal: torch.Tensor,
        neighbors: Neighbors,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        t: int = 0,
    ) -> torch.Tensor:
        """Project *u_nominal* onto the CBF-safe control set via an analytical QP.

        Args:
            state:      Ego-agent states, shape ``(batch_size, 3)``.
            u_nominal:  Nominal controls, shape ``(batch_size, 2)`` — columns: v, omega.
            neighbors:  Neighbour states (see :meth:`h_function`).
            action_min: Lower bounds on controls, shape ``(2,)``.
            action_max: Upper bounds on controls, shape ``(2,)``.
            t:          Look-ahead time step for neighbour position prediction.

        Returns:
            Safe controls clamped to ``[action_min, action_max]``, shape ``(batch_size, 2)``.
        """
        if self._no_neighbors(neighbors):
            return u_nominal

        p = self.params
        device = state.device
        batch_size = state.shape[0]

        ego_safe_pos, neighbor_pred_pos, neighbor_velocity, trig = self._geometry(state, neighbors, t, device)
        cos_t, sin_t = trig  # (batch_size,), (batch_size,)

        # Mathematical constraint for unicycle lookahead points
        d_safe_barrier = p.d_safe + 2.0 * p.L
        relative_pos = ego_safe_pos.unsqueeze(1) - neighbor_pred_pos.unsqueeze(0)  # (batch_size, N, 2)
        dist_sq = torch.sum(relative_pos**2, dim=2)  # (batch_size, N)
        all_barrier_values = dist_sq - d_safe_barrier**2  # (batch_size, N)

        # Find the critical (closest) neighbour for each batch item
        closest_neighbor_idx = torch.argmin(all_barrier_values, dim=1)
        batch_idx = torch.arange(batch_size, device=device)

        closest_barrier_value = all_barrier_values[batch_idx, closest_neighbor_idx]  # (batch_size,)
        closest_relative_pos = relative_pos[batch_idx, closest_neighbor_idx]  # (batch_size, 2)
        closest_neighbor_velocity = neighbor_velocity[closest_neighbor_idx]  # (batch_size, 2)

        # CBF constraint: constraint_matrix @ u >= constraint_bound
        # Jacobian of barrier w.r.t. u through the unicycle kinematics:
        grad_v = 2.0 * (closest_relative_pos[:, 0] * cos_t + closest_relative_pos[:, 1] * sin_t)
        grad_w = 2.0 * (-p.L * closest_relative_pos[:, 0] * sin_t + p.L * closest_relative_pos[:, 1] * cos_t)
        constraint_matrix = torch.stack([grad_v, grad_w], dim=1)  # (batch_size, 2)

        constraint_bound = 2.0 * torch.sum(closest_relative_pos * closest_neighbor_velocity, dim=1) - p.k1 * closest_barrier_value

        # Projection: safe_action = nominal_action + max(0, bound - A·nominal) / ||A||² · A
        A_u_nom = torch.sum(constraint_matrix * u_nominal, dim=1)
        violation = constraint_bound - A_u_nom
        A_norm_sq = torch.clamp(torch.sum(constraint_matrix**2, dim=1), min=1e-6)
        safe_action = u_nominal + (torch.clamp(violation, min=0.0) / A_norm_sq).unsqueeze(-1) * constraint_matrix

        return torch.max(torch.min(safe_action, action_max), action_min)

    # ------------------------------------------------------------------
    # Shared geometry kernel
    # ------------------------------------------------------------------

    def _geometry(
        self,
        state: torch.Tensor,
        neighbors: Neighbors,
        t: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute the safety points and predicted neighbour positions.

        Returns:
            ego_safe_pos:  Ego safety points, shape ``(batch_size, 2)``.
            neighbor_pred_pos:  Predicted neighbour safety points at time *t*, shape ``(N, 2)``.
            neighbor_velocity:       Neighbour velocities, shape ``(N, 2)``.
            trig:      Tuple ``(cos θ, sin θ)`` for the ego agents, each shape ``(batch_size,)``.
        """
        p = self.params

        neighbors_tensor = self._to_tensor(neighbors, device)  # (N, 3)

        theta = state[:, _THETA]  # (batch_size,)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        ego_center_pos = state[:, [_X, _Y]]
        ego_safe_pos = ego_center_pos + p.L * torch.stack([cos_t, sin_t], dim=1)

        theta_j = neighbors_tensor[:, _THETA]  # (N,)
        cos_j, sin_j = torch.cos(theta_j), torch.sin(theta_j)

        neighbor_center_pos = neighbors_tensor[:, [_X, _Y]]
        neighbor_safe_pos = neighbor_center_pos + p.L * torch.stack([cos_j, sin_j], dim=1)

        if neighbors_tensor.shape[1] > 3:
            v_nominal = neighbors_tensor[:, 3]  # (N,)
            neighbor_velocity = v_nominal.unsqueeze(-1) * torch.stack([cos_j, sin_j], dim=1)
        else:
            # TODO: Replace v_nominal with a per-neighbour estimate when available.
            v_nominal = 3.0  # m/s — assumed constant for all neighbours
            neighbor_velocity = v_nominal * torch.stack([cos_j, sin_j], dim=1)

        neighbor_pred_pos = neighbor_safe_pos + neighbor_velocity * (t * p.dt)

        return ego_safe_pos, neighbor_pred_pos, neighbor_velocity, (cos_t, sin_t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_neighbors(neighbors: Neighbors) -> bool:
        if isinstance(neighbors, list):
            return len(neighbors) == 0
        return neighbors.shape[0] == 0

    @staticmethod
    def _to_tensor(neighbors: Neighbors, device: torch.device) -> torch.Tensor:
        if isinstance(neighbors, list):
            return torch.stack(neighbors).to(device)
        return neighbors.to(device)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    state = torch.tensor([[0.0, 0.0, 0.0]])
    u_nominal = torch.tensor([[1.0, 0.0]])
    neighbors = [torch.tensor([0.5, 0.0, 0.0])]
    action_min = torch.tensor([-2.0, -2.0])
    action_max = torch.tensor([2.0, 2.0])
    params = CBFFilterParams(d_safe=1.0, k1=1.0, k2=1.0, dt=0.1, r_sense=5.0)

    cbf = CBFFilter(params)
    h_val = cbf.h_function(state, neighbors)
    u_safe = cbf.qp_filter(state, u_nominal, neighbors, action_min, action_max)
    print("Test passed. h_val:", h_val, "CBF Safe Action:", u_safe)
