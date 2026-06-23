"""Control Barrier Function (CBF) safety filter for multi-agent navigation.

State layout (per agent): [x, y, theta]  — position in meters, heading in radians.
Neighbor tensors share the same layout; shape is (N_neighbors, 3).
"""

from dataclasses import dataclass
from typing import Union

import torch


# Column indices for the state/neighbor tensors
_X, _Y, _THETA = 0, 1, 2


@dataclass
class CBFFilterParams:
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

        Returns a tensor of shape ``(K,)`` where positive values indicate a
        safe configuration and negative values indicate a constraint violation.

        Args:
            state:     Ego-agent states, shape ``(K, 3)`` — columns: x, y, theta.
            neighbors: Neighbour states as a list of ``(3,)`` tensors **or** a
                       pre-stacked tensor of shape ``(N_neighbors, 3)``.
            t:         Look-ahead time step for neighbour position prediction.
        """
        p = self.params
        device = state.device
        K = state.shape[0]

        if self._no_neighbors(neighbors):
            return 1000.0 * torch.ones(K, device=device)

        p_i_safe, p_j_pred, _, _ = self._geometry(state, neighbors, t, device)

        d_safe_barrier = p.d_safe + 2.0 * p.L
        p_rel = p_i_safe.unsqueeze(1) - p_j_pred.unsqueeze(0)  # (K, N, 2)
        dist_sq = torch.sum(p_rel**2, dim=2)                    # (K, N)
        h0 = dist_sq - d_safe_barrier**2

        h_min, _ = torch.min(h0, dim=1)
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
            state:      Ego-agent states, shape ``(K, 3)``.
            u_nominal:  Nominal controls, shape ``(K, 2)`` — columns: v, omega.
            neighbors:  Neighbour states (see :meth:`h_function`).
            action_min: Lower bounds on controls, shape ``(2,)``.
            action_max: Upper bounds on controls, shape ``(2,)``.
            t:          Look-ahead time step for neighbour position prediction.

        Returns:
            Safe controls clamped to ``[action_min, action_max]``, shape ``(K, 2)``.
        """
        if self._no_neighbors(neighbors):
            return u_nominal

        p = self.params
        device = state.device
        K = state.shape[0]

        p_i_safe, p_j_pred, v_j, trig = self._geometry(state, neighbors, t, device)
        cos_t, sin_t = trig  # (K,), (K,)

        # Extra margin in the QP to account for control latency
        d_safe_barrier = p.d_safe + 2.0 * p.L + 0.2
        p_rel = p_i_safe.unsqueeze(1) - p_j_pred.unsqueeze(0)  # (K, N, 2)
        dist_sq = torch.sum(p_rel**2, dim=2)                    # (K, N)
        h0_all = dist_sq - d_safe_barrier**2                    # (K, N)

        # Find the critical (closest) neighbour for each batch item
        crit_idx = torch.argmin(h0_all, dim=1)
        batch_idx = torch.arange(K, device=device)

        h0_crit = h0_all[batch_idx, crit_idx]          # (K,)
        p_rel_crit = p_rel[batch_idx, crit_idx]         # (K, 2)
        v_j_crit = v_j[crit_idx]                        # (K, 2)

        # CBF constraint: A @ u >= B
        # Jacobian of h0 w.r.t. u through the unicycle kinematics:
        #   dh/du_v  = 2 * p_rel · [cos θ,  sin θ]
        #   dh/du_ω  = 2 * p_rel · [-L sin θ, L cos θ]
        A_v = 2.0 * (p_rel_crit[:, 0] * cos_t + p_rel_crit[:, 1] * sin_t)
        A_w = 2.0 * (-p.L * p_rel_crit[:, 0] * sin_t + p.L * p_rel_crit[:, 1] * cos_t)
        A = torch.stack([A_v, A_w], dim=1)  # (K, 2)

        B = 2.0 * torch.sum(p_rel_crit * v_j_crit, dim=1) - p.k1 * h0_crit

        # Projection: u_safe = u_nom + max(0, B - A·u_nom) / ||A||² · A
        A_u_nom = torch.sum(A * u_nominal, dim=1)
        violation = B - A_u_nom
        A_norm_sq = torch.clamp(torch.sum(A**2, dim=1), min=1e-6)
        u_safe = u_nominal + (torch.clamp(violation, min=0.0) / A_norm_sq).unsqueeze(-1) * A

        return torch.max(torch.min(u_safe, action_max), action_min)

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
            p_i_safe:  Ego safety points, shape ``(K, 2)``.
            p_j_pred:  Predicted neighbour safety points at time *t*, shape ``(N, 2)``.
            v_j:       Neighbour velocities, shape ``(N, 2)``.
            trig:      Tuple ``(cos θ, sin θ)`` for the ego agents, each shape ``(K,)``.
        """
        p = self.params

        neighbors_tensor = self._to_tensor(neighbors, device)  # (N, 3)

        theta = state[:, _THETA]                                # (K,)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        p_i_center = state[:, [_X, _Y]]
        p_i_safe = p_i_center + p.L * torch.stack([cos_t, sin_t], dim=1)

        theta_j = neighbors_tensor[:, _THETA]                   # (N,)
        cos_j, sin_j = torch.cos(theta_j), torch.sin(theta_j)

        p_j_center = neighbors_tensor[:, [_X, _Y]]
        p_j_safe = p_j_center + p.L * torch.stack([cos_j, sin_j], dim=1)

        # TODO: Replace v_nominal with a per-neighbour estimate when available.
        v_nominal = 3.0  # m/s — assumed constant for all neighbours
        v_j = v_nominal * torch.stack([cos_j, sin_j], dim=1)

        p_j_pred = p_j_safe + v_j * (t * p.dt)

        return p_i_safe, p_j_pred, v_j, (cos_t, sin_t)

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
# Module-level convenience wrappers (backwards-compatible)
# ---------------------------------------------------------------------------

def cbf_h_function(
    state: torch.Tensor,
    neighbors: Neighbors,
    params: CBFFilterParams,
    t: int = 0,
) -> torch.Tensor:
    """Module-level wrapper around :meth:`CBFFilter.h_function`."""
    return CBFFilter(params).h_function(state, neighbors, t)


def qp_cbf_filter(
    state: torch.Tensor,
    u_nominal: torch.Tensor,
    neighbors: Neighbors,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    params: CBFFilterParams,
    t: int = 0,
) -> torch.Tensor:
    """Module-level wrapper around :meth:`CBFFilter.qp_filter`."""
    return CBFFilter(params).qp_filter(state, u_nominal, neighbors, action_min, action_max, t)


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
