from typing import Callable

import torch


class DualGuardShield:
    """Shielding function that wraps a safety function and a safe control function.

    Can be used directly as a rollout_filter_fn or output_filter_fn in MPPI.
    """

    def __init__(
        self,
        safety_function: Callable,
        safe_control_function: Callable,
        safe_margin: float = 0.0,
    ):
        """Initialize the DualGuardShield."""
        self.safety_function = safety_function
        self.safe_control_function = safe_control_function
        self.safe_margin = safe_margin

    def __call__(self, state: torch.Tensor, u_nominal: torch.Tensor, t: int = 0) -> torch.Tensor:
        try:
            safety_val = self.safety_function(state, t)
        except TypeError:
            safety_val = self.safety_function(state)

        if safety_val.dim() > 1:
            safety_val = safety_val.squeeze(-1)

        # Compute boolean mask where state is unsafe
        unsafe_mask = safety_val < self.safe_margin  # (K,)

        try:
            u_safe_t = self.safe_control_function(state, u_nominal, t)
        except TypeError:
            try:
                u_safe_t = self.safe_control_function(state, t)
            except TypeError:
                u_safe_t = self.safe_control_function(state)

        unsafe_mask = unsafe_mask.to(u_nominal.device)
        u_safe_t = u_safe_t.to(u_nominal.device)

        # Construct shielded action
        u_shielded = torch.where(unsafe_mask.unsqueeze(-1), u_safe_t, u_nominal)
        return u_shielded


if __name__ == "__main__":
    # Smoke test for DualGuardShield
    def mock_safety_fn(state, t=0):
        # State safe if x > 0
        return state[:, 0]

    def mock_safe_control_fn(state, u_nominal, t=0):
        # Safe control is just zeros
        return torch.zeros_like(u_nominal)

    shield = DualGuardShield(
        safety_function=mock_safety_fn,
        safe_control_function=mock_safe_control_fn,
        safe_margin=0.0,
    )

    # 2 states, 2 controls
    # state 0: x = 1.0 (Safe)
    # state 1: x = -1.0 (Unsafe)
    states = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    u_nom = torch.tensor([[5.0, 5.0], [5.0, 5.0]])

    u_shielded = shield(states, u_nom)
    print("Nominal commands:\\n", u_nom)
    print("Shielded commands:\\n", u_shielded)

    assert torch.allclose(u_shielded[0], u_nom[0]), "Safe state should keep nominal control"
    assert torch.allclose(u_shielded[1], torch.zeros_like(u_nom[1])), (
        "Unsafe state should use safe control"
    )
    print("DualGuardShield test passed!")
