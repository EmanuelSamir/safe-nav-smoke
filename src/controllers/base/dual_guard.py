from typing import Callable

import torch

from src.controllers.base.mppi import MPPI, MPPIParams


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
            u_safe_t = self.safe_control_function(state, t)
        except TypeError:
            u_safe_t = self.safe_control_function(state)

        unsafe_mask = unsafe_mask.to(u_nominal.device)
        u_safe_t = u_safe_t.to(u_nominal.device)
        
        # Construct shielded action
        u_shielded = torch.where(unsafe_mask.unsqueeze(-1), u_safe_t, u_nominal)
        return u_shielded


class DualGuard(MPPI):
    """Shielded Model Predictive Path Integral (DualGuard MPPI) Controller.

    Uses functional composition under the hood by mapping safety_function and
    safe_control_function into a unified safety_filter_fn.
    """

    def __init__(
        self,
        params: MPPIParams,
        safety_function: Callable,
        safe_control_function: Callable,
        safe_margin: float = 0.0,
    ):
        """DualGuard.

        Args:
            params (MPPIParams): Configuration parameters.
            safety_function (callable): Evaluates the safety index/value on a batch of states.
            safe_control_function (callable): Computes the safe backup control u_safe(x) on a batch of states.
            safe_margin (float): Offset for safety boundary.
        """

        def backup_filter_fn(
            state: torch.Tensor, u_nominal: torch.Tensor, t: int = 0
        ) -> torch.Tensor:
            try:
                safety_val = safety_function(state, t)
            except TypeError:
                safety_val = safety_function(state)

            if safety_val.dim() > 1:
                safety_val = safety_val.squeeze(-1)

            # Compute boolean mask where state is unsafe
            unsafe_mask = safety_val < safe_margin  # (K,)

            try:
                u_safe_t = safe_control_function(state, t)
            except TypeError:
                u_safe_t = safe_control_function(state)

            unsafe_mask = unsafe_mask.to(u_nominal.device)
            u_safe_t = u_safe_t.to(u_nominal.device)
            
            # Construct shielded action
            u_shielded = torch.where(unsafe_mask.unsqueeze(-1), u_safe_t, u_nominal)
            return u_shielded

        super().__init__(params=params, safety_filter_fn=backup_filter_fn)
        self.safety_function = safety_function
        self.safe_control_function = safe_control_function
        self.safe_margin = safe_margin


if __name__ == "__main__":
    # Test subclass of DualGuard to verify functionality
    class SimpleDualGuard(DualGuard):
        def dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return state + u

        def running_cost(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return torch.sum(state**2, dim=-1) + 0.1 * torch.sum(u**2, dim=-1)

    # Initialize parameters
    params = MPPIParams(
        nx=2,
        noise_sigma=torch.eye(2),
        num_samples=10,
        horizon=5,
        device="cpu",
        u_min=-torch.ones(2),
        u_max=torch.ones(2),
    )

    # Safety functions
    def safety_fn(state, t=0):
        return torch.min(state + 0.5, dim=-1)[0]

    def safe_control_fn(state, t=0):
        return torch.ones_like(state) * 0.5

    # Instantiate controller
    controller = SimpleDualGuard(
        params=params,
        safety_function=safety_fn,
        safe_control_function=safe_control_fn,
        safe_margin=0.0,
    )

    # Initial state (unsafe)
    state = torch.tensor([-1.0, -1.0])

    # Run command
    action = controller.command(state)
    print("Test passed. Action:", action)
