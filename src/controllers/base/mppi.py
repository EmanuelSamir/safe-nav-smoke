from typing import Callable, Optional

import torch
from torch.distributions import MultivariateNormal

from src.controllers.base.schemas import MPPIConfig


class MPPI:
    """Model Predictive Path Integral (MPPI) Controller.

    Implements the stochastic trajectory optimization method described in:
    Williams et al., "Information-Theoretic MPC for Model-Based Reinforcement Learning" (2017)
    """

    def __init__(
        self,
        config: MPPIConfig,
        rollout_filter_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        ] = None,
        output_filter_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        ] = None,
    ):
        self.config = config
        self.device = config.device
        self.nx = config.nx
        self.T = config.horizon
        self.K = config.num_samples
        self.rollout_filter_fn = rollout_filter_fn
        self.output_filter_fn = output_filter_fn

        # --- Control limits (ensure both exist and are converted to tensors) ---
        if config.u_min is None or config.u_max is None:
            raise ValueError("u_min and u_max must be defined")
        self.u_min = torch.tensor(config.u_min, device=self.device, dtype=torch.float32)
        self.u_max = torch.tensor(config.u_max, device=self.device, dtype=torch.float32)

        # --- Define mean and covariance of control noise ---
        if config.noise_sigma is not None:
            self.noise_sigma = torch.tensor(config.noise_sigma, device=self.device, dtype=torch.float32)
            self.dtype = self.noise_sigma.dtype
        elif getattr(config, "alpha_noise_sigma", None) is not None:
            diag_vals = config.alpha_noise_sigma * (self.u_max - self.u_min)
            self.noise_sigma = torch.diag(diag_vals).to(self.device)
            self.dtype = self.noise_sigma.dtype
        else:
            raise ValueError("Either noise_sigma or alpha_noise_sigma must be provided in MPPIConfig")
            
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        
        # --- Determine control dimension (nu) ---
        self.nu = self.noise_sigma.shape[0] if len(self.noise_sigma.shape) > 0 else 1

        if config.noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)
        else:
            noise_mu = torch.tensor(config.noise_mu, dtype=self.dtype)
        self.noise_mu = noise_mu.to(self.device)

        # Create a Gaussian distribution for sampling control noise
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        # --- Initialize nominal control sequence U(t) ---
        if config.u_init is None:
            self.U = self.noise_dist.sample((self.T,))
        else:
            u_init_tensor = torch.tensor(config.u_init, device=self.device, dtype=self.dtype)
            self.U = u_init_tensor.repeat(self.T, 1)

        # --- Buffers for results ---
        self.state = None
        self.synthetic_states = None
        self.info = None
        self.cost_total = None
        self.omega = None
        self.noise = None
        self.perturbed_actions = None

    # =====================================================
    #  ABSTRACT/INTERFACE METHODS (To be overridden by subclasses)
    # =====================================================

    def dynamics(
        self, state: torch.Tensor, u: torch.Tensor, t: Optional[int] = None
    ) -> torch.Tensor:
        """Propagate state given action. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement dynamics")

    def running_cost(
        self, state: torch.Tensor, u: torch.Tensor, t: Optional[int] = None
    ) -> torch.Tensor:
        """Evaluate running cost for state and action. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement running_cost")

    def terminal_state_cost(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Evaluate terminal state cost. Optional, defaults to None."""
        return None

    # =====================================================
    #  UTILITY FUNCTIONS
    # =====================================================

    def _process_bounds(self, u_min, u_max):
        # Legacy stub, logic moved to __init__
        pass

    def _bound_action(self, u: torch.Tensor) -> torch.Tensor:
        """Clamp actions element-wise within limits."""
        return torch.clamp(u, self.u_min, self.u_max)

    # =====================================================
    #  MAIN CONTROL INTERFACE
    # =====================================================

    def command(
        self,
        state: torch.Tensor,
        shift_nominal_trajectory: bool = True,
        info: Optional[dict] = None,
    ) -> torch.Tensor:
        """Compute next control command given current state."""
        # Convert input state to torch tensor with explicit shape (nx,)
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.device)
        self.info = info

        if shift_nominal_trajectory:
            self._shift_nominal_trajectory()

        # Compute costs of sampled trajectories
        cost_total = self._compute_total_cost_batch()

        # Compute weighting (softmax of cost)
        omega = self._compute_weights(cost_total)

        # Weighted average of noise across all trajectories
        #   noise: shape (K, T, nu)
        #   omega: shape (K,)
        #   → weighted perturbation: shape (T, nu)
        perturbation = torch.sum(omega.view(self.K, 1, 1) * self.noise, dim=0)

        # Update nominal control sequence
        self.U += perturbation

        # Apply output safety filter to the final action at t=0
        action_opt = self.U[: self.config.u_per_command]
        if self.output_filter_fn is not None:
            action = self.output_filter_fn(self.state.unsqueeze(0), action_opt, 0)
        else:
            action = action_opt

        return (
            self._bound_action(action[0])
            if self.config.u_per_command == 1
            else self._bound_action(action)
        )

    def _shift_nominal_trajectory(self) -> None:
        """Shift nominal control sequence one step forward."""
        # Roll sequence by -1 → drop first action, shift all, add new u_init at end
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = self.U[-2].clone()

    # =====================================================
    #  COST COMPUTATION
    # =====================================================

    def _compute_total_cost_batch(self) -> torch.Tensor:
        """1. Sample noisy trajectories.

        2. Roll out each trajectory to compute cost.
        3. Add control perturbation cost.
        """
        self._sample_noisy_actions()

        # Action noise penalty term (encourages low-variance controls)
        if self.config.noise_abs_cost:
            action_cost = self.config.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
        else:
            action_cost = self.config.lambda_ * (self.noise @ self.noise_sigma_inv)

        # Rollout to compute running + terminal costs
        rollout_cost, self.synthetic_states, actions = self._rollout_trajectories(
            self.perturbed_actions
        )

        # Sum of cost terms
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        return rollout_cost + perturbation_cost

    # =====================================================
    #  SAMPLING TRAJECTORIES
    # =====================================================

    def _sample_noisy_actions(self) -> None:
        """Sample K trajectories with Gaussian noise over T timesteps.

        Returns:
            self.perturbed_actions: (K, T, nu)
            self.noise:             (K, T, nu)
        """
        # 1. Sample K×T actions from noise distribution:
        # Each trajectory k gets T steps of (nu)-dimensional noise
        noise = self.noise_dist.rsample((self.K, self.T))  # shape (K, T, nu)

        # 2. Broadcast the nominal control U[t] (T, nu) across all K trajectories:
        #    → expand to (K, T, nu)
        U_expanded = self.U.unsqueeze(0).expand(self.K, self.T, self.nu)

        # 3. Add noise to each nominal control
        perturbed = U_expanded + noise

        # 5. Clip controls to physical limits
        self.perturbed_actions = self._bound_action(perturbed)

        # 6. Compute the actual noise applied after clipping
        self.noise = self.perturbed_actions - U_expanded

    # =====================================================
    #  ROLLOUT SIMULATION
    # =====================================================

    def _rollout_trajectories(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate K trajectories over horizon T.

        Args:
            actions: tensor (K, T, nu)

        Returns:
            cost_total: (K,)
            states: (K, T, nx)
            actions: (K, T, nu)
        """
        K, T, nu = actions.shape
        assert nu == self.nu, "Action dimension mismatch"

        # 1. Initialize starting states
        #    If state is single (nx,), repeat K times for each trajectory
        if self.state.shape == (self.nx,):
            state = self.state.unsqueeze(0).expand(K, -1).clone()  # shape (K, nx)
        else:
            state = self.state.clone()

        cost_total = torch.zeros(K, device=self.device, dtype=self.dtype)
        all_states = []
        shielded_actions = []

        # 2. Rollout dynamics for each timestep
        for t in range(T):
            u_t = actions[:, t]  # shape (K, nu)

            # Apply functional safety filter during rollout if configured
            if self.rollout_filter_fn is not None:
                u_shielded_t = self.rollout_filter_fn(state, u_t, t)
            else:
                u_shielded_t = u_t

            u_shielded_t = self._bound_action(u_shielded_t)
            shielded_actions.append(u_shielded_t)

            u_apply = self.config.u_scale * u_shielded_t
            next_state = (
                self.dynamics(state, u_apply, t)
                if self.config.step_dependent_dynamics
                else self.dynamics(state, u_apply)
            )
            c_t = (
                self.running_cost(next_state, u_apply, t)
                if self.config.step_dependent_dynamics
                else self.running_cost(next_state, u_apply)
            )  # running cost
            cost_total += c_t
            all_states.append(next_state)
            state = next_state  # move forward

        # Stack all intermediate states and actions
        states_tensor = torch.stack(all_states, dim=1)
        self.perturbed_actions = torch.stack(shielded_actions, dim=1)
        self.noise = self.perturbed_actions - self.U.unsqueeze(0).expand(K, T, nu)

        # 3. Optional terminal cost
        term_cost = self.terminal_state_cost(states_tensor)
        if term_cost is not None:
            cost_total += term_cost

        return cost_total, states_tensor, self.perturbed_actions

    # =====================================================
    #  WEIGHT COMPUTATION
    # =====================================================

    def _compute_weights(self, cost_total: torch.Tensor) -> torch.Tensor:
        """Compute normalized trajectory weights using PyTorch softmax."""
        self.omega = torch.softmax(-cost_total / self.config.lambda_, dim=0)
        return self.omega

    # =====================================================
    #  MISCELLANEOUS
    # =====================================================

    def reset(self) -> None:
        """Resample a new nominal control sequence."""
        self.U = self.noise_dist.sample((self.T,))


if __name__ == "__main__":
    # Test subclass of MPPI to verify functionality with safety filter fn
    class SimpleMPPI(MPPI):
        def dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return state + u

        def running_cost(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return torch.sum(state**2, dim=-1) + 0.1 * torch.sum(u**2, dim=-1)

    # Initialize parameters
    config = MPPIConfig(
        nx=2,
        noise_sigma=torch.eye(2),
        num_samples=10,
        horizon=5,
        device="cpu",
        u_min=-torch.ones(2),
        u_max=torch.ones(2),
    )

    # Instantiate controller with a dummy safety filter that clamps values to max 0.5
    dummy_filter = lambda state, u, t=0: torch.clamp(u, -0.5, 0.5)
    controller = SimpleMPPI(config, output_filter_fn=dummy_filter)

    # Initial state
    state = torch.tensor([1.0, -1.0])

    # Run command
    action = controller.command(state)
    print("Test passed. Action:", action)
