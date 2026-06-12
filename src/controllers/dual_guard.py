import torch

from controllers.mppi import MPPI


class DualGuard(MPPI):
    """Shielded Model Predictive Path Integral (DualGuard MPPI) Controller.

    Based on the "DualGuard MPPI" algorithm. This is the generic base class that uses a modular
    safety index function and a safe backup control policy to perform "Safe Rollouts" and
    apply an "Output Safety Filter".

    The key mechanism is "Safe Rollouts":
    - During nominal trajectory sampling, if a simulated state violates safety (safety_function(x) < margin),
      the rollout action perturbation is overridden by a perturbation leading to the safe control policy:
      Delta^k_j = u_safe*(x_j) - u_j.
    - The updated control sequence then computes its weight based on these corrected trajectories.
    - An output safety filter is applied at execution time to override commands if the current
      measured state is unsafe: u_0** = u_safe*(x_0).

    This class expects modular safety functions passed in during initialization.
    """

    def __init__(
        self,
        *args,
        safety_function,  # function(state_tensor) -> values: shape (K,) or (K, 1)
        safe_control_function,  # function(state_tensor) -> safe_actions: shape (K, nu)
        safe_margin=0.0,  # Safe set is defined where safety_function(x) >= safe_margin
        **kwargs,
    ):
        """Args:
        safety_function (callable): Evaluates the safety index/value on a batch of states.
        safe_control_function (callable): Computes the safe backup control u_safe(x) on a batch of states.
        safe_margin (float): Offset for safety boundary.
        """
        super().__init__(*args, **kwargs)
        self.safety_function = safety_function
        self.safe_control_function = safe_control_function
        self.safe_margin = safe_margin

    def _compute_total_cost_batch(self):
        """Computes MPPI cost batch, ensuring actions are corrected using the Shield mechanism
        during the rollout phase before calculating the action perturbation cost.
        """
        # 1. Sample nominal Gaussian control perturbations (sets self.perturbed_actions and self.noise)
        self._sample_noisy_actions()

        # 2. Simulate trajectories with dynamic Shield application.
        # This updates self.perturbed_actions and self.noise IN PLACE to include correct safe overrides.
        rollout_cost, self.synthetic_states, _ = self._rollout_trajectories(self.perturbed_actions)

        # 3. Action noise penalty (using the corrected self.noise = Delta^k_j)
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
        else:
            action_cost = self.lambda_ * (self.noise @ self.noise_sigma_inv)

        # 4. Sum of cost terms
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))

        return rollout_cost + perturbation_cost

    def _rollout_trajectories(self, actions):
        """Performs rollout of K trajectories while actively overriding perturbations with the
        safe control backup policy whenever a state becomes unsafe.
        """
        K, T, nu = actions.shape
        assert nu == self.nu, f"Action dimension mismatch: expected {self.nu}, got {nu}"

        # Initialize starting states
        if self.state.shape == (self.nx,):
            state = self.state.unsqueeze(0).repeat(K, 1)  # (K, nx)
        else:
            state = self.state.clone()

        cost_total = torch.zeros(K, device=self.device, dtype=self.dtype)
        all_states = []
        shielded_actions_list = []

        # Rollout dynamics for each step in the horizon
        for t in range(T):
            # === Safe Shield Check (Algorithm 1: "Safe Rollouts") ===
            # Evaluate safety condition of current state x_j BEFORE propagating, passing time step 't'
            try:
                safety_val = self.safety_function(state, t)
            except TypeError:
                safety_val = self.safety_function(state)

            if safety_val.dim() > 1:
                safety_val = safety_val.squeeze(-1)

            # Compute boolean mask where state is unsafe
            unsafe_mask = safety_val < self.safe_margin  # (K,)

            # Nominal perturbed action sampled originally
            u_nominal_t = actions[:, t]  # (K, nu)

            # Compute safe backup control for all states in batch, passing time step 't'
            try:
                u_safe_t = self.safe_control_function(state, t)
            except TypeError:
                u_safe_t = self.safe_control_function(state)

            # Construct shielded action:
            # If safe: u_nominal_t
            # If unsafe: u_safe_t
            u_shielded_t = torch.where(unsafe_mask.unsqueeze(-1), u_safe_t, u_nominal_t)

            # Ensure the resulting physical actions stay within defined controller bounds
            u_shielded_t = self._bound_action(u_shielded_t)

            # Save the shielded action
            shielded_actions_list.append(u_shielded_t)

            # Propagate environment using the actual applied control (u_scale allows custom scaling)
            u_apply = self.u_scale * u_shielded_t
            next_state = self._apply_dynamics(state, u_apply, t)

            # Calculate and sum running cost for this step
            c_t = self._apply_cost(next_state, u_apply, t)
            cost_total += c_t

            # Move to next time step
            all_states.append(next_state)
            state = next_state

        # Finalize arrays and update state
        states_tensor = torch.stack(all_states, dim=1)  # (K, T, nx)
        shielded_actions = torch.stack(shielded_actions_list, dim=1)  # (K, T, nu)

        # Crucial: update the internal class properties to reflect the ACTUALLY APPLIED perturbations Delta^k_j
        self.perturbed_actions = shielded_actions
        U_expanded = self.U.unsqueeze(0).expand(K, T, nu)
        self.noise = (
            self.perturbed_actions - U_expanded
        )  # This is Delta^k_j used in the update rule

        # Optional terminal cost
        if self.terminal_state_cost is not None:
            cost_total += self._apply_terminal_state_cost(states_tensor)

        return cost_total, states_tensor, self.perturbed_actions

    def command(self, state, shift_nominal_trajectory=True, info=None):
        """Computes next control command given state, and applies safety output filter."""
        # 1. Calculate nominal/perturbed MPPI optimal sequence via standard MPPI logic
        # (This leverages overridden _compute_total_cost_batch to compute 'Safe Rollouts' weights)
        action_opt = super().command(
            state, shift_nominal_trajectory=shift_nominal_trajectory, info=info
        )

        # 2. Apply Output Safety Filter (Algorithm 1: "Output Filter")
        # Check if the ACTUAL measured current state x_0 is safe (t=0)
        try:
            safety_val = self.safety_function(self.state.unsqueeze(0), 0)
        except TypeError:
            safety_val = self.safety_function(self.state.unsqueeze(0))

        if safety_val.dim() > 1:
            safety_val = safety_val.squeeze(-1)
        safety_val = safety_val.squeeze()  # Scalar float tensor

        if safety_val < self.safe_margin:
            # Current state is unsafe! Hard-override output with safe policy u_safe*(x_0) at t=0
            try:
                u_safe_0 = self.safe_control_function(self.state.unsqueeze(0), 0).squeeze(0)
            except TypeError:
                u_safe_0 = self.safe_control_function(self.state.unsqueeze(0)).squeeze(0)  # (nu,)

            if self.u_per_command > 1:
                # If controller is configured to return multiple actions at once, repeat safe control
                action = u_safe_0.unsqueeze(0).repeat(self.u_per_command, 1)
            else:
                action = u_safe_0
        else:
            # State is safe, output the calculated MPPI optimized command
            action = action_opt

        return self._bound_action(action)
