import torch
from torch.distributions import MultivariateNormal
import typing

class MPPI:
    """
    Model Predictive Path Integral (MPPI) Controller
    Implements the stochastic trajectory optimization method described in:
    Williams et al., "Information-Theoretic MPC for Model-Based Reinforcement Learning" (2017)
    """

    def __init__(
        self,
        dynamics,                 # function f(x,u) or f(x,u,t)
        running_cost,             # function c(x,u)
        nx,                       # state dimension
        noise_sigma,              # (nu x nu) covariance matrix for action noise
        num_samples=100,          # K - number of sampled trajectories
        horizon=10,               # T - planning horizon
        device="cpu",
        terminal_state_cost=None, # function c(x)
        lambda_=1.0,              # temperature parameter
        noise_mu=None,
        u_min=None, u_max=None,
        u_init=None,
        u_scale=1.0,
        u_per_command=1,
        step_dependent_dynamics=False,
        noise_abs_cost=False,
    ):
        # --- Basic dimensions and parameters ---
        self.device = device
        self.dtype = noise_sigma.dtype
        self.nx = nx                   # dimension of state vector (e.g. [x, y, θ])
        self.T = horizon               # number of timesteps in the horizon
        self.K = num_samples           # number of trajectories to sample
        self.lambda_ = lambda_

        # --- Determine control dimension (nu) ---
        self.nu = noise_sigma.shape[0] if len(noise_sigma.shape) > 0 else 1

        # --- Define mean and covariance of control noise ---
        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)
        self.noise_mu = noise_mu.to(device)
        self.noise_sigma = noise_sigma.to(device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)

        # Create a Gaussian distribution for sampling control noise
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        # --- Control limits (ensure both exist and are tensors) ---
        self.u_min, self.u_max = self._process_bounds(u_min, u_max)
        self.u_scale = u_scale
        self.u_per_command = u_per_command

        # --- Initialize nominal control sequence U(t) ---
        # Each row U[t] ∈ ℝ^{nu} is the control at step t
        if u_init is None:
            self.U = self.noise_dist.sample((self.T,))
        else:
            self.U = u_init.repeat(self.T, 1)

        # --- Define cost and dynamics functions ---
        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.step_dependent_dynamics = step_dependent_dynamics

        # --- Sampling options ---
        self.noise_abs_cost = noise_abs_cost

        # --- Buffers for results ---
        self.state = None
        self.synthetic_states = None
        self.info = None
        self.cost_total = None
        self.omega = None
        self.noise = None
        self.perturbed_actions = None

    # =====================================================
    #  UTILITY FUNCTIONS
    # =====================================================

    def _process_bounds(self, u_min, u_max):
        """Ensure min/max controls are both defined and moved to device."""
        if u_min is None or u_max is None:
            raise ValueError("u_min and u_max must be defined")
        return u_min.to(self.device), u_max.to(self.device)

    def _bound_action(self, u):
        """Clamp actions element-wise within limits."""
        return torch.clamp(u, self.u_min, self.u_max)

    # =====================================================
    #  MAIN CONTROL INTERFACE
    # =====================================================

    def command(self, state, shift_nominal_trajectory=True, info=None):
        """
        Compute next control command given current state.
        """
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

        # Return first control (or block if u_per_command > 1)
        action = self.U[:self.u_per_command]
        return action[0] if self.u_per_command == 1 else action

    def _shift_nominal_trajectory(self):
        """Shift nominal control sequence one step forward."""
        # Roll sequence by -1 → drop first action, shift all, add new u_init at end
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = self.U[-2].clone()

    # =====================================================
    #  COST COMPUTATION
    # =====================================================

    def _compute_total_cost_batch(self):
        """
        1. Sample noisy trajectories.
        2. Roll out each trajectory to compute cost.
        3. Add control perturbation cost.
        """
        self._sample_noisy_actions()

        # Action noise penalty term (encourages low-variance controls)
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
        else:
            action_cost = self.lambda_ * (self.noise @ self.noise_sigma_inv)

        # Rollout to compute running + terminal costs
        rollout_cost, self.synthetic_states, actions = self._rollout_trajectories(self.perturbed_actions)

        # Sum of cost terms
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        return rollout_cost + perturbation_cost

    # =====================================================
    #  SAMPLING TRAJECTORIES
    # =====================================================

    def _sample_noisy_actions(self):
        """
        Sample K trajectories with Gaussian noise over T timesteps.

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

    def _rollout_trajectories(self, actions):
        """
        Simulate K trajectories over horizon T.

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
            state = self.state.unsqueeze(0).repeat(K, 1)  # shape (K, nx)
        else:
            state = self.state.clone()

        cost_total = torch.zeros(K, device=self.device, dtype=self.dtype)
        all_states = []

        # 2. Rollout dynamics for each timestep
        for t in range(T):
            u_t = self.u_scale * actions[:, t]           # shape (K, nu)
            next_state = self._apply_dynamics(state, u_t, t)
            c_t = self._apply_cost(next_state, u_t, t)   # running cost
            cost_total += c_t
            all_states.append(next_state)
            state = next_state                           # move forward

        # Stack all intermediate states → shape (K, T, nx)
        states_tensor = torch.stack(all_states, dim=1)

        # 3. Optional terminal cost
        if self.terminal_state_cost is not None:
            cost_total += self._apply_terminal_state_cost(states_tensor)

        return cost_total, states_tensor, actions

    def _apply_dynamics(self, state, u, t):
        """Call dynamics with or without time index."""
        if self.step_dependent_dynamics:
            return self.dynamics(state, u, t)
        return self.dynamics(state, u)

    def _apply_cost(self, state, u, t):
        """Call running cost with or without time index."""
        if self.step_dependent_dynamics:
            return self.running_cost(state, u, t)
        return self.running_cost(state, u)

    def _apply_terminal_state_cost(self, state):
        """Apply terminal state cost function."""
        return self.terminal_state_cost(state)

    # =====================================================
    #  WEIGHT COMPUTATION
    # =====================================================

    def _compute_weights(self, cost_total):
        """
        Compute normalized trajectory weights using exponential transformation:
        w_i = exp(- (J_i - J_min) / λ ) / Σ exp(...)
        """
        min_cost = torch.min(cost_total)
        exp_weights = torch.exp(-(cost_total - min_cost) / self.lambda_)
        self.omega = exp_weights / torch.sum(exp_weights)
        return self.omega

    # =====================================================
    #  MISCELLANEOUS
    # =====================================================

    def reset(self):
        """Resample a new nominal control sequence."""
        self.U = self.noise_dist.sample((self.T,))

class DualGuardMPPI(MPPI):
    """
    DualGuard MPPI controller.
    Adds:
    - Outer Guard: penalizes unsafe trajectories during rollout.
    - Inner Guard: corrects control near unsafe boundary.
    """

    def __init__(
        self,
        *args,
        hj_value_function,
        hj_grad_function,
        safe_margin=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Hamilton-Jacobi reachability functions
        self.hj_value_function = hj_value_function
        self.hj_grad_function = hj_grad_function

        # DualGuard hyperparameters
        self.beta_outer = beta_outer       # penalty for unsafe trajectories
        self.alpha_inner = alpha_inner     # gain for inner correction
        self.safe_margin = safe_margin     # activation margin

    # -----------------------------------------------------
    # Override rollout: add Outer Guard penalty
    # -----------------------------------------------------
    def _rollout_trajectories(self, actions):
        """
        Same as MPPI but adds HJ-based outer safety penalty.
        """
        K, T, nu = actions.shape
        if self.state.shape == (self.nx,):
            state = self.state.unsqueeze(0).repeat(K, 1)
        else:
            state = self.state.clone()

        cost_total = torch.zeros(K, device=self.device, dtype=self.dtype)
        all_states = []

        for t in range(T):
            u_t = self.u_scale * actions[:, t]
            next_state = self._apply_dynamics(state, u_t, t)
            c_t = self._apply_cost(next_state, u_t, t)
            cost_total += c_t

            # === Outer Guard penalty ===
            if self.hj_value_function is not None:
                V = self.hj_value_function(next_state)
                violation = torch.clamp(-V, min=0.0)
                cost_total += self.beta_outer * violation

            all_states.append(next_state)
            state = next_state

        states_tensor = torch.stack(all_states, dim=1)

        if self.terminal_state_cost is not None:
            cost_total += self._apply_terminal_state_cost(states_tensor)

        return cost_total, states_tensor, actions

    # -----------------------------------------------------
    # Override command: add Inner Guard correction
    # -----------------------------------------------------
    def command(self, state, shift_nominal_trajectory=True, info=None):
        """
        Compute next control command given current state.
        Adds inner-guard projection if near unsafe boundary.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.device)
        self.info = info

        if shift_nominal_trajectory:
            self._shift_nominal_trajectory()

        cost_total = self._compute_total_cost_batch()
        omega = self._compute_weights(cost_total)

        perturbation = torch.sum(omega.view(self.K, 1, 1) * self.noise, dim=0)
        self.U += perturbation

        action = self.U[:self.u_per_command]
        action = action[0] if self.u_per_command == 1 else action

        # === Inner Guard correction ===
        if (
            self.hj_value_function is not None
            and self.hj_grad_function is not None
        ):
            V = self.hj_value_function(self.state)
            if V < self.safe_margin:
                dVdx = self.hj_grad_function(self.state)
                f_xu = self.dynamics(self.state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
                LfV = torch.dot(dVdx, f_xu)
                if LfV < -self.alpha_inner * V:
                    correction = (
                        (-self.alpha_inner * V - LfV)
                        / (torch.norm(dVdx) ** 2 + 1e-6)
                    ) * dVdx
                    action = action + correction

        return action
