# @title MPPI Planner
import functools
import logging

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt

from src.utils import *

logger = logging.getLogger(__name__)

def dubins_dynamics_tensor(current_state: torch.Tensor, action: torch.Tensor, dt: float) \
        -> torch.Tensor:
    """
    current_state: shape(num_samples, dim_x)
    action: shape(num_samples, dim_u)

    Implemented discrete time dynamics with RK-4.

    return:
    next_state: shape(num_samples, dim_x)
    """
    def one_step_dynamics(state, action):
        """Compute the derivatives [dx/dt, dy/dt, dtheta/dt]."""
        x_dot = 5.*torch.cos(state[:, 2])
        y_dot = 5.*torch.sin(state[:, 2])
        theta_dot = action[:, 0]
        return torch.stack([x_dot, y_dot, theta_dot], dim=1)

    # k1
    k1 = one_step_dynamics(current_state, action)
    # k2
    mid_state_k2 = current_state + 0.5 * dt * k1
    k2 = one_step_dynamics(mid_state_k2, action)
    # k3
    mid_state_k3 = current_state + 0.5 * dt * k2
    k3 = one_step_dynamics(mid_state_k3, action)
    # k4
    end_state_k4 = current_state + dt * k3
    k4 = one_step_dynamics(end_state_k4, action)
    # Combine k1, k2, k3, k4 to compute the next state
    next_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    next_state[..., -1] = next_state[..., -1] % (2 * np.pi)
    return next_state

def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


def squeeze_n(v, n_squeeze):
    for _ in range(n_squeeze):
        v = v.squeeze(0)
    return v

def handle_batch_input(n):
    def _handle_batch_input(func):
        """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
            batch_dims = []
            for arg in args:
                if is_tensor_like(arg):
                    if len(arg.shape) > n:
                        # last dimension is type dependent; all previous ones are batches
                        batch_dims = arg.shape[:-(n - 1)]
                        break
                    elif len(arg.shape) < n:
                        n_batch_dims_to_add = n - len(arg.shape)
                        batch_ones_to_add = [1] * n_batch_dims_to_add
                        args = [v.view(*batch_ones_to_add, *v.shape) if is_tensor_like(v) else v for v in args]
                        ret = func(*args, **kwargs)
                        if isinstance(ret, tuple):
                            ret = [squeeze_n(v, n_batch_dims_to_add) if is_tensor_like(v) else v for v in ret]
                            return ret
                        else:
                            if is_tensor_like(ret):
                                return squeeze_n(ret, n_batch_dims_to_add)
                            else:
                                return ret
            # no batches; just return normally
            if not batch_dims:
                return func(*args, **kwargs)

            # reduce all batch dimensions down to the first one
            args = [v.view(-1, *v.shape[-(n - 1):]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
            ret = func(*args, **kwargs)
            # restore original batch dimensions; keep variable dimension (nx)
            if type(ret) is tuple:
                ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                    v.view(*batch_dims, *v.shape[-(n - 1):]) if len(v.shape) == n else v.view(*batch_dims)) for v in
                       ret]
            else:
                if is_tensor_like(ret):
                    if len(ret.shape) == n:
                        ret = ret.view(*batch_dims, *ret.shape[-(n - 1):])
                    else:
                        ret = ret.view(*batch_dims)
            return ret

        return wrapper

    return _handle_batch_input

class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=15, device="cpu",
                 dt=0.1,
                 terminal_state_cost=None,
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 step_dependent_dynamics=False,
                 rollout_samples=1,
                 rollout_var_cost=0,
                 rollout_var_discount=0.95,
                 sample_null_action=False,
                 noise_abs_cost=False):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_
        self.dt = dt

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

    @handle_batch_input(n=2)
    def _dynamics(self, state, u, t, dt):
        return self.F(state, u, t, dt) if self.step_dependency else self.F(state, u, dt)

    @handle_batch_input(n=2)
    def _running_cost(self, state, u, t):
        return self.running_cost(state, u, t)

    @handle_batch_input(n=2)
    def _terminal_state_cost(self, state, u):
        return self.terminal_state_cost(state, u)

    def command(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        # print(f'state shape as input: {state.shape}')
        return self._command(state)

    def _command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()
        # print(f'cost total: {cost_total}')
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        for t in range(self.T):
            self.U[t] += torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.U[:self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]
        return action

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        # self.state.shape = [1, dim_x]
        # make state.shape = K, self.nx
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)
        # print(f'state_shape before rollout: {state.shape}')
        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.M, 1, 1)
        # print(f'state_shape after rollout: {state.shape}')

        states = []
        actions = []
        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
            state = self._dynamics(state, u, t, self.dt)
            c = self._running_cost(state, u, t)
            # print(f'running_cost: \n{c}')
            # c.shape(M, K)
            cost_samples += c
            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # print(f'inter step costs: \n{cost_samples}')

        # Actions is M x K x T x nu
        # States is M x K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self._terminal_state_cost(states[..., -1, :], actions[..., -1, :])
            cost_samples += c

        cost_total += cost_samples.mean(dim=0)

        cost_total += cost_var * self.rollout_var_cost
        return cost_total, states, actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T))
        # print(f'sampled noise: {self.noise}')
        # broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise
        # print(f'samples: \n{self.perturbed_action}')
        if self.sample_null_action:
            self.perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv  # Like original paper

        self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions /= self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                action[:, self._slice_control(t)] = cu
        return action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def get_rollouts(self, state, num_rollouts=1):
        """
            :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
            :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                                 dynamics
            :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device)
        states[:, 0] = state
        for t in range(T):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
                                              self.u_scale * self.U[t].view(num_rollouts, -1), t)
        return states[:, 1:]


class Navigator:
    def __init__(self, planner_type="mppi", device='cpu', dtype=torch.float32, dt=0.1):

        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.planner_type = planner_type

        self._odom_torch = None
        self.planner = self._start_planner()
        self._map_torch = None  # Initialize later with the map data
        self._cell_size = None  # Initialize later with the map resolution
        self._goal_torch = None
        self._goal_thresh = 0.1


    def get_command(self):
        x = self._odom_torch[0]
        y = self._odom_torch[1]
        dist_goal = torch.sqrt((x - self._goal_torch[0]) ** 2 + (y - self._goal_torch[1]) ** 2)
        if dist_goal.item() < self._goal_thresh:
            return torch.tensor([0])
        command = None
        if self.planner_type == "mppi":
            command = self.planner.command(self._odom_torch)
        return command

    def set_odom(self, position, orientation):
        """
        :param position: (array-like): [x, y, z]
        :param orientation: (array-like): theta
        """
        self._odom_torch = torch.tensor([position[0],
                                         position[1],
                                         orientation],
                                         dtype=self.dtype, device=self.device)

    def set_map(self, map_data, map_resolution):
        """
        :param map_data: (array-like): flattened map in row-major order
        :param map_resolution: (float): map resolution
        """
        self._map_torch = torch.tensor(map_data, dtype=self.dtype, device=self.device)
        self._cell_size = map_resolution

    def set_goal(self, position):
        """
        :param position: (array-like): goal position [x, y]
        """
        self._goal_torch = torch.tensor([position[0], position[1]],
                                        dtype=self.dtype, device=self.device)


    def get_sampled_trajectories(self):
        if self.planner_type == "mppi":
            # states: torch.tensor, shape(M, K, T, nx)
            trajectories = self.planner.states
            M, K, T, nx = trajectories.shape
            return trajectories.view(M*K, T, nx)

    def make_mppi_config(self):
        mppi_config = {}

        mppi_config['dynamics'] = dubins_dynamics_tensor
        mppi_config['running_cost'] = self.mppi_cost_func
        mppi_config['nx'] = 3    # [x, y, theta]
        mppi_config['dt'] = self.dt
        mppi_config['noise_sigma'] = torch.tensor([2], dtype=self.dtype, device=self.device)
        mppi_config['num_samples'] = 200
        mppi_config['horizon'] = 20
        mppi_config['device'] = self.device
        mppi_config['u_min'] = torch.tensor([-4])
        mppi_config['u_max'] = torch.tensor([4])
        mppi_config['lambda_'] = 1
        mppi_config['rollout_samples'] = 1
        mppi_config['terminal_state_cost'] = self.mppi_terminal_state_cost_funct
        mppi_config['rollout_var_cost'] = 0.1  # Increase from 0
        mppi_config['rollout_var_discount'] = 0.9  # Adjust from 0.95
        mppi_config['u_init'] = torch.tensor([0.0], dtype=self.dtype, device=self.device)

        return mppi_config


    def _start_planner(self,):
        if self.planner_type == 'mppi':
            mppi_config = self.make_mppi_config()
            return MPPI(**mppi_config)


    def _compute_collision_cost(self, current_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Note: Collision map is 0 for collision, 1 for free space
        current_state: shape(num_samples, dim_x)
        action: shape(num_samples, dim_u)

        return:
        cost: shape(num_samples)
        """
        position_map = (current_state[..., :2]) / self._cell_size
        position_map = torch.round(position_map).long().to(self.device)
        is_out_of_bound = torch.logical_or(
            torch.logical_or(
                position_map[..., 0] < 0, position_map[..., 0] >= self._map_torch.shape[1]
            ),
            torch.logical_or(
                position_map[..., 1] < 0, position_map[..., 1] >= self._map_torch.shape[0]
            ),
        )
        position_map[..., 0] = torch.clamp(position_map[..., 0], 0, self._map_torch.shape[1] - 1)
        position_map[..., 1] = torch.clamp(position_map[..., 1], 0, self._map_torch.shape[0] - 1)
        # Collision check
        collisions = self._map_torch[position_map[..., 1], position_map[..., 0]]
        collisions = torch.where(collisions == 1, torch.tensor(0.0, device=self.device), collisions.float())
        collisions = torch.where(collisions == 0, torch.tensor(1.0, device=self.device), collisions.float())

        # Out of bound cost
        collisions[is_out_of_bound] = 1.0
        return collisions


    def mppi_cost_func(self, current_state: torch.Tensor, action: torch.Tensor, t, weights=(1, 2.0)) -> torch.Tensor:
        """
        current_state: shape(num_samples, dim_x)
        return:
        cost: torch.tensor, shape(num_samples, 1)
        """
        dist_goal_cost = torch.norm(current_state[:, :2] - self._goal_torch, dim=1)
        collision_cost = self._compute_collision_cost(current_state, action)

        cost = weights[0] * dist_goal_cost + weights[1] * collision_cost
        return cost

    def mppi_terminal_state_cost_funct(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        states: shape(M*K, T, dim_x)
        """
        return self.mppi_cost_func(states, actions, 1)
    

if __name__ == "__main__":
    state = torch.tensor([0, 0, np.pi])
    navigator = Navigator()
    navigator.set_odom(state[:2],state[-1])

    map_data = np.zeros((100, 100))
    map_data[10:60, 10:60] = 1
    navigator.set_map(map_data, 0.5)

    navigator.set_goal([50, 50])

    col_x = []
    col_y = []

    non_col_x = []
    non_col_y = []

    for i in range(400):
        u = navigator.get_command()
        state = dubins_dynamics_tensor(state.unsqueeze(0), u.unsqueeze(0), 0.1)
        state = state.squeeze()
        navigator.set_odom(state[:2], state[-1])

        if navigator._compute_collision_cost(state.unsqueeze(0), u.unsqueeze(0)) > 0:
            col_x.append(state[0].item())
            col_y.append(state[1].item())
        else:
            non_col_x.append(state[0].item())
            non_col_y.append(state[1].item())

    command = navigator.get_command()

    print(command)

    plt.scatter(col_x, col_y, c='red')
    plt.scatter(non_col_x, non_col_y, c='green')
    plt.imshow(navigator._map_torch.cpu().numpy(), origin='lower', extent=[0, 50, 0, 50])
    plt.show()
