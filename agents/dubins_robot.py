import logging

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.basic_robot import STATE_THETA, STATE_X, STATE_Y, Robot, RobotParams
from utils import *


class DubinsRobot(Robot):
    def __init__(self, robot_params: RobotParams, log_enabled: bool = False) -> None:
        """Dubins robot is a robot that can move in a 2D space using a Dubins path.
        The state is [x_pos, y_pos, angle].
        The action is [v, omega].
        The dynamics is given by the following equations:
        x_pos_dot = v * cos(angle)
        y_pos_dot = v * sin(angle)
        angle_dot = omega
        """
        super().__init__(robot_params, log_enabled)

        # state is [x_pos, y_pos, angle]
        self.state = np.random.uniform(self.robot_params.state_min, self.robot_params.state_max)

        self.action_max = self.robot_params.action_max
        self.action_min = self.robot_params.action_min
        self.dt = robot_params.dt
        self.L = 0.4  # Lookahead distance for safety point

    def filter_action(self, action: np.ndarray) -> np.ndarray:
        assert action.shape == (self.robot_params.action_dim,), "Action must be a 2D array"
        v, omega = action

        if (v < self.action_min[0] or v > self.action_max[0]) and self.log_enabled:
            logging.warning(f"v must be between {self.action_min[0]} and {self.action_max[0]}")
        v = np.clip(v, self.action_min[0], self.action_max[0])

        if (omega < self.action_min[1] or omega > self.action_max[1]) and self.log_enabled:
            logging.warning(f"omega must be between {self.action_min[1]} and {self.action_max[1]}")
        omega = np.clip(omega, self.action_min[1], self.action_max[1])

        return np.array([v, omega])

    def open_loop_dynamics(self, state):
        return np.zeros_like(state)

    def control_jacobian(self, state):
        theta = state[STATE_THETA]
        return np.array(
            [
                [np.cos(theta), 0.0],
                [np.sin(theta), 0.0],
                [0.0, 1.0],
            ]
        )

    def open_loop_dynamics_jnp(self, state, time=0.0):
        return jnp.zeros_like(state)

    def control_jacobian_jnp(self, state, time=0.0):
        _, _, theta = state
        return jnp.array(
            [
                [jnp.cos(theta), 0.0],
                [jnp.sin(theta), 0.0],
                [0.0, 1.0],
            ]
        )

    def cbf_h_function(
        self, state: torch.Tensor, neighbors: list, d_safe: float, k1: float, dt: float, t: int
    ) -> torch.Tensor:
        K = state.shape[0]
        device = state.device
        if len(neighbors) == 0:
            return 1000.0 * torch.ones(K, device=device)

        p_i_center = state[:, [STATE_X, STATE_Y]]
        theta = state[:, STATE_THETA]
        
        # Ego safety point: p_i_safe = p_i_center + L * [cos(theta), sin(theta)]
        p_i_safe = p_i_center + self.L * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

        if isinstance(neighbors, list):
            neighbors_tensor = torch.stack(neighbors).to(device)
        else:
            neighbors_tensor = neighbors.to(device)

        p_j_center = neighbors_tensor[:, [STATE_X, STATE_Y]]
        theta_j = neighbors_tensor[:, STATE_THETA]
        
        # Neighbor safety point: p_j_safe = p_j_center + L * [cos(theta_j), sin(theta_j)]
        p_j_safe = p_j_center + self.L * torch.stack([torch.cos(theta_j), torch.sin(theta_j)], dim=1)

        # Dynamic neighbor velocity prediction:
        v_nominal = self.action_max[0] / 2.0
        v_j_x = v_nominal * torch.cos(theta_j)
        v_j_y = v_nominal * torch.sin(theta_j)
        v_j = torch.stack([v_j_x, v_j_y], dim=1)

        p_j_pred = p_j_safe + v_j * (t * dt)
        d_safe_barrier = d_safe + 2.0 * self.L + 0.2
        p_rel = p_i_safe.unsqueeze(1) - p_j_pred.unsqueeze(0)  # (K, N_neigh, 2)

        dist_sq = torch.sum(p_rel**2, dim=2)
        h0 = dist_sq - d_safe_barrier**2

        h_comp, _ = torch.min(h0, dim=1)
        return h_comp

    def cbf_safe_control(
        self,
        state: torch.Tensor,
        neighbors: list,
        d_safe: float,
        k1: float,
        k2: float,
        dt: float,
        t: int,
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> torch.Tensor:
        K = state.shape[0]
        device = state.device
        if len(neighbors) == 0:
            return torch.zeros(K, 2, device=device)

        p_i_center = state[:, [STATE_X, STATE_Y]]
        theta = state[:, STATE_THETA]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        v_nominal = self.action_max[0] / 2.0

        # Ego safety point: p_i_safe = p_i_center + L * [cos(theta), sin(theta)]
        p_i_safe = p_i_center + self.L * torch.stack([cos_t, sin_t], dim=1)

        if isinstance(neighbors, list):
            neighbors_tensor = torch.stack(neighbors).to(device)
        else:
            neighbors_tensor = neighbors.to(device)

        p_j_center = neighbors_tensor[:, [STATE_X, STATE_Y]]
        theta_j = neighbors_tensor[:, STATE_THETA]
        
        # Neighbor safety point: p_j_safe = p_j_center + L * [cos(theta_j), sin(theta_j)]
        p_j_safe = p_j_center + self.L * torch.stack([torch.cos(theta_j), torch.sin(theta_j)], dim=1)

        # Dynamic neighbor velocity prediction:
        v_j_x = v_nominal * torch.cos(theta_j)
        v_j_y = v_nominal * torch.sin(theta_j)
        v_j = torch.stack([v_j_x, v_j_y], dim=1)

        p_j_pred = p_j_safe + v_j * (t * dt)
        p_rel = p_i_safe.unsqueeze(1) - p_j_pred.unsqueeze(0)  # (K, N_neigh, 2)
        dist_sq = torch.sum(p_rel**2, dim=2)  # (K, N_neigh)
        d_safe_barrier = d_safe + 2.0 * self.L + 0.2
        h0_all = dist_sq - d_safe_barrier**2  # (K, N_neigh)

        # Find the critical (closest) neighbor for each batch item
        crit_indices = torch.argmin(h0_all, dim=1)
        batch_idx = torch.arange(K, device=device)

        h0_crit = h0_all[batch_idx, crit_indices]
        p_rel_crit = p_rel[batch_idx, crit_indices]
        v_j_crit = v_j[crit_indices]

        # Coefficients for action projection: R(theta, L)
        # Safety point speed = R * u where R = [cos(theta), -L*sin(theta); sin(theta), L*cos(theta)]
        # A = 2 * p_rel^T * R
        A_v = 2.0 * (p_rel_crit[:, 0] * cos_t + p_rel_crit[:, 1] * sin_t)
        A_w = 2.0 * (-self.L * p_rel_crit[:, 0] * sin_t + self.L * p_rel_crit[:, 1] * cos_t)
        A = torch.stack([A_v, A_w], dim=1)  # (K, 2)

        # 1st-degree HOCBF constraint: A * u >= B where B = 2 * p_rel_crit^T * v_j_crit - k1 * h0_crit
        B = 2.0 * torch.sum(p_rel_crit * v_j_crit, dim=1) - k1 * h0_crit

        # Project a preferred nominal forward velocity u_pref = [v_nominal, 0.0] instead of [0.0, 0.0]
        u_pref = torch.zeros(K, 2, device=device)
        u_pref[:, 0] = v_nominal

        # A * u_pref
        A_u_pref = A[:, 0] * u_pref[:, 0]

        # Violation of the preferred control: violation = B - A * u_pref
        violation = B - A_u_pref

        # Projection formula: u_safe = u_pref + (max(0, violation) / ||A||^2) * A
        A_norm_sq = torch.clamp(torch.sum(A**2, dim=1), min=1e-6)
        u_safe = u_pref + (torch.clamp(violation, min=0.0) / A_norm_sq).unsqueeze(-1) * A

        # Clamp analytically computed backup controls to physical bounds
        u_safe_clamped = torch.max(torch.min(u_safe, u_max), u_min)
        return u_safe_clamped

    def hj_safe_control(self, state: np.ndarray, value_grad: np.ndarray) -> np.ndarray:
        grad_x, grad_y, grad_theta = value_grad
        theta = state[STATE_THETA]
        if np.cos(theta) * grad_x + np.sin(theta) * grad_y > 0:
            safe_v = self.robot_params.action_max[0]
        else:
            safe_v = self.robot_params.action_min[0]

        if np.sign(grad_theta) > 0:
            safe_w = self.robot_params.action_max[1]
        else:
            safe_w = self.robot_params.action_min[1]

        return np.array([safe_v, safe_w])

    def bound_state(self, state: np.ndarray) -> np.ndarray:
        x_min = self.robot_params.state_min[STATE_X]
        y_min = self.robot_params.state_min[STATE_Y]
        x_max = self.robot_params.state_max[STATE_X]
        y_max = self.robot_params.state_max[STATE_Y]
        if (
            state[STATE_X] < x_min
            or state[STATE_X] > x_max
            or state[STATE_Y] < y_min
            or state[STATE_Y] > y_max
        ) and self.log_enabled:
            logging.warning(f"State is out of bounds: {state}")
        state[STATE_X] = np.clip(state[STATE_X], x_min, x_max)
        state[STATE_Y] = np.clip(state[STATE_Y], y_min, y_max)
        state[STATE_THETA] = np.mod(state[STATE_THETA], 2 * np.pi)
        return state

    def reset(self, state: np.ndarray) -> None:
        """State is [x_pos, y_pos, angle]"""
        assert state.shape == (self.robot_params.state_dim,), "State must be a 3D array"
        self.state = self.bound_state(state)

    def get_state(self) -> np.ndarray:
        """State is [x_pos, y_pos, angle]"""
        return self.state

    def dynamics(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Applies physical dynamics forecasts for a batch of states and actions.
        Leverages fully vectorized PyTorch RK4 math for Dubins kinematics.
        """
        import torch

        if states.ndim == 1:
            states = states.unsqueeze(0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        device = states.device
        dtype = states.dtype

        # Port control limits to active device
        u_min = torch.tensor(self.action_min, device=device, dtype=dtype)
        u_max = torch.tensor(self.action_max, device=device, dtype=dtype)

        # Clamp actions concurrently inside the GPU/CPU tensor block
        v_clamped = torch.clamp(actions[:, 0], u_min[0], u_max[0])
        omega_clamped = torch.clamp(actions[:, 1], u_min[1], u_max[1])

        dt = float(self.dt)

        def derivative(s, v, omega):
            # Returns [dx, dy, dtheta] derivative shape (K, 3)
            theta = s[:, STATE_THETA]
            dx = torch.cos(theta) * v
            dy = torch.sin(theta) * v
            dtheta = omega
            return torch.stack([dx, dy, dtheta], dim=-1)

        # Vectorized 4th-Order Runge-Kutta propagation
        k1 = derivative(states, v_clamped, omega_clamped)
        k2 = derivative(states + 0.5 * dt * k1, v_clamped, omega_clamped)
        k3 = derivative(states + 0.5 * dt * k2, v_clamped, omega_clamped)
        k4 = derivative(states + dt * k3, v_clamped, omega_clamped)

        next_states = states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Apply physical bounds to tensors
        x_min = float(self.robot_params.state_min[STATE_X])
        y_min = float(self.robot_params.state_min[STATE_Y])
        x_max = float(self.robot_params.state_max[STATE_X])
        y_max = float(self.robot_params.state_max[STATE_Y])

        # Clone and constrain components in-place
        out_states = next_states.clone()
        out_states[:, STATE_X] = torch.clamp(out_states[:, STATE_X], x_min, x_max)
        out_states[:, STATE_Y] = torch.clamp(out_states[:, STATE_Y], y_min, y_max)
        out_states[:, STATE_THETA] = torch.remainder(out_states[:, STATE_THETA], 2.0 * np.pi)

        return out_states


def plot_robot_trajectory(robot: DubinsRobot, trajectory: np.ndarray) -> None:
    ratio_window = robot.robot_params.state_max[STATE_X] / robot.robot_params.state_max[STATE_Y]
    if ratio_window > 1:
        f = plt.figure(figsize=(5, 5 / ratio_window))
    else:
        f = plt.figure(figsize=(5 * ratio_window, 5))
    ax = f.add_subplot(111)
    ax.scatter(trajectory[:, STATE_X], trajectory[:, STATE_Y])
    ax.set_xlim(0, robot.robot_params.state_max[STATE_X])
    ax.set_ylim(0, robot.robot_params.state_max[STATE_Y])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    print("🧪 Iniciando pruebas unitarias de DubinsRobot...")
    robot_params = RobotParams(
        action_dim=2,
        state_dim=3,
        action_max=[3.0, 4.0],
        action_min=[0.0, -4.0],
        state_max=[50.0, 50.0, 2 * np.pi],
        state_min=[0.0, 0.0, 0.0],
        dt=0.1,
    )
    robot = DubinsRobot(robot_params)

    # 1. Test bound_state & filter_action
    assert np.allclose(robot.filter_action(np.array([4.0, 5.0])), np.array([3.0, 4.0]))
    assert np.allclose(
        robot.bound_state(np.array([-1.0, 60.0, 3 * np.pi])), np.array([0.0, 50.0, np.pi])
    )
    print("   ✓ bound_state & filter_action: PASSED")

    # 2. Test control_jacobian & JAX dynamics
    state_dummy = jnp.array([10.0, 20.0, jnp.pi / 2])
    jac = robot.control_jacobian_jnp(state_dummy)
    assert np.allclose(jac, jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), atol=1e-5)
    print("   ✓ JAX Dynamics & Jacobians: PASSED")

    # 3. Test dynamics (PyTorch vectorized RK4)
    states_t = torch.tensor([[10.0, 10.0, 0.0]], dtype=torch.float32)
    actions_t = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    next_states = robot.dynamics(states_t, actions_t)
    assert np.allclose(next_states.numpy(), np.array([[10.2, 10.0, 0.0]]), atol=1e-3)
    print("   ✓ Vectorized RK4 Dynamics: PASSED")

    # 4. Test cbf_h_function with neighbors
    ego_states = torch.tensor([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0]], dtype=torch.float32)
    # Vecinos representados como tensores de estado [x, y, theta]
    neighbors = [
        torch.tensor([11.0, 10.0, np.pi], dtype=torch.float32),  # Cercano (1m)
        torch.tensor([20.0, 10.0, 0.0], dtype=torch.float32),  # Lejano
    ]
    h_vals = robot.cbf_h_function(ego_states, neighbors, d_safe=1.5, k1=2.0, dt=0.1, t=0)
    # safety points under L=0.4 are at distance 0.2m => h0 = 0.04 - 6.25 = -6.21
    assert torch.allclose(h_vals, torch.tensor([-6.21, -6.21]), atol=1e-3)
    print("   ✓ cbf_h_function: PASSED")

    # 5. Test cbf_safe_control (1st-degree Lookahead safety point HOCBF)
    u_min = torch.tensor(robot_params.action_min, dtype=torch.float32)
    u_max = torch.tensor(robot_params.action_max, dtype=torch.float32)
    
    # Caso 5A: Colisión simétrica directa (safety points coincidentes => A=0, frena a cero)
    u_safe_sym = robot.cbf_safe_control(
        state=torch.tensor([[10.0, 10.0, 0.0]], dtype=torch.float32),
        neighbors=neighbors,
        d_safe=1.5,
        k1=2.0,
        k2=2.0,
        dt=0.1,
        t=0,
        u_min=u_min,
        u_max=u_max,
    )
    print(f"   [Simétrico] Control calculado (frena): {u_safe_sym.numpy().tolist()}")
    assert np.isclose(u_safe_sym[0, 0].item(), 0.0, atol=1e-1) # frenado a cero
    
    # Caso 5B: Colisión descentrada/offset (debe girar activamente)
    neighbors_offset = [
        torch.tensor([11.0, 10.3, np.pi], dtype=torch.float32)
    ]
    u_safe_offset = robot.cbf_safe_control(
        state=torch.tensor([[10.0, 10.0, 0.0]], dtype=torch.float32),
        neighbors=neighbors_offset,
        d_safe=1.5,
        k1=2.0,
        k2=2.0,
        dt=0.1,
        t=0,
        u_min=u_min,
        u_max=u_max,
    )
    print(f"   [Descentrado] Control calculado (gira): {u_safe_offset.numpy().tolist()}")
    # Debe tener una velocidad angular omega distinta de cero para esquivar
    assert abs(u_safe_offset[0, 1].item()) > 0.01
    print("   ✓ cbf_safe_control (1st-degree Lookahead Point HOCBF): PASSED")

    # 6. Test hj_safe_control
    u_hj = robot.hj_safe_control(np.array([10.0, 10.0, 0.0]), np.array([-1.0, 0.0, 1.0]))
    # grad_x < 0, grad_y = 0, grad_theta > 0 => safe_v = u_min[0]=0, safe_w = u_max[1]=4
    assert np.allclose(u_hj, np.array([0.0, 4.0]))
    print("   ✓ hj_safe_control: PASSED")

    print("🎉 Todas las pruebas unitarias pasaron con éxito.")
