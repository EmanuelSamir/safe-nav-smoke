import logging
import torch

from src.agents.basic_robot import Robot, RobotParams

# State indices constants for 2D kinematics
X_DIM = 0
Y_DIM = 1
XY_DIM = [X_DIM, Y_DIM]
ANGLE_DIM = 2

V_DIM = 0
OMEGA_DIM = 1


class DubinsRobot(Robot):
    def __init__(self, params: RobotParams, log_enabled: bool = False) -> None:
        """Dubins robot is a robot that can move in a 2D space using a Dubins path.

        The state is [x_pos, y_pos, angle].
        The action is [v, omega].
        The dynamics is given by the following equations:
        x_pos_dot = v * cos(angle)
        y_pos_dot = v * sin(angle)
        angle_dot = omega
        """
        super().__init__(params, log_enabled)

        self.state_min = torch.tensor(
            self.params.state_min, device=params.device, dtype=torch.float32
        )
        self.state_max = torch.tensor(
            self.params.state_max, device=params.device, dtype=torch.float32
        )
        self.action_min = torch.tensor(
            self.params.action_min, device=params.device, dtype=torch.float32
        )
        self.action_max = torch.tensor(
            self.params.action_max, device=params.device, dtype=torch.float32
        )

        # state is [x_pos, y_pos, angle]
        random_val = torch.rand(self.params.state_dim, device=params.device, dtype=torch.float32)
        self.state = random_val * (self.state_max - self.state_min) + self.state_min

    def reset(self, state: torch.Tensor) -> None:
        assert state.shape == (self.params.state_dim,), "State must be a 3D array"
        self.state = self.bound_state(state)

    def get_state(self) -> torch.Tensor:
        return self.state

    def bound_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.log_enabled and (
            torch.any(state[..., XY_DIM] < self.state_min[XY_DIM])
            or torch.any(state[..., XY_DIM] > self.state_max[XY_DIM])
        ):
            logging.warning(f"State is out of bounds: {state}")

        state[..., XY_DIM] = torch.clamp(
            state[..., XY_DIM], self.state_min[XY_DIM], self.state_max[XY_DIM]
        )
        state[..., ANGLE_DIM] = torch.remainder(state[..., ANGLE_DIM], 2 * torch.pi)
        return state

    def filter_action(self, action: torch.Tensor) -> torch.Tensor:
        assert action.shape[-1] == self.params.action_dim, (
            "Action last dimension must match action_dim"
        )
        if self.log_enabled and (
            torch.any(action < self.action_min) or torch.any(action > self.action_max)
        ):
            logging.warning(f"Action {action} out of bounds [{self.action_min}, {self.action_max}]")
        return torch.clamp(action, self.action_min, self.action_max)

    def dynamics(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        actions = self.filter_action(actions)

        dt = self.params.dt

        def derivative(s, a):
            # Returns [dx, dy, dtheta] derivative shape (K, 3)
            theta = s[:, ANGLE_DIM]
            v = a[:, V_DIM]
            omega = a[:, OMEGA_DIM]
            dx = torch.cos(theta) * v
            dy = torch.sin(theta) * v
            dtheta = omega
            return torch.stack([dx, dy, dtheta], dim=-1)

        # Vectorized 4th-Order Runge-Kutta propagation
        k1 = derivative(states, actions)
        k2 = derivative(states + 0.5 * dt * k1, actions)
        k3 = derivative(states + 0.5 * dt * k2, actions)
        k4 = derivative(states + dt * k3, actions)

        next_states = states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        out_states = self.bound_state(next_states)

        return out_states


def run_tests() -> None:
    robot_params = RobotParams(
        action_min=[0.0, -1.0],
        action_max=[1.0, 1.0],
        action_dim=2,
        state_dim=3,
        state_min=[0.0, 0.0, 0.0],
        state_max=[35.0, 35.0, 6.28],
        dt=0.1
    )
    robot = DubinsRobot(robot_params)

    # 1. Test bound_state & filter_action
    v_max = robot_params.action_max[V_DIM]
    w_max = robot_params.action_max[OMEGA_DIM]
    assert torch.allclose(
        robot.filter_action(torch.tensor([v_max + 1.0, w_max + 1.0])),
        torch.tensor([v_max, w_max]),
    )

    # Test bounding with values derived from the configuration instead of hardcoded literals
    # We exceed state_max on y-axis and test angle wrapping
    # robot_params.state_max[1] comes from the config file (e.g. world_size.1)
    expected_y = robot_params.state_max[1]
    expected_state = torch.tensor([0.0, expected_y, torch.pi], dtype=torch.float32)
    assert torch.allclose(
        robot.bound_state(torch.tensor([-1.0, 60.0, 3 * torch.pi])),
        expected_state,
    )
    print("bound_state & filter_action: PASSED")

    # 2. Test dynamics (PyTorch vectorized RK4)
    states_t = torch.tensor([[10.0, 10.0, 0.0]], dtype=torch.float32)
    actions_t = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    next_states = robot.dynamics(states_t, actions_t)
    expected_x = 10.0 + 2.0 * float(robot_params.dt)
    expected_next = torch.tensor([[expected_x, 10.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(next_states, expected_next, atol=1e-3)
    print("Vectorized RK4 Dynamics: PASSED")


if __name__ == "__main__":
    run_tests()
