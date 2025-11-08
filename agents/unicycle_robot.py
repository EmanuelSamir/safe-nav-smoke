import numpy as np
import matplotlib.pyplot as plt
import logging
from src.utils import *
from agents.basic_robot import Robot, RobotParams


class UnicycleRobot(Robot):
    def __init__(self, robot_params: RobotParams, log_enabled: bool = False) -> None:
        """
        Unicycle robot that moves in 2D space with a dynamic unicycle model.

        State:  [x, y, theta, v]
        Action: [a, omega]
        """
        super().__init__(robot_params, log_enabled)

        self.state = np.random.uniform(self.robot_params.state_min, self.robot_params.state_max)
        self.dt = robot_params.dt

        self.action_max = np.array(self.robot_params.action_max)
        self.action_min = np.array(self.robot_params.action_min)

    # -------------------------------------------------------------
    # ACTION FILTERING
    # -------------------------------------------------------------
    def filter_action(self, action: np.ndarray) -> np.ndarray:
        assert action.shape == (self.robot_params.action_dim,), "Action must have shape (2,)"
        a, omega = action

        if (a < self.action_min[0] or a > self.action_max[0]) and self.log_enabled:
            logging.warning(f"a must be between {self.action_min[0]} and {self.action_max[0]}")
        a = np.clip(a, self.action_min[0], self.action_max[0])

        if (omega < self.action_min[1] or omega > self.action_max[1]) and self.log_enabled:
            logging.warning(f"omega must be between {self.action_min[1]} and {self.action_max[1]}")
        omega = np.clip(omega, self.action_min[1], self.action_max[1])

        return np.array([a, omega])

    # -------------------------------------------------------------
    # DYNAMICS STEP (RK4 Integration)
    # -------------------------------------------------------------
    def dynamic_step(self, action: np.ndarray) -> np.ndarray:
        """
        Integrates dynamics using RK4.
        state: [x, y, theta, v]
        action: [a, omega]
        """
        action = self.filter_action(action)

        def f(state, action):
            x, y, theta, v = state
            a, omega = action
            return np.array([
                v * np.cos(theta),   # x_dot
                v * np.sin(theta),   # y_dot
                omega,               # theta_dot
                a                    # v_dot
            ])

        # Runge-Kutta 4th order integration
        k1 = f(self.state, action)
        k2 = f(self.state + 0.5 * self.dt * k1, action)
        k3 = f(self.state + 0.5 * self.dt * k2, action)
        k4 = f(self.state + self.dt * k3, action)

        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.state = self.bound_state(self.state)
        return self.state

    # -------------------------------------------------------------
    # STATE BOUNDING
    # -------------------------------------------------------------
    def bound_state(self, state: np.ndarray) -> np.ndarray:
        """
        Keeps the robot within the defined workspace and wraps angle.
        """
        if ((state[0] < 0 or state[0] > self.robot_params.state_max[0] or
             state[1] < 0 or state[1] > self.robot_params.state_max[1]) and self.log_enabled):
            logging.warning(f"State out of bounds: {state}")

        state[0] = np.clip(state[0], 0, self.robot_params.state_max[0])
        state[1] = np.clip(state[1], 0, self.robot_params.state_max[1])
        state[2] = np.mod(state[2], 2 * np.pi)  # wrap angle
        state[3] = np.clip(state[3], self.robot_params.action_min[0], self.robot_params.action_max[0])
        return state

    # -------------------------------------------------------------
    # INTERFACE METHODS
    # -------------------------------------------------------------
    def reset(self, state: np.ndarray) -> None:
        assert state.shape == (self.robot_params.state_dim,), "State must have shape (4,)"
        self.state = self.bound_state(state)

    def get_state(self) -> np.ndarray:
        return self.state

    # def open_loop_dynamics(self, state):
    #     return np.array([state[3] * np.cos(state[2]), state[3] * np.sin(state[2]), 0.0, 0.0])

    # def control_jacobian(self, state):
    #     return np.array([
    #         [np.cos(state[2]), 0.0],
    #         [np.sin(state[2]), 0.0],
    #         [0.0, 1.0],
    #         [0.0, 0.0],
    #     ])


# -------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------
def plot_robot_trajectory(robot: UnicycleRobot, trajectory: np.ndarray) -> None:
    ratio = robot.robot_params.state_max[0] / robot.robot_params.state_max[1]
    fig = plt.figure(figsize=(5, 5/ratio) if ratio > 1 else (5*ratio, 5))
    ax = fig.add_subplot(111)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', lw=1)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', label='start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', label='end')
    ax.set_xlim(0, robot.robot_params.state_max[0])
    ax.set_ylim(0, robot.robot_params.state_max[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.show()


# -------------------------------------------------------------
# TEST
# -------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    robot_params = RobotParams(
        state_dim=4,
        action_dim=2,
        state_min=np.array([0., 0., 0., 0.]),
        state_max=np.array([50., 30., 2*np.pi, 5.]),
        action_min=np.array([-3., -2.]),   # [a_min, omega_min]
        action_max=np.array([3., 2.]),     # [a_max, omega_max]
        dt=0.1,
    )

    robot = UnicycleRobot(robot_params)
    robot.reset(np.array([10., 5., 0., 0.]))  # start at rest

    trajectory = np.zeros((200, 4))
    for i in range(200):
        robot.dynamic_step(np.array([0.5, 0.01]))  # accelerate forward and turn
        trajectory[i, :] = robot.get_state()

    plot_robot_trajectory(robot, trajectory)
