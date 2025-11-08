import numpy as np
import matplotlib.pyplot as plt
import logging

from src.utils import *

from agents.basic_robot import Robot, RobotParams

class DubinsRobot(Robot):
    def __init__(self, robot_params: RobotParams, log_enabled: bool = False) -> None:
        """
        Dubins robot is a robot that can move in a 2D space using a Dubins path.
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

    def filter_action(self, action: np.ndarray) -> None:
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
        return np.array([
            [np.cos(state[2]), 0.0],
            [np.sin(state[2]), 0.0],
            [0.0, 1.0],
        ])

    def bound_state(self, state: np.ndarray) -> np.ndarray:
        if (state[0] < 0 or state[0] > self.robot_params.state_max[0] or state[1] < 0 or state[1] > self.robot_params.state_max[1]) and self.log_enabled:
            logging.warning(f"State is out of bounds: {state}")
        state[0] = np.clip(state[0], 0, self.robot_params.state_max[0])
        state[1] = np.clip(state[1], 0, self.robot_params.state_max[1])
        state[2] = np.mod(state[2], 2 * np.pi)
        return state

    def reset(self, state: np.ndarray) -> None:
        """
        state is [x_pos, y_pos, angle]
        """
        assert state.shape == (self.robot_params.state_dim,), "State must be a 3D array"
        self.state = self.bound_state(state)

    def get_state(self) -> np.ndarray:
        """
        state is [x_pos, y_pos, angle]
        """
        return self.state

def plot_robot_trajectory(robot: DubinsRobot, trajectory: np.ndarray) -> None:
    ratio_window = robot.robot_params.state_max[0] / robot.robot_params.state_max[1]
    if ratio_window > 1:
        f = plt.figure(figsize=(5 , 5/ ratio_window))
    else:
        f = plt.figure(figsize=(5 * ratio_window, 5 ))
    ax = f.add_subplot(111)
    ax.scatter(trajectory[:, 0], trajectory[:, 1])
    ax.set_xlim(0, robot.robot_params.state_max[0])
    ax.set_ylim(0, robot.robot_params.state_max[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

if __name__ == "__main__":
    robot_params = RobotParams()
    robot = DubinsRobot(robot_params)
    robot.reset(np.array([40., 10., 0.]))
    trajectory = np.zeros((100, 3))
    for i in range(100):
        robot.dynamic_step(np.array([2., 0.4]))
        trajectory[i, :] = robot.get_state()
    plot_robot_trajectory(robot, trajectory)
