import numpy as np
import matplotlib.pyplot as plt
import logging

from src.utils import *

from agents.basic_robot import Robot, RobotParams
from agents.dubins_robot import DubinsRobot

class DubinsRobotFixedVelocity(DubinsRobot):
    def __init__(self, robot_params: RobotParams, log_enabled: bool = False, velocity: float = 8.0) -> None:
        super().__init__(robot_params, log_enabled)
        # Dummy literal velocity. TODO: Remove this.
        self.velocity = velocity

    def filter_action(self, action: np.ndarray) -> None:
        assert action.shape == (1,), "Action must be a 1D array, but got shape: " + str(action.shape)
        omega = action[0]
        if (omega < self.action_min[0] or omega > self.action_max[0]) and self.log_enabled:
            logging.warning(f"omega must be between {self.action_min[1]} and {self.action_max[1]}")
        omega = np.clip(omega, self.action_min[0], self.action_max[0])
        return np.array([omega])

    def dynamic_step(self, action: np.ndarray) -> None:
        return super().dynamic_step(action)

    def open_loop_dynamics(self, state):
        return np.array([self.velocity * np.cos(state[2]), self.velocity * np.sin(state[2]), 0.0])

    def control_jacobian(self, state):
        return np.array([[0.], [0.], [1.]])

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
    robot = DubinsRobotFixedVelocity(robot_params)
    robot.reset(np.array([40., 10., 0.]))
    trajectory = np.zeros((100, 3))
    for i in range(100):
        robot.dynamic_step(np.array([0.2]))
        trajectory[i, :] = robot.get_state()
    plot_robot_trajectory(robot, trajectory)
