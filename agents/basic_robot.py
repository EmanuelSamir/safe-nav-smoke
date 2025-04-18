import numpy as np
import matplotlib.pyplot as plt
import warnings

from src.utils import *

class RobotParams:
    world_x_size: int = 80
    world_y_size: int = 30

    v_max: float = 5.0
    omega_max: float = 4.0

    v_min: float = 0.0
    omega_min: float = -4.0

    dt: float = 0.1

class BasicRobot:
    def __init__(self, robot_params: RobotParams) -> None:
        self.robot_params = robot_params
        # state is [x_pos, y_pos, angle]
        self.pos_x = 0
        self.pos_y = 0
        self.angle = 0
    
    def dynamic_step(self, v: float, omega: float) -> None:
        if v < self.robot_params.v_min or v > self.robot_params.v_max:
            warnings.warn(f"v must be between {self.robot_params.v_min} and {self.robot_params.v_max}")
            v = np.clip(v, self.robot_params.v_min, self.robot_params.v_max)

        if omega < self.robot_params.omega_min or omega > self.robot_params.omega_max:
            warnings.warn(f"omega must be between {self.robot_params.omega_min} and {self.robot_params.omega_max}")
            omega = np.clip(omega, self.robot_params.omega_min, self.robot_params.omega_max)

        # Euler integration.
        # TODO: Use Runge-Kutta method.
        self.pos_x += v * np.cos(self.angle) * self.robot_params.dt
        self.pos_y += v * np.sin(self.angle) * self.robot_params.dt

        self.pos_x, self.pos_y = clip_world(self.pos_x, self.pos_y, self.robot_params.world_x_size, self.robot_params.world_y_size)

        self.angle += omega * self.robot_params.dt
        self.angle = np.mod(self.angle, 2 * np.pi)

    def reset(self, pos_x: float, pos_y: float, angle: float) -> None:
        self.pos_x, self.pos_y = clip_world(pos_x, pos_y, self.robot_params.world_x_size, self.robot_params.world_y_size)
        self.angle = np.mod(angle, 2 * np.pi)

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.angle = angle

    def get_state(self) -> np.ndarray:
        return np.array([self.pos_x, self.pos_y, self.angle])
    

def plot_robot_trajectory(robot: BasicRobot, trajectory: np.ndarray) -> None:
    ratio_window = robot_params.world_x_size / robot_params.world_y_size
    if ratio_window > 1:
        f = plt.figure(figsize=(5 , 5/ ratio_window))
    else:
        f = plt.figure(figsize=(5 * ratio_window, 5 ))
    ax = f.add_subplot(111)
    ax.scatter(trajectory[:, 0], trajectory[:, 1])
    ax.set_xlim(0, robot_params.world_x_size)
    ax.set_ylim(0, robot_params.world_y_size)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

if __name__ == "__main__":
    robot_params = RobotParams()
    robot = BasicRobot(robot_params)
    robot.reset(40., 10., 0.)
    trajectory = np.zeros((100, 3))
    for i in range(100):
        robot.dynamic_step(4., 0.4)
        trajectory[i, :] = robot.get_state()
    plot_robot_trajectory(robot, trajectory)