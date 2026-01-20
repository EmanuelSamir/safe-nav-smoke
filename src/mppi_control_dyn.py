# mppi_planner.py
import logging
import warnings
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.mppi import MPPI
from src.utils import world_to_index
from agents.dubins_robot import DubinsRobot
from agents.dubins_robot_fixed_velocity import DubinsRobotFixedVelocity
from agents.unicycle_robot import UnicycleRobot
from agents.basic_robot import RobotParams
from src.risk_map_builder import get_value_in_map_from_coords

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ==========================================================
# CONFIG DATACLASS
# ==========================================================
@dataclass
class MPPIControlParams:
    num_samples: int = 50
    horizon: int = 14
    device: str = "cpu"
    lambda_: float = 3.0
    noise_abs_cost: bool = False
    alpha_noise_sigma: float = 10.0

# ==========================================================
# NOMINAL CONTROL WRAPPER
# ==========================================================
class MPPIControlDyn:
    """
    High-level wrapper for MPPI-based control of a Dubins robot.
    Handles map setup, cost computation, and control loop.
    """

    def __init__(
        self,
        robot_params: RobotParams,
        robot_type: str,
        resolution: float,
        goal_thresh: float = 0.1,
        device="cpu",
        dtype=torch.float32,
        mppi_params: MPPIControlParams = MPPIControlParams(),
        dt=0.1,
    ):
        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.resolution = resolution
        self.robot_params = robot_params
        if robot_type == "dubins2d":
            self.robot = DubinsRobot(robot_params)
        elif robot_type == "dubins2d_fixed_velocity":
            self.robot = DubinsRobotFixedVelocity(robot_params)
        elif robot_type == "unicycle2d":
            self.robot = UnicycleRobot(robot_params)
        self.params = mppi_params

        self._goal = None
        self._maps = deque(maxlen=self.params.horizon)
        self._goal_thresh = goal_thresh

        self.planner = self._init_mppi_planner()

    # ======================================================
    #  DYNAMICS INTERFACE
    # ======================================================
    def dynamics(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        """
        Applies Dubins dynamics for a batch of states and actions.
        """
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        next_states = []
        for s, a in zip(states, actions):
            self.robot.reset(s)
            self.robot.dynamic_step(a.cpu().numpy())
            next_states.append(self.robot.get_state())

        return torch.stack(next_states).to(self.device)

    # ======================================================
    #  MPPI PLANNER INITIALIZATION
    # ======================================================
    def _init_mppi_planner(self) -> MPPI:
        """
        Configure and instantiate the MPPI planner.
        """
        sigma = self.params.alpha_noise_sigma * np.diag(
            self.robot.robot_params.action_max - self.robot.robot_params.action_min
        )

        config = dict(
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            terminal_state_cost=self.terminal_state_cost,
            nx=self.robot.robot_params.state_dim,
            noise_sigma=torch.tensor(sigma, dtype=self.dtype, device=self.device),
            num_samples=self.params.num_samples,
            horizon=self.params.horizon,
            device=self.device,
            u_min=torch.tensor(
                self.robot.robot_params.action_min, dtype=self.dtype, device=self.device
            ),
            u_max=torch.tensor(
                self.robot.robot_params.action_max, dtype=self.dtype, device=self.device
            ),
            lambda_=self.params.lambda_,
            noise_abs_cost=self.params.noise_abs_cost,
            step_dependent_dynamics=True,
        )
        return MPPI(**config)

    # ======================================================
    #  STATE AND MAP MANAGEMENT
    # ======================================================
    def set_goal(self, goal_position):
        self._goal = torch.tensor(goal_position, dtype=self.dtype, device=self.device)

    def set_maps(self, maps: deque):
        """
        Sets a deque of tuple of (coords, flatten_risk_map) (0 = low risk -> high risk).
        coords: np.ndarray, shape (n, 2)
        flatten_risk_map: np.ndarray, shape (n,)
        The deque is used to store the maps for future planning steps.
        """
        assert len(maps) == self.params.horizon, "Maps deque must have the same length as the horizon"
        for coords, flatten_risk_map in maps:
            assert coords.shape[1] == 2, "Coords must be a 2D array"
            assert flatten_risk_map.shape[0] == coords.shape[0], "Flatten risk map must have the same number of rows as coords"
            self._maps.append((coords, flatten_risk_map))

    def set_state(self, state):
        """
        Sets the true or simulated robot state (x, y, theta).
        """
        self.robot.reset(state)

    # ======================================================
    #  CONTROL COMPUTATION
    # ======================================================
    def get_command(self) -> torch.Tensor:
        """
        Compute the next control input given the current state.
        """
        if self._goal is None:
            logger.warning("Goal not set. Returning zero action.")
            return torch.zeros_like(
                torch.tensor(self.robot.robot_params.action_min, device=self.device)
            )

        x, y, theta = self.robot.get_state()
        state = torch.tensor([x, y, theta], dtype=self.dtype, device=self.device)
        dist_to_goal = torch.norm(state[:2] - self._goal)

        if dist_to_goal < self._goal_thresh:
            return torch.zeros_like(state[:2])

        command = self.planner.command(state)
        return command

    # ======================================================
    #  TRAJECTORY HANDLING
    # ======================================================
    def get_sampled_trajectories(self):
        """
        Returns sampled trajectories (K, T, nx) if available.
        """
        if getattr(self.planner, "synthetic_states", None) is None:
            logger.warning("No trajectories available yet.")
            return None
        return self.planner.synthetic_states.detach().cpu()

    # ======================================================
    #  COST FUNCTIONS
    # ======================================================
    def _compute_risk_cost(self, states: torch.Tensor, t: int) -> torch.Tensor:
        """
        Compute collision cost for states in world coordinates.
        states: torch.Tensor, shape (n, 2)
        t: int
        Returns:
            torch.Tensor, shape (n,)
        """
        if len(self._maps) == 0:
            return torch.zeros(states.shape[0], dtype=self.dtype, device=self.device)
            # raise ValueError("Maps not set. Please call set_maps() before planning.")

        coords, flatten_risk_map = self._maps[t]

        risk_values = get_value_in_map_from_coords(states[:, :2], coords, flatten_risk_map)

        return risk_values

    def running_cost(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        """
        Weighted sum of goal distance and risk cost.
        """
        w=(1.0, 20.0)
        dist_cost = torch.norm(states[:, :2] - self._goal, dim=1)
        risk_cost = self._compute_risk_cost(states, t)
        return w[0] * dist_cost + w[1] * risk_cost

    def terminal_state_cost(self, states: torch.Tensor) -> torch.Tensor:
        """
        Strong negative reward for reaching the goal safely.
        """
        K, T, nx = states.shape
        goal_reached = torch.norm(states[:, :, :2] - self._goal, dim=2) < self._goal_thresh
        cost = torch.zeros(K, dtype=self.dtype, device=self.device)

        for k in range(K):
            if goal_reached[k].any():
                cost[k] = -100.0  # safe goal reward
        return cost

    # ======================================================
    #  VISUALIZATION
    # ======================================================
    def visualize_rollouts(self, ax):
        """
        Plot all sampled trajectories and the weighted mean path.
        """
        trajectories = self.get_sampled_trajectories()
        if trajectories is None:
            return

        omega = self.planner.omega.detach().cpu()
        weighted_traj = (omega[:, None, None] * trajectories).sum(dim=0).numpy()

        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.3, linewidth=0.5)

        ax.plot(weighted_traj[:, 0], weighted_traj[:, 1], color="blue", linewidth=2)


if __name__ == "__main__":
    robot_params = RobotParams.load_from_yaml("agents/dubins2d_cfg.yaml")
    robot_params.state_max = np.array([50, 35]) # (x, y)
    robot_params.state_min = np.array([0, 0])
    resolution = 1.0
    goal_thresh = 1.0
    device = 'cpu'
    dtype = torch.float32
    dt = 0.1

    state = torch.tensor([1, 1, 0.0])
    nominal_control = MPPIControlDyn(robot_params, "dubins2d", resolution, goal_thresh, device, dtype, dt)
    nominal_control.set_state(state)
    nominal_control.set_goal([25.0, 10.0]) # (x, y)

    H, W = 35, 50
    map_data = np.zeros((H, W), dtype=bool)  # y, x
    map_data[5:20, 10:20] = True

    # generar coordenadas (x, y)
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y, indexing='xy')
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (N, 2)

    flatten_risk_map = map_data.ravel()
    nominal_control.set_maps(deque([(coords, flatten_risk_map)] * 10))

    simulator = DubinsRobot(robot_params)
    simulator.reset(state)

    f, ax = plt.subplots()

    ax.imshow(map_data, origin='lower', cmap='gray', extent=[0, 50, 0, 35])

    for i in tqdm(range(200)):
        for line in ax.lines:
            line.remove()

        ax.scatter(state[0].item(), state[1].item(), c='blue', marker='.', s=1)
        
        u = nominal_control.get_command()
        nominal_control.visualize_rollouts(ax)

        state = simulator.get_state()
        u_sim = u.cpu().numpy()
        simulator.dynamic_step(u_sim)
        state = simulator.get_state()
        nominal_control.set_state(state)
        
        plt.draw()
        plt.pause(0.01)



