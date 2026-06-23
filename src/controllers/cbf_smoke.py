import os
import sys

# Ensure project root and src/ are in sys.path before executing imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_path = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import logging
import numpy as np
import torch
import skfmm

from src.utils.optimization import solve_qp_batch_pytorch
from src.env.smoke_env import EnvConfig
from src.agents.basic_robot import RobotParams
from src.controllers.schemas import CBFSmokeConfig

logger = logging.getLogger(__name__)


def solve_qp_numpy(
    u_nom: np.ndarray,
    R_diag: np.ndarray,
    A: np.ndarray,
    C: float,
    u_min: np.ndarray,
    u_max: np.ndarray,
    max_iters: int = 20,
    rho: float = 10.0
) -> np.ndarray:
    """NumPy wrapper that converts arguments to PyTorch tensors,
    calls solve_qp_batch_pytorch, and converts the result back to NumPy.
    Keeps NumPy-to-PyTorch interface details encapsulated locally.
    """
    device = "cpu"
    u_nom_t = torch.tensor(u_nom, dtype=torch.float32, device=device)
    R_diag_t = torch.tensor(R_diag, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    C_t = torch.tensor(C, dtype=torch.float32, device=device)
    u_min_t = torch.tensor(u_min, dtype=torch.float32, device=device)
    u_max_t = torch.tensor(u_max, dtype=torch.float32, device=device)

    u_t = solve_qp_batch_pytorch(
        u_nom_t, R_diag_t, A_t, C_t, u_min_t, u_max_t, max_iters, rho
    )
    return u_t.detach().cpu().numpy()


class CBFSmokeController:
    """Single-agent Control Barrier Function controller for navigating through smoke."""

    def __init__(
        self,
        config: CBFSmokeConfig,
        env_params: EnvConfig,
        robot_params: RobotParams,
        goal: np.ndarray,
        num_agents: int = 1,
    ):
        # Assert to guarantee this controller only runs with a single agent
        assert num_agents == 1, f"CBFSmokeController only supports a single agent, but got num_agents={num_agents}"

        self.config = config
        self.env_params = env_params
        self.robot_params = robot_params
        self.goal = goal

        self.h_discrete_artifacts = {}
        self._grid_dims = None  # Caches (H, W, N) grid size

        # Map configurations
        self.R_diag = np.array(self.config.R_diag)
        self.rho = self.config.rho
        self.u_min = np.array(self.robot_params.action_min)
        self.u_max = np.array(self.robot_params.action_max)
        self.n_u = 2
        self.n_x = 3
        self.k1 = self.config.k1
        self.k2 = self.config.k2
        self.smoke_threshold = self.config.smoke_threshold
        self.margin = self.config.margin

        assert self.R_diag.shape == (self.n_u,), f"R_diag must be of shape ({self.n_u},)"

    def nominal_control(self, state: np.ndarray) -> np.ndarray:
        """Computes nominal control driving the robot toward the goal.
        state: [x, y, angle]
        """
        v_max = self.u_max[0]
        w_min = self.u_min[1]
        w_max = self.u_max[1]

        location = state[:2]
        angle = state[2]

        desired_angle = np.arctan2(self.goal[1] - location[1], self.goal[0] - location[0])
        e_angle = desired_angle - angle
        e_angle = (e_angle + np.pi) % (2 * np.pi) - np.pi

        if np.isclose(e_angle, 0.0, atol=1e-2):
            w = 0.0
        elif np.sign(e_angle) > 0:
            w = w_max
        else:
            w = w_min

        u = np.array([v_max, w])
        return u

    def update_h_discrete(
        self, smoke_values: np.ndarray, smoke_positions: np.ndarray, robot_pose: np.ndarray
    ):
        """Builds distance map using fast marching method from smoke grid values."""
        if torch.is_tensor(smoke_values):
            smoke_values = smoke_values.detach().cpu().numpy()
        else:
            smoke_values = np.asarray(smoke_values)

        if torch.is_tensor(smoke_positions):
            smoke_positions = smoke_positions.detach().cpu().numpy()
        else:
            smoke_positions = np.asarray(smoke_positions)

        if torch.is_tensor(robot_pose):
            robot_pose = robot_pose.detach().cpu().numpy()
        else:
            robot_pose = np.asarray(robot_pose)

        assert smoke_values.ndim == 1, "Smoke values must be a 1D array"
        assert smoke_positions.ndim == 2, "Smoke positions must be a 2D array"
        assert smoke_values.shape[0] == smoke_positions.shape[0], (
            "Smoke values and positions must have the same length"
        )

        N = smoke_values.shape[0]
        if N == 0:
            self.h_discrete_artifacts = {}
            return

        x_min = np.min(smoke_positions[:, 0])
        x_max = np.max(smoke_positions[:, 0])
        y_min = np.min(smoke_positions[:, 1])
        y_max = np.max(smoke_positions[:, 1])

        # Optimize: compute/factorize H and W once and cache them
        if self._grid_dims is None or self._grid_dims[2] != N:
            ratio = (x_max - x_min) / max(1e-5, y_max - y_min)
            Hs = [int(np.floor(np.sqrt(N / ratio))), int(np.ceil(np.sqrt(N / ratio)))]
            Ws = [int(np.floor(np.sqrt(N * ratio))), int(np.ceil(np.sqrt(N * ratio)))]
            found = False
            for h in Hs:
                for w in Ws:
                    if h > 0 and w > 0 and h * w == N:
                        self._grid_dims = (h, w, N)
                        found = True
                        break
                if found:
                    break
            if not found:
                h = int(np.round(np.sqrt(N)))
                w = N // h
                self._grid_dims = (h, w, N)

        H, W, _ = self._grid_dims

        smoke_map = smoke_values.reshape(H, W)
        # 1 is free, 0 is occupied (unsafe)
        occupancy_grid = (smoke_map < self.smoke_threshold).astype(int)

        if np.all(occupancy_grid == 1):
            self.h_discrete_artifacts = {}
            return

        # Compute physical resolution (meters per cell)
        dx_grid = (x_max - x_min) / max(1, W - 1)
        dy_grid = (y_max - y_min) / max(1, H - 1)

        # Compute signed distance map using skfmm
        distance_map = skfmm.distance(occupancy_grid - 0.5, dx=(dy_grid, dx_grid)) - self.margin

        self.h_discrete_artifacts["x_range"] = [x_min, x_max]
        self.h_discrete_artifacts["y_range"] = [y_min, y_max]
        self.h_discrete_artifacts["W"] = W
        self.h_discrete_artifacts["H"] = H
        self.h_discrete_artifacts["dx"] = dx_grid
        self.h_discrete_artifacts["dy"] = dy_grid
        self.h_discrete_artifacts["distance_map"] = distance_map

    def _nearest_index(self, pose: np.ndarray) -> tuple[int, int]:
        x, y, th = pose
        x_min, x_max = self.h_discrete_artifacts["x_range"]
        y_min, y_max = self.h_discrete_artifacts["y_range"]
        W, H = self.h_discrete_artifacts["W"], self.h_discrete_artifacts["H"]

        x_idx = int(round((x - x_min) / (x_max - x_min) * (W - 1)))
        y_idx = int(round((y - y_min) / (y_max - y_min) * (H - 1)))

        x_idx = max(0, min(W - 1, x_idx))
        y_idx = max(0, min(H - 1, y_idx))

        return x_idx, y_idx

    def h_discrete(self, pose: np.ndarray, return_gradient: bool = False):
        """Retrieves barrier value and computes local gradient in O(1)."""
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        else:
            pose = np.asarray(pose)

        if not self.h_discrete_artifacts:
            if return_gradient:
                return np.inf, 0.0, 0.0, 0.0
            return np.inf

        x, y, th = pose
        x_min, x_max = self.h_discrete_artifacts["x_range"]
        y_min, y_max = self.h_discrete_artifacts["y_range"]

        if x < x_min or x > x_max or y < y_min or y > y_max:
            if return_gradient:
                return np.inf, 0.0, 0.0, 0.0
            return np.inf

        W = self.h_discrete_artifacts["W"]
        H = self.h_discrete_artifacts["H"]
        dx = self.h_discrete_artifacts["dx"]
        dy = self.h_discrete_artifacts["dy"]
        distance_map = self.h_discrete_artifacts["distance_map"]

        x_idx, y_idx = self._nearest_index(pose)
        h_val = distance_map[y_idx, x_idx]

        if return_gradient:
            # Optimize: compute derivatives locally in O(1) instead of using np.gradient on the whole grid.
            # X derivative (dh_dx)
            if W <= 1:
                dh_dx = 0.0
            elif x_idx == 0:
                dh_dx = (distance_map[y_idx, 1] - distance_map[y_idx, 0]) / dx
            elif x_idx == W - 1:
                dh_dx = (distance_map[y_idx, W - 1] - distance_map[y_idx, W - 2]) / dx
            else:
                dh_dx = (distance_map[y_idx, x_idx + 1] - distance_map[y_idx, x_idx - 1]) / (2.0 * dx)

            # Y derivative (dh_dy)
            if H <= 1:
                dh_dy = 0.0
            elif y_idx == 0:
                dh_dy = (distance_map[1, x_idx] - distance_map[0, x_idx]) / dy
            elif y_idx == H - 1:
                dh_dy = (distance_map[H - 1, x_idx] - distance_map[H - 2, x_idx]) / dy
            else:
                dh_dy = (distance_map[y_idx + 1, x_idx] - distance_map[y_idx - 1, x_idx]) / (2.0 * dy)

            dh_dth = 0.0  # Barrier is independent of vehicle heading angle
            return h_val, dh_dx, dh_dy, dh_dth

        return h_val

    def get_command(self, state: np.ndarray, f=None, g=None) -> np.ndarray:
        """Main control loop mapping current state to a safety-filtered command."""
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        else:
            state = np.asarray(state)

        u_nom = self.nominal_control(state)

        if not self.h_discrete_artifacts:
            return u_nom

        h, h_x, h_y, h_th = self.h_discrete(state, return_gradient=True)
        if h == np.inf:
            return u_nom

        x, y, th = state
        v_nom, w_nom = u_nom

        # First derivative h_dot_nom
        h_dot_nom = h_x * v_nom * np.cos(th) + h_y * v_nom * np.sin(th)

        # Q(x) = directional derivative wrt angle
        Q = -h_x * np.sin(th) + h_y * np.cos(th)

        # P(x): curvature of distance field (zero for distance fields)
        P = 0.0

        # nominal second derivative ddot_h
        ddh_nom = P * v_nom**2 + Q * v_nom * w_nom

        # partial derivatives
        dddh_dv = 2 * P * v_nom + Q * w_nom
        dddh_dw = Q * v_nom

        C = (
            ddh_nom
            + self.k1 * h_dot_nom
            + self.k2 * (h_dot_nom + self.k1 * h)
            - dddh_dv * v_nom
            - dddh_dw * w_nom
        )

        A = np.array([dddh_dv, dddh_dw])

        # Execute high-performance ADMM/analytical solver
        u_safe = solve_qp_numpy(
            u_nom, self.R_diag, A, C, self.u_min, self.u_max, max_iters=20, rho=self.rho
        )
        return u_safe


if __name__ == "__main__":
    import os
    import sys
    from hydra import initialize, compose
    from omegaconf import OmegaConf

    # Set up sys.path dynamically to import other project packages
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=== Testing CBFSmokeController with Hydra Config ===")
    
    with initialize(version_base=None, config_path="../../configs"):
        # Load configs, overriding controller to select cbf_smoke and sensor to global
        cfg = compose(config_name="config", overrides=["+controller=cbf_smoke", "env/sensors@sensor=global"])
        
        cbf_config = cfg.controller
        print(f"Loaded config type: {type(cbf_config)}")
        print("Config Values:")
        print(OmegaConf.to_yaml(cbf_config))

        # Resolve config and extract schemas
        OmegaConf.resolve(cfg)
        env_params = OmegaConf.to_object(cfg.env)
        robot_params = OmegaConf.to_object(cfg.agent)
        sensor_params = OmegaConf.to_object(cfg.sensor)
        smoke_params = OmegaConf.to_object(cfg.simulator)

        # Force single agent parameters for test environment
        env_params.num_agents = 1
        env_params.max_steps = 2
        env_params.initial_locations = [[5.0, 15.0]]
        env_params.goal_locations = [[25.0, 15.0]]

        # Ensure global sensor size matches environment size
        sensor_params.world_x_size = env_params.world_x_size
        sensor_params.world_y_size = env_params.world_y_size

        goal = np.array(env_params.goal_locations[0])

        # Instantiate Controller
        controller = CBFSmokeController(
            config=cbf_config,
            env_params=env_params,
            robot_params=robot_params,
            goal=goal,
            num_agents=1
        )
        print("Successfully instantiated CBFSmokeController!")

        # 1. Test single agent assertion: instantiating with num_agents=2 should fail
        try:
            CBFSmokeController(
                config=cbf_config,
                env_params=env_params,
                robot_params=robot_params,
                goal=goal,
                num_agents=2
            )
            raise AssertionError("Assertion failed: num_agents=2 did not raise an error.")
        except AssertionError as e:
            if "only supports a single agent" in str(e):
                print("Assertion test passed: controller raises error for num_agents > 1.")
            else:
                raise e

        # 2. Run simulation loop for exactly 2 steps using SmokeEnv
        from src.env.smoke_env import SmokeEnv
        
        print("Initializing SmokeEnv...")
        env = SmokeEnv(
            env_params=env_params,
            robot_params=robot_params,
            sensor_params=sensor_params,
            simulator_params=smoke_params,
        )

        print("Resetting SmokeEnv...")
        obs, _ = env.reset(seed=42)
        print("SmokeEnv reset successfully.")

        print("Running controller simulation for exactly 2 steps...")
        for step in range(2):
            agent_obs = obs["agent_0"]
            location = np.asarray(agent_obs["location"])
            angle = float(np.ravel(agent_obs["angle"])[0])
            state = np.array([location[0], location[1], angle])

            # Extract smoke grid and positions
            smoke_values = agent_obs["smoke_density"].flatten()
            smoke_positions = agent_obs["smoke_density_location"]

            # Update controller barrier function h
            controller.update_h_discrete(smoke_values, smoke_positions, state)

            # Compute safe projected action
            cmd = controller.get_command(state)
            print(f"Step {step+1}: State={state}, Nominal={controller.nominal_control(state)}, Safe Action={cmd}")

            # Step environment
            obs, rewards, terminateds, truncateds, infos = env.step({"agent_0": cmd})

        env.close()
        print("Verification completed successfully!")

