
import numpy as np
import cvxpy as cp
import skfmm
from dataclasses import dataclass
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv
from agents.basic_robot import RobotParams

class CBFController:
    def __init__(self, env_params: EnvParams, robot_params: RobotParams, goal: np.ndarray, smoke_threshold: float = 0.75):
        self.env_params = env_params
        self.robot_params = robot_params
        self.goal = goal

        self.h_discrete_artifacts = {}
        self.smoke_threshold = smoke_threshold

        # Leaving literals. Consider make it parameters
        self.R = np.diag([1.0, 1.0])
        self.rho = 5.0
        self.u_min = np.array(self.robot_params.action_min)
        self.u_max = np.array(self.robot_params.action_max)
        self.n_u = 2
        self.n_x = 3
        self.k1 = 5.0
        self.k2 = 5.0
        self.alpha = lambda h: self.k * h

        assert self.R.shape == (self.n_u, self.n_u), f"R must be of shape {self.n_u}x{self.n_u}"
        
    def nominal_control(self, state: np.ndarray):
        """
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

    def update_h_discrete(self, smoke_values: np.ndarray, smoke_positions: np.ndarray, robot_pose: np.ndarray):
        assert smoke_values.ndim == 1, "Smoke values must be a 1D array"
        assert smoke_positions.ndim == 2, "Smoke positions must be a 2D array"
        assert smoke_values.shape[0] == smoke_positions.shape[0], "Smoke values and positions must have the same length"

        x_min = np.min(smoke_positions[:, 0])
        x_max = np.max(smoke_positions[:, 0])
        y_min = np.min(smoke_positions[:, 1])
        y_max = np.max(smoke_positions[:, 1])
        N = smoke_values.shape[0]

        ratio = (x_max - x_min) / (y_max - y_min)
        # H = np.rint(np.sqrt(N / ratio)).astype(int)
        # W = np.rint(np.sqrt(N * ratio)).astype(int)
        Hs = [int(np.floor(np.sqrt(N/ratio))), int(np.ceil(np.sqrt(N/ratio)))]
        Ws = [int(np.floor(np.sqrt(N*ratio))), int(np.ceil(np.sqrt(N*ratio)))]
        H,W = next((h,w) for h in Hs for w in Ws if h>0 and w>0 and h*w==N)

        smoke_map = smoke_values.reshape(H, W)
        # 1 is free, 0 is occupied
        occupancy_grid = (smoke_map < self.smoke_threshold).astype(int)

        if np.all(occupancy_grid == 1):
            self.h_discrete_artifacts = {}
            return

        # Compute physical resolution (meters per cell)
        dx_grid = (x_max - x_min) / max(1, W - 1)
        dy_grid = (y_max - y_min) / max(1, H - 1)

        # We assume square-ish cells. Skfmm takes a single dx or a tuple. We use the tuple (dy, dx).
        # Margin is in meters.
        margin = 1.0
        distance_map = skfmm.distance(occupancy_grid - 0.5, dx=(dy_grid, dx_grid)) - margin
        h = distance_map.ravel()

        # np.gradient assumes spacing of 1 index unless specified. Provide actual spacing.
        grad_y, grad_x = np.gradient(distance_map, dy_grid, dx_grid)
        dh_dx = grad_x.ravel()
        dh_dy = grad_y.ravel()
        dh_dth = np.zeros_like(h)

        self.h_discrete_artifacts["x_range"] = [x_min, x_max]
        self.h_discrete_artifacts["y_range"] = [y_min, y_max]
        self.h_discrete_artifacts["W"] = W
        self.h_discrete_artifacts["H"] = H  
        self.h_discrete_artifacts["grid_points"] = smoke_positions
        self.h_discrete_artifacts["h"] = h
        self.h_discrete_artifacts["dh_dx"] = dh_dx
        self.h_discrete_artifacts["dh_dy"] = dh_dy
        self.h_discrete_artifacts["dh_dth"] = dh_dth

    def _nearest_index(self, pose):
        x, y, th = pose
        x_min, x_max = self.h_discrete_artifacts["x_range"]
        y_min, y_max = self.h_discrete_artifacts["y_range"]
        W, H = self.h_discrete_artifacts["W"], self.h_discrete_artifacts["H"]

        # Handle exact bounds safely
        x_idx = int(round((x - x_min) / (x_max - x_min) * (W - 1)))
        y_idx = int(round((y - y_min) / (y_max - y_min) * (H - 1)))

        x_idx = max(0, min(W - 1, x_idx))
        y_idx = max(0, min(H - 1, y_idx))

        # We assume the flat arrays are reshaped as (H, W) or raveled in row-major
        return y_idx * W + x_idx

    def h_discrete(self, pose: np.ndarray, return_gradient=False):
        if not self.h_discrete_artifacts:
            raise ValueError("H discrete artifacts not computed. Call update_h_discrete first.")
        x, y, th = pose

        x_min, x_max = self.h_discrete_artifacts["x_range"]
        y_min, y_max = self.h_discrete_artifacts["y_range"]

        if x < x_min or x > x_max or y < y_min or y > y_max:
            return np.inf, 0.0, 0.0, 0.0

        idx = self._nearest_index(pose)
        if return_gradient:
            return self.h_discrete_artifacts["h"][idx], self.h_discrete_artifacts["dh_dx"][idx], self.h_discrete_artifacts["dh_dy"][idx], self.h_discrete_artifacts["dh_dth"][idx]
        return self.h_discrete_artifacts["h"][idx]
        
    def get_command(self, state, f, g):
        u_nom = self.nominal_control(state)
        v_nom, w_nom = u_nom

        if not self.h_discrete_artifacts:
            return u_nom

        h, h_x, h_y, h_th = self.h_discrete(state, return_gradient=True)
        if h == np.inf:
            return u_nom

        x, y, th = state

        # First derivative h_dot_nom
        h_dot_nom = h_x * v_nom * np.cos(th) + h_y * v_nom * np.sin(th)

        # Q(x) = directional derivative wrt angle
        Q = -h_x * np.sin(th) + h_y * np.cos(th)

        # P(x): curvature of distance field
        # approximated as 0 (good enough)
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

        u = cp.Variable(2)
        xi = cp.Variable(nonneg=True)

        # constraint A_v*v + A_w*w + C >= -xi
        cbf_constraint = dddh_dv * u[0] + dddh_dw * u[1] + C >= -xi

        cost = 0.5 * cp.quad_form(u - u_nom, self.R) + self.rho * cp.square(xi)
        constraints = [
            cbf_constraint,
            u >= self.u_min,
            u <= self.u_max,
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)

        if prob.status not in ["optimal", "optimal_inaccurate"] or u.value is None:
            return u_nom

        return u.value


        

