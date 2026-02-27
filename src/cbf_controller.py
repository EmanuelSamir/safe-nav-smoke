
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
        self.rho = 1.0
        self.u_min = np.array(self.robot_params.action_min)
        self.u_max = np.array(self.robot_params.action_max)
        self.n_u = 2
        self.n_x = 3
        self.k = 0.5
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

        if False:
            bearing_cost, dc_dx, dc_dy, dc_dth = self.compute_additional_bearing(h, smoke_positions, robot_pose)

            # add bearing contribution into h
            lambda_b = 3.0 # <-- tuning parameter
            h = h - lambda_b * bearing_cost

            dh_dx = dh_dx - lambda_b * dc_dx
            dh_dy = dh_dy - lambda_b * dc_dy
            dh_dth = dh_dth - lambda_b * dc_dth
            print("dh_dx", dh_dx, dc_dx)
            print("dh_dy", dh_dy, dc_dy)
            print("dh_dth", dh_dth, dc_dth)

        self.h_discrete_artifacts["x_range"] = [x_min, x_max]
        self.h_discrete_artifacts["y_range"] = [y_min, y_max]
        self.h_discrete_artifacts["W"] = W
        self.h_discrete_artifacts["H"] = H  
        self.h_discrete_artifacts["grid_points"] = smoke_positions
        self.h_discrete_artifacts["h"] = h
        self.h_discrete_artifacts["dh_dx"] = dh_dx
        self.h_discrete_artifacts["dh_dy"] = dh_dy
        self.h_discrete_artifacts["dh_dth"] = dh_dth

    def compute_additional_bearing(self, h: np.ndarray, smoke_positions: np.ndarray, robot_pose: np.ndarray): 
        # --- Robot state ----
        px, py, th = robot_pose
        p = np.array([px, py])
        
        heading = np.array([np.cos(th), np.sin(th)])

        # vector robot -> all obstacle points
        d = smoke_positions - p   # shape: (N, 2)
        dist = np.linalg.norm(d, axis=1)  # shape: (N,)

        # avoid division by zero
        dist = np.maximum(dist, 1e-6)

        # normalized directions to obstacles
        n = d / dist[:, None]     # shape: (N, 2)

        # dot product to check FOV: cos(angle)
        cos_angles = n @ heading  # shape: (N,)

        # Keep obstacles within ±90° (FOV = 180°)
        # i.e., cos(angle) >= 0
        mask_fov = cos_angles >= 0.0

        # If none in FOV, fallback to closest globally (optional)
        if not np.any(mask_fov):
            idx = np.argmin(h)
            p_obs = smoke_positions[idx]

        # consider only within-FOV obstacles
        h_fov = h.copy()
        h_fov[~mask_fov] = np.inf

        idx = np.argmin(h_fov)
        p_obs = smoke_positions[idx]

        d = p_obs - p
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            dist = 1e-6

        # unit direction from robot -> obstacle
        n = d / dist
        nx, ny = n

        # ---- bearing cost ----
        # c = nx*cos(th) + ny*sin(th)
        print("Cost: ",nx, ny, np.cos(th), np.sin(th))
        bearing_cost = nx*np.cos(th) + ny*np.sin(th)
        print("Cost: ",bearing_cost)

        # gradient wrt theta
        dc_dtheta = -nx*np.sin(th) + ny*np.cos(th)

        # gradient wrt x,y
        # dc/dp = -(I - n n^T) * heading / dist
        heading = np.array([np.cos(th), np.sin(th)])
        P = np.eye(2) - np.outer(n, n)
        dc_dp = -(P @ heading) / dist
        dc_dx, dc_dy = dc_dp

        return bearing_cost, dc_dx, dc_dy, dc_dtheta


    # def update_h_discrete(self, occupancy_grid: np.ndarray, bounds: np.ndarray, robot_pose: np.ndarray):
    #     """
    #     occupancy_grid: [H, W]. 1 is free, 0 is occupied
    #     robot_pose: [2]
    #     bounds: [2, 2]
    #     """
    #     assert occupancy_grid.ndim == 2, "Occupancy grid must be a 2D array"

    #     H, W = occupancy_grid.shape

    #     x_min, y_min = bounds[0]
    #     x_max, y_max = bounds[-1]

    #     x = np.linspace(x_min, x_max, W)
    #     y = np.linspace(y_min, y_max, H)

    #     xx, yy = np.meshgrid(x, y)

    #     grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    #     # Convert occupancy grid to distance map
    #     # grid_map assumed: 1 = free, 0 = obstacle
    #     distance_map = skfmm.distance(occupancy_grid - 0.5, dx=0.5)

    #     h = distance_map.ravel()

    #     gradient_maps = np.gradient(distance_map)

    #     dh_dx = gradient_maps[1].ravel()
    #     dh_dy = gradient_maps[0].ravel()

    #     self.h_discrete_artifacts["x_range"] = [x_min, x_max]
    #     self.h_discrete_artifacts["y_range"] = [y_min, y_max]
    #     self.h_discrete_artifacts["grid_points"] = grid_points
    #     self.h_discrete_artifacts["h"] = h
    #     self.h_discrete_artifacts["dh_dx"] = dh_dx
    #     self.h_discrete_artifacts["dh_dy"] = dh_dy

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

    # def get_command(self, state: np.ndarray, f: np.ndarray, g: np.ndarray):
    #     u_nom = self.nominal_control(state)
    #     if not self.h_discrete_artifacts:
    #         return u_nom

    #     h, dh_dx, dh_dy, dh_dth = self.h_discrete(state, return_gradient=True)
    #     if h == np.inf:
    #         print("CBFController: State out of bounds")
    #         return u_nom

    #     assert g.shape == (self.n_x, self.n_u), f"g must be of shape {self.n_x}x{self.n_u}"
    #     assert f.shape == (self.n_x,), f"f must be of shape {self.n_x}"

    #     h_dot = np.array([dh_dx, dh_dy, dh_dth]).reshape(1, self.n_x)

    #     Lf_h = h_dot @ f
    #     Lg_h = h_dot @ g

    #     u = cp.Variable(self.n_u)
    #     xi = cp.Variable(nonneg=True)
    #     cost = 0.5 * cp.quad_form(u - u_nom, self.R) + self.rho * cp.square(xi)
    #     constraints = [
    #         Lf_h + Lg_h @ u >= -self.alpha(h) - xi,
    #         u >= self.u_min,
    #         u <= self.u_max
    #     ]
    #     prob = cp.Problem(cp.Minimize(cost), constraints)
    #     prob.solve(solver=cp.OSQP)

    #     u_value = u.value
    #     xi_value = xi.value
    #     status = prob.status

    #     return u_value


    # def get_command(self, state: np.ndarray, f: np.ndarray, g: np.ndarray):
    #     u_nom = self.nominal_control(state)
    #     if not self.h_discrete_artifacts:
    #         return u_nom

    #     h, h_x, h_y, h_th = self.h_discrete(state, return_gradient=True)
    #     if h == np.inf:
    #         return u_nom

    #     k1 = 1.5
    #     k2 = 2.5

    #     x, y, th = state
    #     v_nom, w_nom = u_nom

    #     # ------------------------------
    #     # First derivative h_dot
    #     # ------------------------------
    #     h_dot = h_x * v_nom * np.cos(th) + h_y * v_nom * np.sin(th)

    #     # ------------------------------
    #     # Second derivative terms
    #     # ------------------------------

    #     # Q(x) = directional derivative wrt orientation
    #     Q = -h_x * np.sin(th) + h_y * np.cos(th)

    #     # P(x) is mostly curvature of h-field (unknown); approximate with 0
    #     P = 0.0   

    #     A = Q * v_nom   # coefficient multiplying w
    #     B = (
    #         P * v_nom**2
    #         + k1 * h_dot
    #         + k2 * (h_dot + k1 * h)
    #     )

    #     # ------------------------------
    #     # Build the QP
    #     # ------------------------------
    #     u = cp.Variable(self.n_u)
    #     xi = cp.Variable(nonneg=True)

    #     # HOCBF constraint:
    #     #   A * w + B >= -xi
    #     cbf_constraint = A * u[1] + B >= -xi

    #     cost = 0.5 * cp.quad_form(u - u_nom, self.R) + self.rho * cp.square(xi)
    #     constraints = [
    #         cbf_constraint,
    #         u >= self.u_min,
    #         u <= self.u_max,
    #     ]

    #     prob = cp.Problem(cp.Minimize(cost), constraints)
    #     prob.solve(solver=cp.OSQP)

    #     if prob.status not in ["optimal", "optimal_inaccurate"] or u.value is None:
    #         return u_nom

    #     return u.value

        
    def get_command(self, state, f, g):
        u_nom = self.nominal_control(state)
        v_nom, w_nom = u_nom

        if not self.h_discrete_artifacts:
            return u_nom

        h, h_x, h_y, h_th = self.h_discrete(state, return_gradient=True)
        if h == np.inf:
            return u_nom

        x, y, th = state

        # ==========================
        # First derivative h_dot_nom
        # ==========================
        h_dot_nom = h_x * v_nom * np.cos(th) + h_y * v_nom * np.sin(th)

        # ==========================
        # Q(x) = directional derivative wrt angle
        # ==========================
        Q = -h_x * np.sin(th) + h_y * np.cos(th)

        # ==========================
        # P(x): curvature of distance field
        # approximated as 0 (good enough)
        # ==========================
        P = 0.0

        # ==========================
        # nominal second derivative ddot_h
        # ==========================
        ddh_nom = P * v_nom**2 + Q * v_nom * w_nom

        # ==========================
        # partial derivatives
        # ==========================
        dddh_dv = 2 * P * v_nom + Q * w_nom
        dddh_dw = Q * v_nom


        k1 = 1.5
        k2 = 2.5

        # ==========================
        # constant term C
        # ==========================
        C = (
            ddh_nom
            + k1 * h_dot_nom
            + k2 * (h_dot_nom + k1 * h)
            - dddh_dv * v_nom
            - dddh_dw * w_nom
        )

        # ==========================
        # QP VARIABLES
        # ==========================
        u = cp.Variable(2)
        xi = cp.Variable(nonneg=True)

        # constraint A_v*v + A_w*w + C >= -xi
        cbf_constraint = dddh_dv * u[0] + dddh_dw * u[1] + C >= -xi

        # ==========================
        # COST
        # ==========================
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


        

