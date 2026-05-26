import time
import numpy as np


class HOCBF2ndDegreeAnalyticalController:
    def __init__(self, u_min, u_max, d_safe=0.8, k1=3.0, k2=3.0):
        self.u_min = u_min
        self.u_max = u_max
        self.d_safe = d_safe
        self.k1 = k1
        self.k2 = k2
        self.R_diag = np.array([1.0, 1.0])

    def solve_qp_analytical(self, u_nom, A, C):
        """Analytical 2D QP solver in closed form (zero iterations).
        
        min 1/2 (u - u_nom)^T R (u - u_nom)
        s.t. A u + C >= 0, u_min <= u <= u_max
        """
        R_v, R_w = self.R_diag[0], self.R_diag[1]
        v_nom, w_nom = u_nom[0], u_nom[1]
        v_min, w_min = self.u_min[0], self.u_min[1]
        v_max, w_max = self.u_max[0], self.u_max[1]
        A_v, A_w = A[0], A[1]
        
        # 1. Fast path: check if nominal is safe
        if A_v * v_nom + A_w * w_nom + C >= 0:
            return np.array([np.clip(v_nom, v_min, v_max), np.clip(w_nom, w_min, w_max)])
            
        # 2. Constraint is active: A_v * v + A_w * w + C = 0
        eps = 1e-9
        
        # If steering has no authority on the barrier (A_w is 0)
        if np.abs(A_w) < eps:
            w_feas = np.clip(w_nom, w_min, w_max)
            if np.abs(A_v) < eps:
                # Degenerate: A = [0, 0] and C < 0 -> Infeasible
                return np.array([v_min, w_feas])
                
            # A_v * v >= -C
            v_bound = -C / A_v
            if A_v > 0:
                v_feas_min = max(v_min, v_bound)
                v_feas_max = v_max
            else:
                v_feas_min = v_min
                v_feas_max = min(v_max, v_bound)
                
            if v_feas_min > v_feas_max:
                return np.array([v_min, w_feas]) # Fallback
                
            v_opt = np.clip(v_nom, v_feas_min, v_feas_max)
            return np.array([v_opt, w_feas])
            
        # Standard case: A_w is non-zero
        # w = alpha * v + beta
        alpha = -A_v / A_w
        beta = -C / A_w
        
        # Unconstrained quadratic minimizer along the boundary line
        denom = R_v + R_w * alpha**2
        num = R_v * v_nom - R_w * alpha * (beta - w_nom)
        v_unconstrained = num / denom
        
        # Intersect bounds: v in [v_min, v_max] and w(v) in [w_min, w_max]
        if np.abs(alpha) < eps:
            v_w_min = v_min
            v_w_max = v_max
        else:
            val1 = (w_min - beta) / alpha
            val2 = (w_max - beta) / alpha
            v_w_min = min(val1, val2)
            v_w_max = max(val1, val2)
        
        v_feas_min = max(v_min, v_w_min)
        v_feas_max = min(v_max, v_w_max)
        
        if v_feas_min > v_feas_max:
            # Physically infeasible (no intersection between safe set and motor box)
            # Fallback: full braking (v_min) and clip steering
            return np.array([v_min, np.clip(w_nom, w_min, w_max)])
            
        # Clip to feasible range
        v_opt = np.clip(v_unconstrained, v_feas_min, v_feas_max)
        w_opt = alpha * v_opt + beta
        
        return np.array([v_opt, w_opt])

    def get_control(self, state, goal, obstacles):
        x, y, th = state
        v_nom = self.u_max[0]
        
        desired_angle = np.arctan2(goal[1] - y, goal[0] - x)
        e_angle = desired_angle - th
        e_angle = (e_angle + np.pi) % (2 * np.pi) - np.pi
        w_nom = np.clip(5.0 * e_angle, self.u_min[1], self.u_max[1])
        u_nom = np.array([v_nom, w_nom])
        
        if len(obstacles) == 0:
            return u_nom

        p_c = np.array([x, y])
        
        # Find the most critical obstacle (closest to robot center)
        min_h = np.inf
        crit_obs = None
        
        for obs in obstacles:
            p_obs = np.array(obs['pos'])
            d = np.linalg.norm(p_c - p_obs)
            d_barrier = obs['r'] + self.d_safe
            h = d - d_barrier
            if h < min_h:
                min_h = h
                crit_obs = obs
                
        p_obs = np.array(crit_obs['pos'])
        d = np.linalg.norm(p_c - p_obs)
        d_barrier = crit_obs['r'] + self.d_safe
        h = d - d_barrier
        
        if h > 1.5:
            return u_nom
            
        h_x = (x - p_obs[0]) / (d + 1e-6)
        h_y = (y - p_obs[1]) / (d + 1e-6)
        
        h_dot_nom = h_x * v_nom * np.cos(th) + h_y * v_nom * np.sin(th)
        Q = -h_x * np.sin(th) + h_y * np.cos(th)
        P = 0.0
        ddh_nom = P * v_nom**2 + Q * v_nom * w_nom
        
        dddh_dv = Q * w_nom
        dddh_dw = Q * v_nom
        
        A = np.array([dddh_dv, dddh_dw])
        C = (
            ddh_nom
            + self.k1 * h_dot_nom
            + self.k2 * (h_dot_nom + self.k1 * h)
            - dddh_dv * v_nom
            - dddh_dw * w_nom
        )
        
        u_safe = self.solve_qp_analytical(u_nom, A, C)
        return u_safe


def simulate_dubins(state, u, dt):
    x, y, th = state
    v, w = u
    def derivs(s):
        theta = s[2]
        return np.array([v * np.cos(theta), v * np.sin(theta), w])
        
    k1 = derivs(state)
    k2 = derivs(state + 0.5 * dt * k1)
    k3 = derivs(state + 0.5 * dt * k2)
    k4 = derivs(state + dt * k3)
    
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
    return next_state
