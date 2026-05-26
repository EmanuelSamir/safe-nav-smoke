import time
import numpy as np


class HOCBF2ndDegreeADMMController:
    def __init__(self, u_min, u_max, d_safe=0.8, k1=3.0, k2=3.0):
        self.u_min = u_min
        self.u_max = u_max
        self.d_safe = d_safe
        self.k1 = k1
        self.k2 = k2
        self.R_diag = np.array([1.0, 1.0])
        self.last_iterations = 0 # Track iterations run

    def solve_qp_admm(self, u_nom, A, C, max_iters=30, rho=10.0, tol=1e-4):
        """ADMM 2D QP solver with early stopping tolerance check."""
        u = u_nom.copy()
        z = max(0.0, np.dot(A, u) + C)
        y = 0.0
        
        R_mat = np.diag(self.R_diag)
        M = R_mat + rho * np.outer(A, A)
        
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        det = a * d - b * c
        inv_det = 1.0 / (det + 1e-8)
        
        self.last_iterations = max_iters
        
        for k in range(max_iters):
            u_prev = u.copy()
            z_prev = z
            
            temp = rho * (z - C - y)
            rhs = self.R_diag * u_nom + A * temp
            
            u_1 = inv_det * (d * rhs[0] - b * rhs[1])
            u_2 = inv_det * (-c * rhs[0] + a * rhs[1])
            u = np.array([u_1, u_2])
            
            u = np.clip(u, self.u_min, self.u_max)
            A_u = np.dot(A, u)
            z = max(0.0, A_u + C + y)
            y = y + A_u + C - z
            
            # Tolerance/Early stopping check
            primal_residual = np.abs(u - u_prev)
            slack_residual = np.abs(z - z_prev)
            if k > 0 and np.max(primal_residual) < tol and slack_residual < tol:
                self.last_iterations = k + 1
                break
                
        return u

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
            self.last_iterations = 0
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
        
        u_safe = self.solve_qp_admm(u_nom, A, C)
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
