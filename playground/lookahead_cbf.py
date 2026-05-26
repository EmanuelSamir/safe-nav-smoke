import time
import numpy as np


class LookaheadCBFController:
    def __init__(self, u_min, u_max, L=0.4, d_safe=0.8, k1=3.0):
        self.u_min = u_min
        self.u_max = u_max
        self.L = L
        self.d_safe = d_safe
        self.k1 = k1

    def get_control(self, state, goal, obstacles):
        """obstacles: list of dicts [{'pos': [x, y], 'r': radius}]"""
        x, y, th = state
        v_nom = self.u_max[0]
        
        # Nominal goal-seeking control
        desired_angle = np.arctan2(goal[1] - y, goal[0] - x)
        e_angle = desired_angle - th
        e_angle = (e_angle + np.pi) % (2 * np.pi) - np.pi
        w_nom = np.clip(5.0 * e_angle, self.u_min[1], self.u_max[1])
        u_nom = np.array([v_nom, w_nom])
        
        if len(obstacles) == 0:
            return u_nom

        cos_t, sin_t = np.cos(th), np.sin(th)
        p_i = np.array([x + self.L * cos_t, y + self.L * sin_t])
        
        # Find the most critical obstacle (closest to Lookahead point)
        min_h0 = np.inf
        crit_obs = None
        
        for obs in obstacles:
            p_obs = np.array(obs['pos'])
            p_rel = p_i - p_obs
            dist = np.linalg.norm(p_rel)
            d_barrier = obs['r'] + self.d_safe + self.L
            h0 = dist**2 - d_barrier**2
            if h0 < min_h0:
                min_h0 = h0
                crit_obs = obs
        
        # Apply 1st-degree Lookahead CBF to the critical obstacle
        p_obs = np.array(crit_obs['pos'])
        p_rel = p_i - p_obs
        dist = np.linalg.norm(p_rel)
        d_barrier = crit_obs['r'] + self.d_safe + self.L
        h0 = dist**2 - d_barrier**2
        
        # If safe under nominal control
        R = np.array([
            [cos_t, -self.L * sin_t],
            [sin_t, self.L * cos_t]
        ])
        dot_p_nom = R @ u_nom
        dot_h0_nom = 2 * p_rel @ dot_p_nom
        if dot_h0_nom >= -self.k1 * h0:
            return u_nom
            
        A_v = 2.0 * (p_rel[0] * cos_t + p_rel[1] * sin_t)
        A_w = 2.0 * (-self.L * p_rel[0] * sin_t + self.L * p_rel[1] * cos_t)
        A = np.array([A_v, A_w])
        B = -self.k1 * h0
        
        A_norm_sq = max(np.sum(A**2), 1e-6)
        violation = B - np.dot(A, u_nom)
        
        u_safe = u_nom + (max(0.0, violation) / A_norm_sq) * A
        u_safe_clamped = np.clip(u_safe, self.u_min, self.u_max)
        
        return u_safe_clamped


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
