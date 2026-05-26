import time
import numpy as np


class HOCBF2ndDegreeController:
    def __init__(self, u_min, u_max, d_safe=1.0, k1=5.0, k2=5.0):
        self.u_min = u_min
        self.u_max = u_max
        self.d_safe = d_safe
        self.k1 = k1
        self.k2 = k2
        self.R_diag = np.array([1.0, 1.0])

    def solve_qp_admm(self, u_nom, A, C, max_iters=20, rho=10.0):
        """NumPy implementation of the vectorized ADMM 2D QP solver."""
        u = u_nom.copy()
        # Constraint: A u + C >= 0 -> slack z = max(0, A u + C)
        z = max(0.0, np.dot(A, u) + C)
        y = 0.0 # dual variable
        
        R_mat = np.diag(self.R_diag)
        M = R_mat + rho * np.outer(A, A)
        
        # Cramer's rule for 2D inversion M * u = rhs
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        det = a * d - b * c
        inv_det = 1.0 / (det + 1e-8)
        
        for _ in range(max_iters):
            temp = rho * (z - C - y)
            rhs = self.R_diag * u_nom + A * temp
            
            u_1 = inv_det * (d * rhs[0] - b * rhs[1])
            u_2 = inv_det * (-c * rhs[0] + a * rhs[1])
            u = np.array([u_1, u_2])
            
            u = np.clip(u, self.u_min, self.u_max)
            A_u = np.dot(A, u)
            z = max(0.0, A_u + C + y)
            y = y + A_u + C - z
            
        return u

    def get_control(self, state, goal, obs_pos, obs_r):
        x, y, th = state
        v_nom = self.u_max[0]
        
        # Nominal control (seek goal)
        desired_angle = np.arctan2(goal[1] - y, goal[0] - x)
        e_angle = desired_angle - th
        e_angle = (e_angle + np.pi) % (2 * np.pi) - np.pi
        w_nom = np.clip(5.0 * e_angle, self.u_min[1], self.u_max[1])
        u_nom = np.array([v_nom, w_nom])
        
        # Distance directly to robot center
        p_c = np.array([x, y])
        p_obs = np.array(obs_pos)
        d = np.linalg.norm(p_c - p_obs)
        d_barrier = obs_r + self.d_safe
        h = d - d_barrier
        
        # Check if nominal control is already very safe
        if h > 1.5:
            return u_nom
            
        # Spatial gradients
        h_x = (x - obs_pos[0]) / (d + 1e-6)
        h_y = (y - obs_pos[1]) / (d + 1e-6)
        
        # 1st time derivative
        h_dot_nom = h_x * v_nom * np.cos(th) + h_y * v_nom * np.sin(th)
        
        # Directional derivative wrt heading angle
        Q = -h_x * np.sin(th) + h_y * np.cos(th)
        P = 0.0 # curvature assumed zero as in cbf_ctrl.py
        
        # 2nd time derivative nominal
        ddh_nom = P * v_nom**2 + Q * v_nom * w_nom
        
        # Partial derivatives for linearization A * u >= -C
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
        
        # Solve HOCBF safety QP via ADMM solver
        u_safe = self.solve_qp_admm(u_nom, A, C, max_iters=20)
        return u_safe


def simulate_dubins(state, u, dt):
    x, y, th = state
    v, w = u
    # RK4 integration
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


if __name__ == "__main__":
    print("🚗 Simulando Navegación con HOCBF de 2º Grado (Enfoque B)...")
    
    state = np.array([0.0, 0.0, 0.0]) # Inicio
    goal = np.array([10.0, 10.0])     # Meta
    obs_pos = np.array([5.0, 5.0])    # Obstáculo en diagonal
    obs_r = 1.5
    
    u_min = np.array([0.0, -4.0])
    u_max = np.array([3.0, 4.0])
    
    controller = HOCBF2ndDegreeController(u_min, u_max)
    dt = 0.1
    steps = 150
    
    trajectory = []
    times = []
    collisions = 0
    
    for step in range(steps):
        trajectory.append(state.copy())
        
        # Medir tiempo de resolución del CBF
        t0 = time.perf_counter()
        u = controller.get_control(state, goal, obs_pos, obs_r)
        times.append(time.perf_counter() - t0)
        
        # Verificar colisiones con centro del robot
        dist_to_obs = np.linalg.norm(state[:2] - obs_pos)
        if dist_to_obs < obs_r:
            collisions += 1
            
        state = simulate_dubins(state, u, dt)
        
        # Terminar si llega a la meta
        if np.linalg.norm(state[:2] - goal) < 0.5:
            print(f"🎯 ¡Meta alcanzada en el paso {step}!")
            break
            
    trajectory = np.array(trajectory)
    times_us = np.array(times) * 1e6 # Convertir a microsegundos
    
    print("\n📊 RESULTADOS ENFOQUE B (HOCBF 2º Grado + ADMM):")
    print(f"   - Tiempo total acumulado del solver: {np.sum(times) * 1000.0:.3f} ms")
    print(f"   - Tiempo promedio por iteración:     {np.mean(times_us):.3f} μs")
    print(f"   - Máximo tiempo en una iteración:    {np.max(times_us):.3f} μs")
    print(f"   - Pasos con colisión detectada:      {collisions} (centro del robot)")
    print(f"   - Posición Final:                    {state[:2]}")
