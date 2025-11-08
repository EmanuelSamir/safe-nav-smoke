import numpy as np
import cvxpy as cp


import cvxpy as cp
import numpy as np

class CBFHJ:
    def __init__(self, R: np.ndarray, rho: float, u_min: np.ndarray, u_max: np.ndarray, alpha_type: str, b_margin: float, x_dim: int, u_dim: int, k: float = 0.75, S: float = 20.0):
        self.R = R
        self.rho = rho
        self.u_min = u_min
        self.u_max = u_max
        self.b_margin = b_margin
        self.n_u = u_dim
        self.n_x = x_dim
        self.k = k
        self.S = S
        if alpha_type == "linear":
            self.alpha = lambda h: self.k * h
        elif alpha_type == "quadratic":
            self.alpha = lambda h: self.k * h**2
        else:
            raise ValueError(f"Invalid alpha type: {alpha_type}")

        assert self.u_min.shape == self.u_max.shape == (u_dim,), f"u_min and u_max must have the same shape"
        assert self.R.shape == (self.n_u, self.n_u), f"R must be of shape {self.n_u}x{self.n_u}"

    def compute_control(self, u_nom: np.ndarray, V: float, V_grad: np.ndarray, f, g):
        assert u_nom.shape == (self.n_u,), f"u_nom must be of shape {self.n_u}"
        assert V_grad.shape == f.shape, f"V_grad must be of shape {f.shape}"
        assert g.shape == (self.n_x, self.n_u), f"g must be of shape {self.n_x}x{self.n_u}"
        assert f.shape == (self.n_x,), f"f must be of shape {self.n_x}"

        V = V * self.S
        V_grad = V_grad * self.S

        h = V - self.b_margin
        Lf_h = V_grad.reshape(1, self.n_x) @ f
        Lg_h = V_grad.reshape(1, self.n_x) @ g

        u = cp.Variable(self.n_u)
        xi = cp.Variable(nonneg=True)
        cost = 0.5 * cp.quad_form(u - u_nom, self.R) + self.rho * cp.square(xi)
        constraints = [
            Lf_h + Lg_h @ u >= -self.alpha(h) - xi,
            u >= self.u_min,
            u <= self.u_max
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)
        return u.value, xi.value, prob.status

if __name__ == "__main__":
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    rho = 1.0
    u_min = np.array([-1.0, -1.0])
    u_max = np.array([1.0, 1.0])
    alpha_type = "linear"
    b_margin = 0.0
    x_dim = 2
    u_dim = 2
    cbf_hj = CBFHJ(R, rho, u_min, u_max, alpha_type, b_margin, x_dim, u_dim, k=0.75)
    u_nom = np.array([0.0, 0.0])
    V = 0.0
    V_grad = np.array([1.0, 1.0])
    f = np.array([1.0, 1.0])
    g = np.array([[1.0, 1.0], [1.0, 1.0]])
    u, xi, status = cbf_hj.compute_control(u_nom, V, V_grad, f, g)
    print(u, xi, status)