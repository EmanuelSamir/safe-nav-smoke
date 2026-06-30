"""Dynamics definitions for relative states and kinematics."""

import jax.numpy as jnp
from hj_reachability import dynamics, sets


class RelativeDubinsDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """JAX dynamics for the relative state between two identical Dubins cars.

    States:
        xr, yr, thr
    Control (Ego):
        u = [v_1, w_1] -> [v, omega]
    Disturbance (Opponent):
        d = [v_2, w_2] -> [v, omega]

    Relative kinematics:
        d xr/dt   = -v_1 + v_2 * cos(thr) + w_1 * yr
        d yr/dt   = v_2 * sin(thr) - w_1 * xr
        d thr/dt  = w_2 - w_1
    """

    def __init__(self, action_min, action_max, control_mode="max", disturbance_mode="min"):
        if hasattr(action_min, "detach"):
            action_min = action_min.detach().cpu().numpy()
        if hasattr(action_max, "detach"):
            action_max = action_max.detach().cpu().numpy()
        control_space = sets.Box(jnp.array(action_min), jnp.array(action_max))
        disturbance_space = sets.Box(jnp.array(action_min), jnp.array(action_max))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.zeros_like(state)

    def control_jacobian(self, state, time):
        xr, yr, thr = state[0], state[1], state[2]
        return jnp.array([[-1.0, yr], [0.0, -xr], [0.0, -1.0]])

    def disturbance_jacobian(self, state, time):
        xr, yr, thr = state[0], state[1], state[2]
        return jnp.array([[jnp.cos(thr), 0.0], [jnp.sin(thr), 0.0], [0.0, 1.0]])


class AbsoluteDubinsDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """HJ Reachability Affine Dynamics for a Dubins car in absolute frame.

    States:
        x, y, theta
    Control:
        u = [v, omega]

    Absolute kinematics:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = omega
    """

    def __init__(
        self, action_min, action_max, control_mode: str = "max", disturbance_mode: str = "min"
    ) -> None:
        if hasattr(action_min, "detach"):
            action_min = action_min.detach().cpu().numpy()
        if hasattr(action_max, "detach"):
            action_max = action_max.detach().cpu().numpy()

        control_space = sets.Box(jnp.array(action_min), jnp.array(action_max))

        # Configure disturbance space (defaults to zero bounds)
        dist_min = jnp.array(action_min) * 0.0
        dist_max = jnp.array(action_max) * 0.0
        disturbance_space = sets.Box(dist_min, dist_max)

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state: jnp.ndarray, time_sec: float) -> jnp.ndarray:
        return jnp.zeros_like(state)

    def control_jacobian(self, state: jnp.ndarray, time_sec: float) -> jnp.ndarray:
        return jnp.array([[jnp.cos(state[2]), 0.0], [jnp.sin(state[2]), 0.0], [0.0, 1.0]])

    def disturbance_jacobian(self, state: jnp.ndarray, time_sec: float) -> jnp.ndarray:
        return jnp.zeros((state.shape[0], self.disturbance_space.ndim))
