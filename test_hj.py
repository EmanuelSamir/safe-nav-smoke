import jax.numpy as jnp
from hj_reachability import dynamics, sets

action_min = jnp.array([0.0, -4.0])
action_max = jnp.array([6.0, 4.0])

class RelativeDubinsDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self):
        control_space = sets.Box(action_min, action_max)
        disturbance_space = sets.Box(action_min, action_max)
        super().__init__(control_space=control_space, disturbance_space=disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.zeros_like(state)

    def control_jacobian(self, state, time):
        xr, yr, thr = state[0], state[1], state[2]
        return jnp.array([[-1.0, yr], [0.0, -xr], [0.0, -1.0]])

    def disturbance_jacobian(self, state, time):
        xr, yr, thr = state[0], state[1], state[2]
        return jnp.array([[jnp.cos(thr), 0.0], [jnp.sin(thr), 0.0], [0.0, 1.0]])

dyn = RelativeDubinsDynamics()
state = jnp.array([1.0, 2.0, 0.5])
time = 0.0

cj = dyn.control_jacobian(state, time)
mm = dyn.control_space.max_magnitudes
print("cj shape:", cj.shape)
print("mm shape:", mm.shape)
try:
    term2 = jnp.sum(jnp.abs(cj) * mm, axis=-1)
    print("term2 shape:", term2.shape)
except Exception as e:
    print("term2 failed:", e)

