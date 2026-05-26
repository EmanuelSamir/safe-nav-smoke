import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import numpy as np
import torch

from controllers.dual_guard import DualGuard
from controllers.mppi_ctrl import MPPICtrl, MPPICtrlParams

logger = logging.getLogger(__name__)


# ==========================================================
#  PRIVATE: MPPICtrl backed by DualGuard instead of MPPI
# ==========================================================
class _DualGuardMPPICtrl(MPPICtrl):
    """MPPICtrl whose planner is DualGuard.

    `safety_function` and `safe_control_function` are set to None at init and
    injected at each planning step by MultiDualGuardCBFCtrl, allowing fully
    decentralized per-agent CBF constraints.
    """

    def _init_mppi_planner(self) -> DualGuard:
        sigma = self.params.alpha_noise_sigma * np.diag(
            self.robot.robot_params.action_max - self.robot.robot_params.action_min
        )
        return DualGuard(
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            terminal_state_cost=self.terminal_state_cost,
            nx=self.robot.robot_params.state_dim,
            noise_sigma=torch.tensor(sigma, dtype=self.dtype, device=self.device),
            num_samples=self.params.num_samples,
            horizon=self.params.horizon,
            device=self.device,
            u_min=torch.tensor(
                self.robot.robot_params.action_min, dtype=self.dtype, device=self.device
            ),
            u_max=torch.tensor(
                self.robot.robot_params.action_max, dtype=self.dtype, device=self.device
            ),
            lambda_=self.params.lambda_,
            noise_abs_cost=self.params.noise_abs_cost,
            step_dependent_dynamics=True,
            safety_function=None,  # injected per planning step
            safe_control_function=None,  # injected per planning step
            safe_margin=0.0,
        )


# ==========================================================
#  MULTI-AGENT DECENTRALIZED DUALGUARD-CBF MANAGER
# ==========================================================
class MultiDualGuardCBFCtrl:
    """Centralized orchestrator for decentralized multi-agent collision-free
    planning using CBF safety functions inside DualGuard MPPI.

    Each agent holds an independent `_DualGuardMPPICtrl` (= MPPICtrl + DualGuard
    planner). At every step, per-agent CBF functions are wired into the planner:
        safety_function      = robot.cbf_h_function(states, neighbors, ...)  -> h(x)
        safe_control_function = robot.cbf_safe_control(states, neighbors, ...) -> u_safe (QP-CBF)

    For HJ-based safety (future): swap to safety_function = value_function,
    safe_control_function = least_restrictive_filter.
    """

    def __init__(
        self,
        num_agents: int,
        robot_params: Any,
        robot_type: str,
        goal_thresh: float = 0.1,
        device: str = "cpu",
        dtype=torch.float32,
        mppi_params: MPPICtrlParams = MPPICtrlParams(),
        dt: float = 0.1,
        r_sense: float = 6.0,
        d_safe: float = 1.2,
        k1: float = 2.5,
        k2: float = 2.5,
    ):
        self.num_agents = num_agents
        self.device = device
        self.dtype = dtype
        self.r_sense = r_sense
        self.d_safe = d_safe
        self.k1 = k1
        self.k2 = k2
        self.dt = dt

        self.agents_controllers: Dict[str, _DualGuardMPPICtrl] = {
            f"agent_{i}": _DualGuardMPPICtrl(
                robot_params=robot_params,
                robot_type=robot_type,
                goal_thresh=goal_thresh,
                device=device,
                dtype=dtype,
                mppi_params=mppi_params,
                dt=dt,
            )
            for i in range(num_agents)
        }

        # Persistent executor to avoid thread spawn/tear-down overhead on each step
        self.executor = ThreadPoolExecutor(max_workers=num_agents)

    # ======================================================
    #  EXTERNAL INTERFACE
    # ======================================================
    def set_goals(self, goals: Dict[str, Any]):
        """Set goals per agent. Format: {"agent_0": [gx, gy], ...}"""
        for key, goal_pos in goals.items():
            if key in self.agents_controllers:
                self.agents_controllers[key].set_goal(goal_pos)

    def set_maps(self, maps_deque: deque):
        """Broadcast a shared risk-map deque to all agent controllers."""
        for ctrl in self.agents_controllers.values():
            ctrl.set_maps(maps_deque)

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Decentralized planning step.

        Args:
            current_obs: observation dict from the environment, keyed by agent id.
                         Each value must contain at minimum: 'location' (2,), 'angle' (float).

        Returns:
            Dict mapping agent id -> control tensor.
        """
        # Build raw state tensors [x, y, theta] for all agents
        agents_states = {
            key: torch.tensor(
                [obs["location"][0], obs["location"][1], obs["angle"]],
                dtype=torch.float32,
            )
            for key, obs in current_obs.items()
            if obs is not None and key in self.agents_controllers
        }

        def plan_agent(ego_key: str, ego_ctrl: _DualGuardMPPICtrl) -> tuple:
            if ego_key not in agents_states:
                return ego_key, None

            ego_state = agents_states[ego_key]

            # 1. Sync robot state
            ego_ctrl.set_state(ego_state.numpy())

            # 2. Local sensing: neighbors within r_sense
            ego_pos = ego_state[:2]
            neighbors = [
                state
                for k, state in agents_states.items()
                if k != ego_key and torch.norm(ego_pos - state[:2]) <= self.r_sense
            ]

            # 3. Wire CBF safety functions into DualGuard for this step
            #    safety_function  : h(x)   (CBF value — positive = safe)
            #    safe_control_function : u_safe (QP-CBF analytical solution)
            u_min = ego_ctrl.planner.u_min
            u_max = ego_ctrl.planner.u_max

            ego_ctrl.planner.safety_function = lambda states, t: ego_ctrl.robot.cbf_h_function(
                states, neighbors, self.d_safe, self.k1, self.dt, t
            )
            ego_ctrl.planner.safe_control_function = lambda states, t: (
                ego_ctrl.robot.cbf_safe_control(
                    states, neighbors, self.d_safe, self.k1, self.k2, self.dt, t, u_min, u_max
                )
            )

            # 4. Plan
            return ego_key, ego_ctrl.get_command()

        commands: Dict[str, torch.Tensor] = {}

        if self.num_agents == 1:
            for key, ctrl in self.agents_controllers.items():
                k, cmd = plan_agent(key, ctrl)
                if cmd is not None:
                    commands[k] = cmd
        else:
            # Reuse the persistent executor to avoid per-step thread pool creation/destruction overhead
            futures = [
                self.executor.submit(plan_agent, key, ctrl)
                for key, ctrl in self.agents_controllers.items()
            ]
            for future in futures:
                key, cmd = future.result()
                if cmd is not None:
                    commands[key] = cmd

        return commands

    def visualize_rollouts(self, ax, draw_samples: bool = False):
        """Plot weighted-mean rollout paths for every agent."""
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        for i, (key, ctrl) in enumerate(self.agents_controllers.items()):
            ctrl.visualize_rollouts(ax, color=colors[i % len(colors)], draw_samples=draw_samples)


if __name__ == "__main__":
    import os

    import matplotlib.animation as animation
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from agents.basic_robot import RobotParams
    from agents.dubins_robot import DubinsRobot

    print("🧪 Iniciando simulación Swarm Circle Swap (10 drones) con HOCBF Shielding...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DT = 0.1
    N_DRONES = 10
    CIRCLE_RADIUS = 10.0
    D_SAFE = 1.2
    R_SENSE = 3.0
    GOAL_THRESH = 0.6

    # 1. Configuración de parámetros físicos
    robot_params = RobotParams(
        action_dim=2,
        state_dim=3,
        action_max=[6.0, 4.0],
        action_min=[0.0, -4.0],
        state_max=[30.0, 30.0, 2 * np.pi],
        state_min=[-30.0, -30.0, 0.0],
        robot_type="dubins2d",
        dt=DT,
    )

    mppi_params = MPPICtrlParams(num_samples=100, horizon=15, lambda_=1.2)
    multi_ctrl = MultiDualGuardCBFCtrl(
        num_agents=N_DRONES,
        robot_params=robot_params,
        robot_type="dubins2d",
        goal_thresh=GOAL_THRESH,
        device=device,
        mppi_params=mppi_params,
        dt=DT,
        r_sense=R_SENSE,
        d_safe=D_SAFE,
        k1=2.5,
        k2=2.5,
    )

    # 2. Establecer metas y posiciones iniciales
    goals_dict = {}
    trajectories = {f"agent_{i}": [] for i in range(N_DRONES)}
    sims = {}

    for i in range(N_DRONES):
        theta = i * (2 * np.pi / N_DRONES)
        px = CIRCLE_RADIUS * np.cos(theta)
        py = CIRCLE_RADIUS * np.sin(theta)
        gx = -px
        gy = -py

        goals_dict[f"agent_{i}"] = np.array([gx, gy], dtype=np.float32)
        angle_to_center = np.arctan2(-py, -px)

        sim = DubinsRobot(robot_params)
        sim.reset(np.array([px, py, angle_to_center], dtype=np.float32))
        sims[f"agent_{i}"] = sim
        trajectories[f"agent_{i}"].append(sim.get_state().copy())

    multi_ctrl.set_goals(goals_dict)

    # Dummy risk map deque para MPPI
    _xs = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, 0.5)
    _ys = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, 0.5)
    _coords = (
        np.stack(np.meshgrid(_xs, _ys, indexing="xy"), axis=-1).reshape(-1, 2).astype(np.float32)
    )
    _risk = np.zeros(len(_coords), dtype=np.float32)
    multi_ctrl.set_maps(deque([(_coords, _risk)] * mppi_params.horizon, maxlen=mppi_params.horizon))

    # 3. Ejecutar la simulación
    print("🚀 Iniciando bucle de control (60 pasos)...")
    max_steps = 120
    for step in tqdm(range(max_steps)):
        current_obs = {}
        for i in range(N_DRONES):
            key = f"agent_{i}"
            state = sims[key].get_state()
            current_obs[key] = {
                "location": state[:2],
                "angle": state[2],
            }

        actions = multi_ctrl.get_commands(current_obs)

        for i in range(N_DRONES):
            key = f"agent_{i}"
            u_cmd = actions[key].cpu().numpy()
            sims[key].dynamic_step(u_cmd)
            trajectories[key].append(sims[key].get_state().copy())

    print("🎉 Simulación completada.")

    # Calculate separation statistics
    min_dist_overall = float("inf")
    steps_recorded = len(trajectories["agent_0"])

    for s in range(steps_recorded):
        positions = []
        for i in range(N_DRONES):
            positions.append(trajectories[f"agent_{i}"][s][:2])
        for i in range(N_DRONES):
            for j in range(i + 1, N_DRONES):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist_overall:
                    min_dist_overall = d

    print(f"\nDistancia mínima de separación: {min_dist_overall:.4f} metros (D_SAFE = {D_SAFE})")

    # 4. Renderizar animación
    print("🎬 Renderizando animación GIF...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.set_ylim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Circular Swap - Decentralized HOCBF Shielded Swarm (Dubins)")

    spawn_circ = plt.Circle(
        (0, 0), CIRCLE_RADIUS, color="gray", fill=False, linestyle="--", alpha=0.5
    )
    ax.add_patch(spawn_circ)

    colors = plt.cm.rainbow(np.linspace(0, 1, N_DRONES))
    for i in range(N_DRONES):
        goal = goals_dict[f"agent_{i}"]
        ax.scatter(goal[0], goal[1], marker="*", color=colors[i], s=150, zorder=5)

    drone_patches = []
    trail_lines = []

    for i in range(N_DRONES):
        patch = plt.Circle((0, 0), D_SAFE / 2.0, color=colors[i], alpha=0.6)
        ax.add_patch(patch)
        drone_patches.append(patch)
        (line,) = ax.plot([], [], color=colors[i], linewidth=1.5, alpha=0.8)
        trail_lines.append(line)

    def init():
        for i in range(N_DRONES):
            drone_patches[i].center = (
                trajectories[f"agent_{i}"][0][0],
                trajectories[f"agent_{i}"][0][1],
            )
            trail_lines[i].set_data([], [])
        return drone_patches + trail_lines

    def animate(frame):
        for i in range(N_DRONES):
            key = f"agent_{i}"
            state = trajectories[key][frame]
            drone_patches[i].center = (state[0], state[1])
            trail_x = [pt[0] for pt in trajectories[key][: frame + 1]]
            trail_y = [pt[1] for pt in trajectories[key][: frame + 1]]
            trail_lines[i].set_data(trail_x, trail_y)
        return drone_patches + trail_lines

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=steps_recorded, interval=100, blit=True
    )

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "multi_drone_hocbf_swap.gif")
    ani.save(out_path, writer="pillow")
    print(f"✅ Animación guardada con éxito en '{out_path}'!")
