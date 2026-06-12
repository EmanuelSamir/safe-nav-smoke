import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import numpy as np
import torch

from controllers.mppi_ctrl import MPPICtrl, MPPICtrlParams

logger = logging.getLogger(__name__)


# ==========================================================
#  PRIVATE: MPPICtrl with Output Safety Filtering
# ==========================================================
class _CBFFilteringMPPICtrl(MPPICtrl):
    """MPPICtrl whose output command is filtered at execution time.

    If the nominal MPPI optimized control is unsafe (violates the HOCBF constraint),
    it is hard-overridden with the analytical CBF safe control:
        if A * u_mppi < B:
            u_out = u_safe
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_neighbors = []
        self.d_safe = 1.2
        self.k1 = 2.5
        self.k2 = 2.5

    def compute_cbf_constraint(self, state: torch.Tensor, neighbors: list) -> tuple:
        """Compute the HOCBF constraint coefficients A and B at the current state.

        Constraint is: A * u >= B
        """
        device = state.device
        if len(neighbors) == 0:
            return None, None

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Ego state components
        p_i_center = state[:, :2]
        theta = state[:, 2]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Ego lookahead/safety point
        p_i_safe = p_i_center + self.robot.L * torch.stack([cos_t, sin_t], dim=1)

        # Neighbor states tensor
        if isinstance(neighbors, list):
            neighbors_tensor = torch.stack(neighbors).to(device)
        else:
            neighbors_tensor = neighbors.to(device)

        p_j_center = neighbors_tensor[:, :2]
        theta_j = neighbors_tensor[:, 2]

        # Neighbor lookahead/safety point
        p_j_safe = p_j_center + self.robot.L * torch.stack(
            [torch.cos(theta_j), torch.sin(theta_j)], dim=1
        )

        # Dynamic neighbor velocity prediction (at t=0)
        v_nominal = self.robot.action_max[0] / 2.0
        v_j_x = v_nominal * torch.cos(theta_j)
        v_j_y = v_nominal * torch.sin(theta_j)
        v_j = torch.stack([v_j_x, v_j_y], dim=1)  # (N_neigh, 2)

        # Relative safety positions
        p_rel = p_i_safe.unsqueeze(1) - p_j_safe.unsqueeze(0)  # (1, N_neigh, 2)
        dist_sq = torch.sum(p_rel**2, dim=2)  # (1, N_neigh)
        d_safe_barrier = self.d_safe + 2.0 * self.robot.L + 0.2
        h0_all = dist_sq - d_safe_barrier**2  # (1, N_neigh)

        # Find closest critical neighbor
        crit_idx = torch.argmin(h0_all, dim=1)

        h0_crit = h0_all[0, crit_idx]
        p_rel_crit = p_rel[0, crit_idx]
        v_j_crit = v_j[crit_idx]

        # A_v = 2 * p_rel^T * cos(theta)
        # A_w = 2 * p_rel^T * [-L*sin(theta); L*cos(theta)]
        A_v = 2.0 * (p_rel_crit[:, 0] * cos_t + p_rel_crit[:, 1] * sin_t)
        A_w = 2.0 * (
            -self.robot.L * p_rel_crit[:, 0] * sin_t + self.robot.L * p_rel_crit[:, 1] * cos_t
        )
        A = torch.stack([A_v, A_w], dim=1)  # (1, 2)

        # B = 2 * p_rel_crit^T * v_j_crit - k1 * h0_crit
        B = 2.0 * torch.sum(p_rel_crit * v_j_crit, dim=1) - self.k1 * h0_crit

        return A, B

    def get_command(self) -> torch.Tensor:
        """Compute nominal MPPI optimized control and apply analytical CBF output safety filter."""
        # 1. Compute nominal MPPI control command
        u_mppi = super().get_command()

        # 2. If no neighbors, nominal control is safe
        if len(self.current_neighbors) == 0:
            return u_mppi

        # 3. Retrieve current state
        state_np = self.robot.get_state()
        state = torch.tensor(state_np, dtype=self.dtype, device=self.device)

        # 4. Check safety against CBF constraint A * u >= B
        A, B = self.compute_cbf_constraint(state, self.current_neighbors)
        if A is not None:
            # A: (1, 2), u_mppi: (2,)
            A_u = torch.sum(A * u_mppi, dim=1)  # (1,)
            if A_u < B:
                # Unsafe nominal control! Hard override with analytical CBF backup control
                u_safe = self.robot.cbf_safe_control(
                    state.unsqueeze(0) if state.dim() == 1 else state,
                    self.current_neighbors,
                    self.d_safe,
                    self.k1,
                    self.k2,
                    self.dt,
                    0,  # t=0
                    self.planner.u_min,
                    self.planner.u_max,
                )
                return u_safe.squeeze(0)

        return u_mppi


# ==========================================================
#  MULTI-AGENT DECENTRALIZED CBF-FILTERING MANAGER
# ==========================================================
class MultiCBFFilteringCtrl:
    """Centralized orchestrator for decentralized multi-agent safe navigation
    using output CBF safety filters to override unsafe MPPI controls.
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

        self.agents_controllers: Dict[str, _CBFFilteringMPPICtrl] = {
            f"agent_{i}": _CBFFilteringMPPICtrl(
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

        self.executor = ThreadPoolExecutor(max_workers=num_agents)

    def set_goals(self, goals: Dict[str, Any]):
        """Set goals per agent. Format: {"agent_0": [gx, gy], ...}"""
        for key, goal_pos in goals.items():
            if key in self.agents_controllers:
                self.agents_controllers[key].set_goal(goal_pos)

    def set_maps(self, maps_deque: deque):
        """Broadcast shared risk-map deque to all agent controllers."""
        for ctrl in self.agents_controllers.values():
            ctrl.set_maps(maps_deque)

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Decentralized planning step with analytical output safety filtering."""
        agents_states = {
            key: torch.tensor(
                [obs["location"][0], obs["location"][1], obs["angle"]],
                dtype=torch.float32,
            )
            for key, obs in current_obs.items()
            if obs is not None and key in self.agents_controllers
        }

        def plan_agent(ego_key: str, ego_ctrl: _CBFFilteringMPPICtrl) -> tuple:
            if ego_key not in agents_states:
                return ego_key, None

            ego_state = agents_states[ego_key]
            ego_ctrl.set_state(ego_state.numpy())

            # Local neighbor sensing
            ego_pos = ego_state[:2]
            neighbors = [
                state
                for k, state in agents_states.items()
                if k != ego_key and torch.norm(ego_pos - state[:2]) <= self.r_sense
            ]

            # Inject neighbors and safety metrics
            ego_ctrl.current_neighbors = neighbors
            ego_ctrl.d_safe = self.d_safe
            ego_ctrl.k1 = self.k1
            ego_ctrl.k2 = self.k2

            return ego_key, ego_ctrl.get_command()

        commands: Dict[str, torch.Tensor] = {}

        if self.num_agents == 1:
            for key, ctrl in self.agents_controllers.items():
                k, cmd = plan_agent(key, ctrl)
                if cmd is not None:
                    commands[k] = cmd
        else:
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
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from agents.basic_robot import RobotParams
    from agents.dubins_robot import DubinsRobot

    print("🧪 Iniciando simulación Swarm Circle Swap (10 drones) con CBF Filtering Controller...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DT = 0.1
    N_DRONES = 10
    CIRCLE_RADIUS = 10.0
    D_SAFE = 1.6
    R_SENSE = 4.0
    GOAL_THRESH = 0.6

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
    multi_ctrl = MultiCBFFilteringCtrl(
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

    # Dummy risk map deque for MPPI
    _xs = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, 0.5)
    _ys = np.arange(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2, 0.5)
    _coords = (
        np.stack(np.meshgrid(_xs, _ys, indexing="xy"), axis=-1).reshape(-1, 2).astype(np.float32)
    )
    _risk = np.zeros(len(_coords), dtype=np.float32)
    multi_ctrl.set_maps(deque([(_coords, _risk)] * mppi_params.horizon, maxlen=mppi_params.horizon))

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.set_ylim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Circular Swap - Swarm Circle Swap CBF Filtering Real-time (Dubins)")

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
        init_state = trajectories[f"agent_{i}"][0]
        patch = plt.Circle((init_state[0], init_state[1]), D_SAFE / 2.0, color=colors[i], alpha=0.6)
        ax.add_patch(patch)
        drone_patches.append(patch)
        (line,) = ax.plot(
            [init_state[0]], [init_state[1]], color=colors[i], linewidth=1.5, alpha=0.8
        )
        trail_lines.append(line)

    plt.draw()
    plt.pause(0.1)

    print("🚀 Iniciando bucle de control (120 pasos)...")
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
            state = sims[key].get_state().copy()
            trajectories[key].append(state)

            drone_patches[i].center = (state[0], state[1])
            trail_x = [pt[0] for pt in trajectories[key]]
            trail_y = [pt[1] for pt in trajectories[key]]
            trail_lines[i].set_data(trail_x, trail_y)

        plt.draw()
        plt.pause(0.001)

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

    plt.ioff()
    plt.show()
