import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import numpy as np
import torch

from controllers.mppi_ctrl import MPPICtrl, MPPICtrlParams

logger = logging.getLogger(__name__)


# ==========================================================
#  PRIVATE: MPPICtrl with CBF Running Cost Penalty
# ==========================================================
class _CBFPenalizeMPPICtrl(MPPICtrl):
    """MPPICtrl whose running cost is augmented with a proportional CBF penalty.

    Penalizes samples that transgress the safety boundaries at the lookahead point
    and the robot center:
        C_cbf = C * max(-h(z_1) + alpha * h(z_2), 0)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_neighbors = []
        self.d_safe = 1.2
        self.k1 = 2.5
        self.C_pen = 1000.0
        self.alpha_pen = 1.0

    def running_cost(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        """Weighted sum of goal distance, risk cost, and proportional CBF penalty."""
        # 1. Base running cost (goal distance + environmental risk map)
        cost = super().running_cost(states, actions, t)

        # 2. Add proportional CBF penalty if neighbors exist
        if len(self.current_neighbors) == 0:
            return cost

        K = states.shape[0]
        device = states.device

        # Ego coordinates [x, y, theta]
        p_i_center = states[:, :2]
        theta = states[:, 2]

        # Ego safety/lookahead point
        p_i_safe = p_i_center + self.robot.L * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

        # Neighbor coordinates
        if isinstance(self.current_neighbors, list):
            neighbors_tensor = torch.stack(self.current_neighbors).to(device)
        else:
            neighbors_tensor = self.current_neighbors.to(device)

        p_j_center = neighbors_tensor[:, :2]
        theta_j = neighbors_tensor[:, 2]

        # Neighbor safety/lookahead point
        p_j_safe = p_j_center + self.robot.L * torch.stack([torch.cos(theta_j), torch.sin(theta_j)], dim=1)

        # Predict neighbor velocities and positions at future time t
        v_nominal = self.robot.action_max[0] / 2.0
        v_j_x = v_nominal * torch.cos(theta_j)
        v_j_y = v_nominal * torch.sin(theta_j)
        v_j = torch.stack([v_j_x, v_j_y], dim=1)  # (N_neigh, 2)

        # A. Lookahead point barrier (z_1)
        p_j_pred_safe = p_j_safe + v_j * (t * self.dt)
        d_safe_barrier = self.d_safe + 2.0 * self.robot.L + 0.2
        p_rel_safe = p_i_safe.unsqueeze(1) - p_j_pred_safe.unsqueeze(0)  # (K, N_neigh, 2)
        dist_sq_safe = torch.sum(p_rel_safe**2, dim=2)  # (K, N_neigh)
        h1_all = dist_sq_safe - d_safe_barrier**2  # (K, N_neigh)
        h1, _ = torch.min(h1_all, dim=1)  # (K,) critical lookahead barrier

        # B. Center point barrier (z_2)
        p_j_pred_center = p_j_center + v_j * (t * self.dt)
        p_rel_center = p_i_center.unsqueeze(1) - p_j_pred_center.unsqueeze(0)  # (K, N_neigh, 2)
        dist_sq_center = torch.sum(p_rel_center**2, dim=2)  # (K, N_neigh)
        h2_all = dist_sq_center - self.d_safe**2  # (K, N_neigh)
        h2, _ = torch.min(h2_all, dim=1)  # (K,) critical center barrier

        # Proportional CBF Penalty: Ccbf = C * max(-h1 + alpha * h2, 0)
        penalty = self.C_pen * torch.clamp(-h1 + self.alpha_pen * h2, min=0.0)

        return cost + penalty


# ==========================================================
#  MULTI-AGENT DECENTRALIZED CBF-PENALIZE MANAGER
# ==========================================================
class MultiCBFPenalizeCtrl:
    """Centralized orchestrator for decentralized multi-agent safe navigation
    using proportional CBF penalties directly in MPPI cost evaluation.
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
        C_pen: float = 1000.0,
        alpha_pen: float = 1.0,
    ):
        self.num_agents = num_agents
        self.device = device
        self.dtype = dtype
        self.r_sense = r_sense
        self.d_safe = d_safe
        self.k1 = k1
        self.C_pen = C_pen
        self.alpha_pen = alpha_pen
        self.dt = dt

        self.agents_controllers: Dict[str, _CBFPenalizeMPPICtrl] = {
            f"agent_{i}": _CBFPenalizeMPPICtrl(
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
        """Decentralized planning step with CBF-penalized cost functions."""
        agents_states = {
            key: torch.tensor(
                [obs["location"][0], obs["location"][1], obs["angle"]],
                dtype=torch.float32,
            )
            for key, obs in current_obs.items()
            if obs is not None and key in self.agents_controllers
        }

        def plan_agent(ego_key: str, ego_ctrl: _CBFPenalizeMPPICtrl) -> tuple:
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
            ego_ctrl.C_pen = self.C_pen
            ego_ctrl.alpha_pen = self.alpha_pen

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
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        for i, (key, ctrl) in enumerate(self.agents_controllers.items()):
            ctrl.visualize_rollouts(ax, color=colors[i % len(colors)], draw_samples=draw_samples)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from agents.basic_robot import RobotParams
    from agents.dubins_robot import DubinsRobot

    print("🧪 Iniciando simulación Swarm Circle Swap (10 drones) con CBF Penalize Controller...")

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
    multi_ctrl = MultiCBFPenalizeCtrl(
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
        C_pen=1500.0,
        alpha_pen=1.0,
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
    ax.set_title("Circular Swap - Swarm Circle Swap CBF Penalize Real-time (Dubins)")

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
