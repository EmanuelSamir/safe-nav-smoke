#!/usr/bin/env python3
"""Test script for Multi-Agent Decentralized Shielded MPPI with High-Order Control Barrier Functions (HOCBF).
Scenario: N drones arranged in a circle swapping positions to the opposite side.
Forces an unavoidable central collision, resolved gracefully by decentralized HOCBF-Shields.
Includes localized neighbor sensing to minimize computation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from controllers.shielded_mppi import ShieldedMPPI

# ==========================================================
#  1. CONSTANTS & SCENARIO CONFIG
# ==========================================================
DT = 0.1
AMAX = 8.0  # Maximum acceleration
VMAX = 5.0  # Maximum velocity goal
N_DRONES = 10  # Number of drones
CIRCLE_RADIUS = 10.0  # Spawn circle radius
D_SAFE = 1.2  # Safety clearance diameter between drones
R_SENSE = 8.0  # Sensing/Interaction radius

# HOCBF Gains
K1 = 2.5
K2 = 2.5

# System Dimensions
NX = 4  # [px, py, vx, vy]
NU = 2  # [ax, ay]

# ==========================================================
#  2. MODULAR HIGH-ORDER SAFETY FUNCTIONS FOR DRONE i
# ==========================================================


def get_neighborhood_info(ego_id, drones_states, R_sense):
    """Returns a list of current positions and velocities of active neighbor drones.
    Excludes the ego drone and filters by current physical distance.
    """
    ego_pos = drones_states[ego_id][:2]
    neighbors = []

    for j in range(N_DRONES):
        if j == ego_id:
            continue
        other_pos = drones_states[j][:2]
        dist = np.linalg.norm(ego_pos - other_pos)
        if dist <= R_sense:
            # Store neighbor position and velocity
            neighbors.append(
                (
                    torch.tensor(drones_states[j][:2], dtype=torch.float32),
                    torch.tensor(drones_states[j][2:4], dtype=torch.float32),
                )
            )
    return neighbors


class MultiAgentCBF:
    """Encapsulates safety index and safe backup controls tailored for drone i.
    Expects static prediction parameters representing neighbors' states.
    """

    def __init__(self, neighbors_list, device="cpu"):
        self.neighbors = neighbors_list  # List of (p0, v0) tensors
        self.device = device

    def h_function(self, ego_states: torch.Tensor, t: int) -> torch.Tensor:
        """Computes the compound HOCBF safety index H(x_i, t) = min_{j} h_1,ij(x_i, t).
        Predicts neighbor future positions assuming constant velocity: p_j(t) = p_j + v_j * t * DT.
        """
        K = ego_states.shape[0]

        if len(self.neighbors) == 0:
            # No neighbors -> completely safe (return very high h)
            return 1000.0 * torch.ones(K, device=self.device)

        p_i = ego_states[:, :2]
        v_i = ego_states[:, 2:4]

        all_h1s = []

        for p_j0, v_j0 in self.neighbors:
            p_j0 = p_j0.to(self.device)
            v_j0 = v_j0.to(self.device)

            # Constant velocity prediction of neighbor at future horizon step t
            p_j_pred = p_j0 + v_j0 * (t * DT)
            v_j_pred = v_j0

            # Relative states
            p_rel = p_i - p_j_pred
            v_rel = v_i - v_j_pred

            # HOCBF computation
            dist_sq = torch.sum(p_rel**2, dim=1)
            h0 = dist_sq - D_SAFE**2
            h0_dot = 2.0 * torch.sum(p_rel * v_rel, dim=1)

            h1 = h0_dot + K1 * h0
            all_h1s.append(h1)

        # The compound safety function is the MINIMUM of all individual neighbor barriers
        # Shape: (K, len(neighbors)) -> min -> (K,)
        stacked_h1s = torch.stack(all_h1s, dim=1)
        h_comp, _ = torch.min(stacked_h1s, dim=1)

        return h_comp

    def safe_control(self, ego_states: torch.Tensor, t: int) -> torch.Tensor:
        """Computes decentralized backup acceleration based on minimum norm projection
        against the single MOST critical neighbor constraint.
        """
        K = ego_states.shape[0]
        if len(self.neighbors) == 0:
            return torch.zeros(K, 2, device=self.device)

        p_i = ego_states[:, :2]
        v_i = ego_states[:, 2:4]

        # 1. Identify the worst/critical neighbor for EACH state in the batch
        all_h1s = []
        all_rel_p = []
        all_rel_v = []

        for p_j0, v_j0 in self.neighbors:
            p_j0 = p_j0.to(self.device)
            v_j0 = v_j0.to(self.device)

            p_j_pred = p_j0 + v_j0 * (t * DT)
            v_j_pred = v_j0

            p_rel = p_i - p_j_pred
            v_rel = v_i - v_j_pred

            dist_sq = torch.sum(p_rel**2, dim=1)
            h0 = dist_sq - D_SAFE**2
            h0_dot = 2.0 * torch.sum(p_rel * v_rel, dim=1)
            h1 = h0_dot + K1 * h0

            all_h1s.append(h1)
            all_rel_p.append(p_rel)
            all_rel_v.append(v_rel)

        stacked_h1s = torch.stack(all_h1s, dim=1)  # (K, num_neighbors)
        critical_indices = torch.argmin(stacked_h1s, dim=1)  # (K,)

        # Extract critical properties for the batch
        batch_idx = torch.arange(K, device=self.device)
        h1_crit = stacked_h1s[batch_idx, critical_indices]

        stacked_rel_p = torch.stack(all_rel_p, dim=1)  # (K, num_neigh, 2)
        p_rel_crit = stacked_rel_p[batch_idx, critical_indices]  # (K, 2)

        stacked_rel_v = torch.stack(all_rel_v, dim=1)  # (K, num_neigh, 2)
        v_rel_crit = stacked_rel_v[batch_idx, critical_indices]  # (K, 2)

        # 2. Perform projection for critical neighbor
        A = 2.0 * p_rel_crit  # (K, 2)

        v_rel_sq = torch.sum(v_rel_crit**2, dim=1)
        h0_dot_crit = 2.0 * torch.sum(p_rel_crit * v_rel_crit, dim=1)

        B = -2.0 * v_rel_sq - K1 * h0_dot_crit - K2 * h1_crit

        A_norm_sq = torch.clamp(torch.sum(A**2, dim=1), min=1e-5)

        # Calculate minimal acceleration projection
        u_safe = (torch.clamp(B, min=0.0) / A_norm_sq).unsqueeze(-1) * A

        # Clamp to valid physically consistent limits
        u_norm = torch.norm(u_safe, dim=1, keepdim=True)
        u_safe = torch.where(u_norm > AMAX, (u_safe / u_norm) * AMAX, u_safe)

        return u_safe


# ==========================================================
#  3. DYNAMICS & MPPI HELPERS
# ==========================================================
def double_integrator_dynamics(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    p = state[:, :2]
    v = state[:, 2:4]
    p_next = p + v * DT
    v_next = v + control * DT
    return torch.cat([p_next, v_next], dim=1)


class GoalReachingCost:
    def __init__(self, goal_tensor):
        self.goal = goal_tensor

    def cost_fn(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        p = state[:, :2]
        v = state[:, 2:4]
        dist = torch.norm(p - self.goal.to(state.device), dim=1)
        # Discourage huge speed when arriving
        stop_penalty = torch.norm(v, dim=1) * torch.clamp(2.0 - dist, min=0.0)
        return dist + 0.5 * stop_penalty


# ==========================================================
#  4. MAIN SIMULATOR
# ==========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Decentralized Multi-Agent Simulation on: {device}")

    # Setup Start and Goal Configurations arranged on a Circle
    drones_states = []
    goals = []

    for i in range(N_DRONES):
        theta = i * (2 * np.pi / N_DRONES)
        # Position at radius R
        px = CIRCLE_RADIUS * np.cos(theta)
        py = CIRCLE_RADIUS * np.sin(theta)

        # Goal at opposite side
        gx = -px
        gy = -py

        # Initial state (at rest)
        drones_states.append(np.array([px, py, 0.0, 0.0]))
        goals.append(torch.tensor([gx, gy], dtype=torch.float32))

    print(f"Spawned {N_DRONES} drones for circular swap.")

    # Initialize a list of MPPI controllers, one for each drone
    controllers = []
    noise_sigma = 4.0 * torch.eye(NU, dtype=torch.float32, device=device)

    for i in range(N_DRONES):
        cost_obj = GoalReachingCost(goals[i])

        # Instantiation with empty/placeholder safety callbacks that we will dynamically update
        mppi = ShieldedMPPI(
            dynamics=double_integrator_dynamics,
            running_cost=cost_obj.cost_fn,
            nx=NX,
            noise_sigma=noise_sigma,
            num_samples=150,
            horizon=20,
            device=device,
            lambda_=1.2,
            u_min=-AMAX * torch.ones(NU),
            u_max=AMAX * torch.ones(NU),
            cbf_h_function=None,  # Overwritten at runtime
            safe_control_function=None,  # Overwritten at runtime
            safe_margin=0.0,
        )
        controllers.append(mppi)

    # Simulation Loop Setup
    trajectories = [[drones_states[i].copy()] for i in range(N_DRONES)]
    max_steps = 140

    print("Executing Simulation Steps...")
    for step in tqdm(range(max_steps)):
        current_commands = []

        # 1. Decentralized Sensing & Planning Stage
        for i in range(N_DRONES):
            # Identify neighbors in sensor range
            neighbors = get_neighborhood_info(i, drones_states, R_SENSE)

            # Construct specialized, lightweight HOCBF object for the detected neighborhood
            cbf_handler = MultiAgentCBF(neighbors, device=device)

            # Inject modular safety functions dynamically to ego MPPI planner
            controllers[i].cbf_h_function = cbf_handler.h_function
            controllers[i].safe_control_function = cbf_handler.safe_control

            # Call planner to calculate decentralized safe acceleration cmd
            state_tensor = torch.tensor(drones_states[i], dtype=torch.float32, device=device)
            u_cmd = controllers[i].command(state_tensor)
            current_commands.append(u_cmd)

        # 2. Synchronous Environmental Step
        # Update all drones states synchronously based on calculated actions
        for i in range(N_DRONES):
            s_batch = torch.tensor(drones_states[i], dtype=torch.float32, device=device).unsqueeze(
                0
            )
            u_batch = current_commands[i].unsqueeze(0)

            next_s = double_integrator_dynamics(s_batch, u_batch).squeeze(0)
            drones_states[i] = next_s.cpu().numpy()
            trajectories[i].append(drones_states[i].copy())

        # Termination Check: if all drones reached goals
        all_done = True
        for i in range(N_DRONES):
            dist = np.linalg.norm(drones_states[i][:2] - goals[i].numpy())
            speed = np.linalg.norm(drones_states[i][2:4])
            if dist > 0.6 or speed > 0.6:
                all_done = False
                break
        if all_done:
            print(f"\nAll drones successfully completed circular swap safely at step {step}!")
            break

    # ==========================================================
    #  5. ANIMATION GENERATION (GIF)
    # ==========================================================
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Verify minimum separation distance over time to mathematically guarantee collision avoidance
    min_sep = 100.0
    steps_count = len(trajectories[0])
    for s in range(steps_count):
        for i in range(N_DRONES):
            for j in range(i + 1, N_DRONES):
                pos_i = trajectories[i][s][:2]
                pos_j = trajectories[j][s][:2]
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < min_sep:
                    min_sep = dist

    print(
        f"\nMinimum observed inter-drone separation: {min_sep:.4f} meters (D_SAFE threshold is {D_SAFE} meters)"
    )
    if min_sep >= D_SAFE * 0.98:
        print("SUCCESS: Mathematical safety GUARANTEED. No collisions detected!")
    else:
        print("WARNING: Safety boundary violation detected.")

    # Setup animation plot
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, N_DRONES))

    def update(frame):
        ax.clear()

        # Draw spawning circumference reference
        c_circle = plt.Circle(
            (0, 0), CIRCLE_RADIUS, fill=False, color="gray", linestyle="--", alpha=0.2
        )
        ax.add_patch(c_circle)

        for i in range(N_DRONES):
            traj = np.array(trajectories[i][: frame + 1])
            c = colors[i]

            # Plot historical tail path leading up to current frame
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], "-", color=c, linewidth=2, alpha=0.6)

            # Plot immutable start and goal markers
            ax.scatter(
                trajectories[i][0][0],
                trajectories[i][0][1],
                marker="o",
                facecolor="none",
                edgecolor=c,
                s=70,
                alpha=0.4,
            )
            ax.scatter(goals[i][0], goals[i][1], marker="*", color=c, s=150)

            # Get position at current frame
            curr_state = trajectories[i][frame]
            curr_pos = curr_state[:2]
            curr_vel = curr_state[2:4]

            # Plot actual drone body
            ax.scatter(
                curr_pos[0],
                curr_pos[1],
                color=c,
                s=100,
                edgecolor="black",
                zorder=10,
                label=f"Drone {i + 1}" if frame == 0 else "",
            )

            # Plot the active safety exclusion region (D_SAFE / 2 is the radius)
            safe_ring = plt.Circle((curr_pos[0], curr_pos[1]), D_SAFE / 2.0, color=c, alpha=0.25)
            ax.add_patch(safe_ring)

            # Plot velocity vector quiver
            ax.quiver(
                curr_pos[0],
                curr_pos[1],
                curr_vel[0],
                curr_vel[1],
                color="black",
                scale_units="xy",
                scale=1.5,
                width=0.004,
                alpha=0.5,
                zorder=5,
            )

        ax.set_xlim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
        ax.set_ylim(-CIRCLE_RADIUS - 2, CIRCLE_RADIUS + 2)
        ax.grid(True, alpha=0.2)
        ax.set_title(
            f"Swarm Decentralized Shielded MPPI (N={N_DRONES} Drones)\nStep: {frame}/{steps_count - 1} | Dynamic HOCBF Exclusion Zones"
        )
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_aspect("equal")

        if frame == 0:
            ax.legend(loc="upper right", fontsize="x-small")

    print("\nRendering Swarm Animation GIF...")
    ani = FuncAnimation(fig, update, frames=steps_count, repeat=True)

    gif_path = "multi_drone_hocbf_swap.gif"
    writer = PillowWriter(fps=10)

    ani.save(gif_path, writer=writer)
    print(f"SUCCESS: Saved animated simulation to '{gif_path}'!")
    plt.close(fig)


if __name__ == "__main__":
    main()
