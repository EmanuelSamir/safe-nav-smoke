#!/usr/bin/env python3
"""
Test script for Shielded MPPI using a High-Order Control Barrier Function (HOCBF).
Sets up a 2D Double Integrator system dynamics and navigates around a circular obstacle.

For a Double Integrator, position is relative degree 2 with respect to acceleration control.
Therefore, standard CBF cannot be used directly, and we must formulate a 2nd-order HOCBF.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from controllers.shielded_mppi import ShieldedMPPI

# ==========================================================
#  1. DEFINE SIMULATION & DYNAMICS CONSTANTS
# ==========================================================
DT = 0.1
AMAX = 8.0   # Maximum acceleration limit
VMAX = 5.0   # Target maximum velocity limit for cost scaling

# System Dimensions
NX = 4  # States: [px, py, vx, vy]
NU = 2  # Controls: [ax, ay]

# Task Goals
GOAL = torch.tensor([10.0, 10.0])
START = torch.tensor([0.0, 0.0, 0.0, 0.0]) # At rest

# Obstacle Definition
OBSTACLE_CENTER = torch.tensor([5.0, 5.0])
OBSTACLE_RADIUS = 2.0

# HOCBF Gains
K1 = 2.5  # Gain for 1st-order CBF function
K2 = 2.5  # Gain for 2nd-order condition

# ==========================================================
#  2. MODULAR HIGH-ORDER SAFETY FUNCTIONS (HOCBF)
# ==========================================================
def hocbf_h(state_batch: torch.Tensor) -> torch.Tensor:
    """
    High-Order CBF Safety Index h_1(x).
    Checks both position and velocity compatibility to avoid the obstacle.
    
    State layout: [px, py, vx, vy]
    
    h0(x) = ||p - p_obs||^2 - R^2
    h1(x) = h0_dot(x) + K1 * h0(x)
          = 2 * (p - p_obs)^T * v + K1 * (||p - p_obs||^2 - R^2)
    """
    p = state_batch[:, :2]
    v = state_batch[:, 2:4]
    device = state_batch.device
    
    # Computes vector from obstacle center to position
    p_diff = p - OBSTACLE_CENTER.to(device)
    dist_sq = torch.sum(p_diff ** 2, dim=1)
    
    # Position set function
    h0 = dist_sq - OBSTACLE_RADIUS ** 2
    
    # Time derivative: h0_dot = 2 * (p - p_obs)^T * v
    h0_dot = 2.0 * torch.sum(p_diff * v, dim=1)
    
    # HOCBF safety index
    h1 = h0_dot + K1 * h0
    
    return h1

def hocbf_safe_control(state_batch: torch.Tensor) -> torch.Tensor:
    """
    Analytical Safety Backup Control Policy derived from HOCBF condition.
    
    We require:
    h1_dot >= -K2 * h1
    Since h1_dot = 2 * ||v||^2 + 2 * (p - p_obs)^T * a + K1 * h0_dot
    
    This can be rearranged as:
    A(x) * a >= B(x)
    where:
    A(x) = 2 * (p - p_obs)^T
    B(x) = -2 * ||v||^2 - K1 * h0_dot - K2 * h1
    
    We analyticaly project the zero acceleration onto this inequality constraint.
    """
    p = state_batch[:, :2]
    v = state_batch[:, 2:4]
    device = state_batch.device
    
    # Components
    p_diff = p - OBSTACLE_CENTER.to(device)
    dist_sq = torch.sum(p_diff ** 2, dim=1)
    h0 = dist_sq - OBSTACLE_RADIUS ** 2
    h0_dot = 2.0 * torch.sum(p_diff * v, dim=1)
    h1 = h0_dot + K1 * h0
    
    # Constraint coefficient matrix/vector A(x)
    A = 2.0 * p_diff  # (K, 2)
    
    # Constraint scalar value B(x)
    v_sq = torch.sum(v ** 2, dim=1)
    B = -2.0 * v_sq - K1 * h0_dot - K2 * h1
    
    # Minimum-norm correction projection
    A_norm_sq = torch.sum(A ** 2, dim=1)
    # Small epsilon to prevent division by zero if exactly centered on obstacle
    A_norm_sq = torch.clamp(A_norm_sq, min=1e-5)
    
    # Corrective acceleration logic:
    # If B > 0, standard zero acceleration violates A*a >= B.
    # We compute the minimum correction in the normal direction.
    u_safe = (torch.clamp(B, min=0.0) / A_norm_sq).unsqueeze(-1) * A
    
    # Clamp resulting acceleration safely to physical limits
    u_safe_norm = torch.norm(u_safe, dim=1, keepdim=True)
    u_safe = torch.where(
        u_safe_norm > AMAX,
        (u_safe / u_safe_norm) * AMAX,
        u_safe
    )
    
    return u_safe

# ==========================================================
#  3. TASK DYNAMICS & COSTS
# ==========================================================
def double_integrator_dynamics(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """
    Standard Double Integrator Dynamics.
    x_dot = v, v_dot = a
    State: [px, py, vx, vy], Control: [ax, ay]
    """
    p = state[:, :2]
    v = state[:, 2:4]
    a = control  # Acceleration is the control input
    
    # Euler integration steps
    p_next = p + v * DT
    v_next = v + a * DT
    
    # Combine back into continuous state tensor
    return torch.cat([p_next, v_next], dim=1)

def state_running_cost(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """
    Running Cost evaluating state performance.
    1. Promotes minimal distance to the goal.
    2. Adds dynamic cost for control efforts.
    """
    p = state[:, :2]
    v = state[:, 2:4]
    device = state.device
    
    # Distance to target
    dist_to_goal = torch.norm(p - GOAL.to(device), dim=1)
    
    # High velocity penalty when approaching goal (promotes stopping)
    stopping_penalty = torch.norm(v, dim=1) * torch.clamp(3.0 - dist_to_goal, min=0.0)
    
    # Total combined state cost
    return dist_to_goal + 0.5 * stopping_penalty

# ==========================================================
#  4. MAIN RUNNER
# ==========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Covariance Sigma for control perturbations (acceleration noise)
    noise_sigma = 4.0 * torch.eye(NU, dtype=torch.float32, device=device)
    
    # Configure modular ShieldedMPPI with HOCBF
    mppi = ShieldedMPPI(
        dynamics=double_integrator_dynamics,
        running_cost=state_running_cost,
        nx=NX,
        noise_sigma=noise_sigma,
        num_samples=150,                 # More samples for 4D state space
        horizon=25,                     # Predict ahead 2.5 seconds
        device=device,
        lambda_=1.2,                    # Free energy scale
        u_min=-AMAX * torch.ones(NU),   # Min acceleration
        u_max= AMAX * torch.ones(NU),   # Max acceleration
        cbf_h_function=hocbf_h,
        safe_control_function=hocbf_safe_control,
        safe_margin=0.0,                # Margin evaluated on h1
    )

    # Run simulator
    state = START.clone().to(device)
    traj = [state.cpu().numpy()]
    
    print("\nStarting Simulation with High-Order Control Barrier Functions (HOCBF)...")
    
    # Execute for a reasonable timeframe for second-order navigation
    max_steps = 120
    
    for step in tqdm(range(max_steps)):
        # Compute optimal shielded MPPI command
        # Reshapes input to match internally if 1D
        u_cmd = mppi.command(state)
        
        # Apply dynamics in simulated real world
        # Reshape state to (1, NX) for standard batch compatibility
        state_batch = state.unsqueeze(0)
        u_batch = u_cmd.unsqueeze(0)
        
        next_state = double_integrator_dynamics(state_batch, u_batch).squeeze(0)
        state = next_state
        traj.append(state.cpu().numpy())
        
        # Terminate if goal reached with low residual speed
        p = state[:2]
        v = state[2:4]
        dist_to_goal = torch.norm(p - GOAL.to(device)).item()
        speed = torch.norm(v).item()
        
        if dist_to_goal < 0.5 and speed < 0.5:
            print(f"\nGoal reached smoothly at step {step}!")
            break
            
    traj = np.array(traj)
    
    # ==========================================================
    #  5. VISUALIZATION
    # ==========================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot Safety Danger Zone (Geometry Obstacle)
    obstacle = plt.Circle(
        (OBSTACLE_CENTER[0], OBSTACLE_CENTER[1]), 
        OBSTACLE_RADIUS, 
        color='red', 
        alpha=0.4, 
        label='Obstacle (Position Danger Zone)'
    )
    ax.add_patch(obstacle)
    
    # Setup labels
    ax.scatter(START[0], START[1], marker='o', color='blue', s=100, label='Start')
    ax.scatter(GOAL[0], GOAL[1], marker='*', color='green', s=200, label='Goal')
    
    # Plot complete 2nd-order trajectory path
    ax.plot(traj[:, 0], traj[:, 1], '-bo', markersize=3, linewidth=2, label='HOCBF Shielded MPPI')
    
    # Plot dynamic velocity vectors along path to show inertia steering
    step_interval = 5
    for idx in range(0, len(traj), step_interval):
        px, py, vx, vy = traj[idx]
        ax.quiver(px, py, vx, vy, color='black', angles='xy', scale_units='xy', scale=2.0, 
                  alpha=0.6, width=0.005)
                  
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.grid(True)
    ax.set_title("Shielded MPPI with High-Order Control Barrier Functions (HOCBF)\n(Double Integrator System)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(loc='upper left')
    
    plt.savefig("shielded_mppi_hocbf_demo.png")
    print("\nSaved HOCBF simulation result to 'shielded_mppi_hocbf_demo.png'!")
    plt.show()

if __name__ == "__main__":
    main()
