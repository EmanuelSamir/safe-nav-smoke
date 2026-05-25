import logging
import numpy as np
import torch
from collections import deque
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from controllers.mppi_control_dyn import MPPIControlDyn, MPPIControlParams
from controllers.shielded_mppi import ShieldedMPPI

logger = logging.getLogger(__name__)

# ==========================================================
#  1. SINGLE-AGENT SHIELDED WRAPPER
# ==========================================================
class ShieldedMPPIControlDyn(MPPIControlDyn):
    """
    High-level wrapper for Shielded-MPPI-based control of a single agent.
    Inherits all map and cost integration from nominal MPPIControlDyn but uses
    ShieldedMPPI to inject safety overrides.
    """
    def __init__(self, *args, safe_margin: float = 0.0, **kwargs):
        self.safe_margin = safe_margin
        super().__init__(*args, **kwargs)

    def _init_mppi_planner(self) -> ShieldedMPPI:
        """
        Override to instantiate ShieldedMPPI instead of standard MPPI.
        """
        sigma = self.params.alpha_noise_sigma * np.diag(
            self.robot.robot_params.action_max - self.robot.robot_params.action_min
        )

        config = dict(
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
            cbf_h_function=None,         # Dynamically injected during planning
            safe_control_function=None,  # Dynamically injected during planning
            safe_margin=self.safe_margin,
        )
        return ShieldedMPPI(**config)


# ==========================================================
#  2. MULTI-AGENT CBF VECTORIZED LOGIC
# ==========================================================
class AgentAgentCBF:
    """
    Encapsulates vectorized HOCBF safety index and projection computations
    specifically designed for nonholonomic multi-agent scenarios.
    Supports: 'dubins2d', 'dubins2d_fixed_velocity', and 'unicycle2d'.
    """
    def __init__(
        self,
        robot_type: str,
        neighbors_list: list,
        d_safe: float,
        k1: float,
        k2: float,
        dt: float,
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        device: str = "cpu"
    ):
        self.robot_type = robot_type
        self.neighbors = neighbors_list  # list of dicts with keys: 'p_safe', 'v_safe', 'p_center', 'v_center'
        self.d_safe = d_safe
        self.k1 = k1
        self.k2 = k2
        self.dt = dt
        self.u_min = u_min.to(device)
        self.u_max = u_max.to(device)
        self.device = device
        self.L = 0.5  # Lookahead distance for Dubins safety point

    def h_function(self, ego_states: torch.Tensor, t: int) -> torch.Tensor:
        K = ego_states.shape[0]
        if len(self.neighbors) == 0:
            return 1000.0 * torch.ones(K, device=self.device)

        if self.robot_type == "unicycle":
            # Unicycle: State [x, y, theta, v]. Direct center HOCBF (Rel Degree 2).
            p_i = ego_states[:, :2]
            theta = ego_states[:, 2]
            v = ego_states[:, 3]
            v_i = torch.stack([v * torch.cos(theta), v * torch.sin(theta)], dim=1)

            all_h1s = []
            for n in self.neighbors:
                # Constant velocity prediction of neighbor center
                p_j_pred = n['p_center'].to(self.device) + n['v_center'].to(self.device) * (t * self.dt)
                v_j_pred = n['v_center'].to(self.device)

                p_rel = p_i - p_j_pred
                v_rel = v_i - v_j_pred

                dist_sq = torch.sum(p_rel ** 2, dim=1)
                h0 = dist_sq - self.d_safe ** 2
                h0_dot = 2.0 * torch.sum(p_rel * v_rel, dim=1)

                h1 = h0_dot + self.k1 * h0
                all_h1s.append(h1)

            stacked_h1s = torch.stack(all_h1s, dim=1)
            h_comp, _ = torch.min(stacked_h1s, dim=1)
            return h_comp

        else:
            # Dubins (fixed or variable): State [x, y, theta]. Lookahead Point HOCBF (Rel Degree 1).
            theta = ego_states[:, 2]
            p_i_center = ego_states[:, :2]
            # Safety point
            p_i_safe = p_i_center + self.L * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

            all_h0s = []
            for n in self.neighbors:
                # Predict neighbor safety point
                p_j_pred = n['p_safe'].to(self.device) + n['v_safe'].to(self.device) * (t * self.dt)
                p_rel = p_i_safe - p_j_pred

                dist_sq = torch.sum(p_rel ** 2, dim=1)
                h0 = dist_sq - self.d_safe ** 2
                all_h0s.append(h0)

            stacked_h0s = torch.stack(all_h0s, dim=1)
            h_comp, _ = torch.min(stacked_h0s, dim=1)
            return h_comp

    def safe_control(self, ego_states: torch.Tensor, t: int) -> torch.Tensor:
        K = ego_states.shape[0]
        if len(self.neighbors) == 0:
            return torch.zeros(K, self.u_min.shape[0], device=self.device)

        if self.robot_type == "unicycle":
            # Unicycle HOCBF (Rel Degree 2) Vectorized Projection
            p_i = ego_states[:, :2]
            theta = ego_states[:, 2]
            v = ego_states[:, 3]
            v_i = torch.stack([v * torch.cos(theta), v * torch.sin(theta)], dim=1)

            all_h1s, all_rel_p, all_rel_v = [], [], []
            for n in self.neighbors:
                p_j_pred = n['p_center'].to(self.device) + n['v_center'].to(self.device) * (t * self.dt)
                v_j_pred = n['v_center'].to(self.device)

                p_rel = p_i - p_j_pred
                v_rel = v_i - v_j_pred

                h0 = torch.sum(p_rel ** 2, dim=1) - self.d_safe ** 2
                h0_dot = 2.0 * torch.sum(p_rel * v_rel, dim=1)
                h1 = h0_dot + self.k1 * h0

                all_h1s.append(h1)
                all_rel_p.append(p_rel)
                all_rel_v.append(v_rel)

            stacked_h1s = torch.stack(all_h1s, dim=1)
            crit_indices = torch.argmin(stacked_h1s, dim=1)
            batch_idx = torch.arange(K, device=self.device)

            h1_crit = stacked_h1s[batch_idx, crit_indices]
            p_rel_crit = torch.stack(all_rel_p, dim=1)[batch_idx, crit_indices]
            v_rel_crit = torch.stack(all_rel_v, dim=1)[batch_idx, crit_indices]

            # Unicycle Dynamics Matrix: M(theta, v)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            
            # A_coeff = 2 * p_rel^T * M
            # Row 1: cos(t), -v*sin(t)
            # Row 2: sin(t), v*cos(t)
            A_a = 2.0 * (p_rel_crit[:, 0] * cos_t + p_rel_crit[:, 1] * sin_t)
            A_w = 2.0 * (-v * p_rel_crit[:, 0] * sin_t + v * p_rel_crit[:, 1] * cos_t)
            A = torch.stack([A_a, A_w], dim=1)  # (K, 2)

            v_rel_sq = torch.sum(v_rel_crit ** 2, dim=1)
            h0_dot_crit = 2.0 * torch.sum(p_rel_crit * v_rel_crit, dim=1)
            B = -2.0 * v_rel_sq - self.k1 * h0_dot_crit - self.k2 * h1_crit

            A_norm_sq = torch.clamp(torch.sum(A**2, dim=1), min=1e-5)
            u_safe = (torch.clamp(B, min=0.0) / A_norm_sq).unsqueeze(-1) * A

        else:
            # Dubins Lookahead HOCBF (Rel Degree 1) Vectorized Projection
            theta = ego_states[:, 2]
            p_i_center = ego_states[:, :2]
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            p_i_safe = p_i_center + self.L * torch.stack([cos_t, sin_t], dim=1)

            all_h0s, all_rel_p, all_v_neighbor_safe = [], [], []
            for n in self.neighbors:
                p_j_pred = n['p_safe'].to(self.device) + n['v_safe'].to(self.device) * (t * self.dt)
                p_rel = p_i_safe - p_j_pred

                h0 = torch.sum(p_rel ** 2, dim=1) - self.d_safe ** 2
                all_h0s.append(h0)
                all_rel_p.append(p_rel)
                all_v_neighbor_safe.append(n['v_safe'].to(self.device))

            stacked_h0s = torch.stack(all_h0s, dim=1)
            crit_indices = torch.argmin(stacked_h0s, dim=1)
            batch_idx = torch.arange(K, device=self.device)

            h0_crit = stacked_h0s[batch_idx, crit_indices]
            p_rel_crit = torch.stack(all_rel_p, dim=1)[batch_idx, crit_indices]
            
            # Stack constant neighbor velocities along dim 0 -> (num_neighbors, 2)
            # and index by critical indices to expand along the particle dimension (K)
            v_j_safe_tensor = torch.stack(all_v_neighbor_safe, dim=0)
            v_j_safe_crit = v_j_safe_tensor[crit_indices] # (K, 2)

            # Coefficients for action projection: R(theta, L)
            # Safety point speed = R * u where R = [cos(t), -L*sin(t); sin(t), L*cos(t)]
            # A_coeff = 2 * p_rel^T * R
            A_v = 2.0 * (p_rel_crit[:, 0] * cos_t + p_rel_crit[:, 1] * sin_t)
            A_w = 2.0 * (-self.L * p_rel_crit[:, 0] * sin_t + self.L * p_rel_crit[:, 1] * cos_t)

            if self.robot_type == "dubins2d_fixed_velocity":
                # 1D control: only angular velocity 'w'. Action dim = 1.
                A = A_w.unsqueeze(-1) # (K, 1)
                # Extract fixed velocity parameter from robot configs or action max
                v_fixed = self.u_max[0] if self.u_max.ndim > 0 else 1.0 
                term_v = 2.0 * (p_rel_crit[:, 0] * cos_t + p_rel_crit[:, 1] * sin_t) * v_fixed
                
                B = 2.0 * torch.sum(p_rel_crit * v_j_safe_crit, dim=1) - self.k1 * h0_crit - term_v
            else:
                # 2D control: [v, w]
                A = torch.stack([A_v, A_w], dim=1) # (K, 2)
                B = 2.0 * torch.sum(p_rel_crit * v_j_safe_crit, dim=1) - self.k1 * h0_crit

            A_norm_sq = torch.clamp(torch.sum(A**2, dim=1), min=1e-5)
            u_safe = (torch.clamp(B, min=0.0) / A_norm_sq).unsqueeze(-1) * A

        # Clamp analytically computed backup controls to physical bounds
        u_safe_clamped = torch.max(torch.min(u_safe, self.u_max), self.u_min)
        return u_safe_clamped


# ==========================================================
#  3. MULTI-AGENT DECENTRALIZED CONTROLLER MANAGER
# ==========================================================
class MultiShieldedMPPIControl:
    """
    Centralized orchestrator for decentralized multi-agent collision-free planning.
    Holds N independent ShieldedMPPI controllers, processes local neighborhood perception
    per agent, and dynamically injects analytical HOCBF exclusion constraints.
    """
    def __init__(
        self,
        num_agents: int,
        robot_params: Any,
        robot_type: str,
        goal_thresh: float = 0.1,
        device: str = "cpu",
        dtype=torch.float32,
        mppi_params: MPPIControlParams = MPPIControlParams(),
        dt: float = 0.1,
        r_sense: float = 6.0,
        d_safe: float = 1.2,
        k1: float = 2.5,
        k2: float = 2.5,
    ):
        self.num_agents = num_agents
        self.robot_type = robot_type.replace("2d", "") # Clean type e.g. unicycle
        self.device = device
        self.dtype = dtype
        self.r_sense = r_sense
        self.d_safe = d_safe
        self.k1 = k1
        self.k2 = k2
        self.dt = dt
        
        # Instantiate specific controller for each agent
        self.agents_controllers: Dict[str, ShieldedMPPIControlDyn] = {}
        for i in range(num_agents):
            agent_key = f"agent_{i}"
            self.agents_controllers[agent_key] = ShieldedMPPIControlDyn(
                robot_params=robot_params,
                robot_type=robot_type,
                goal_thresh=goal_thresh,
                device=device,
                dtype=dtype,
                mppi_params=mppi_params,
                dt=dt,
                safe_margin=0.0
            )
            
    # ======================================================
    #  CORE EXTERNAL INTERFACE
    # ======================================================
    def set_goals(self, goals: Dict[str, Any]):
        """
        Set goals for each agent. Format: {"agent_0": [gx, gy], ...}
        """
        for agent_key, goal_pos in goals.items():
            if agent_key in self.agents_controllers:
                self.agents_controllers[agent_key].set_goal(goal_pos)

    def set_maps(self, maps_deque: deque):
        """
        Broadcast risk maps deque to all agent controllers for running cost computation.
        """
        for ctrl in self.agents_controllers.values():
            ctrl.set_maps(maps_deque)

    def get_commands(self, current_obs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Main decentralized step. Receives the observation dictionary from SmokeEnv,
        extracts state/velocities, computes localized multi-agent constraints,
        and evaluates optimal safe MPPI commands for every active agent.
        """
        # 1. Extract all physical properties for neighbor modeling
        agents_physical = {}
        for i in range(self.num_agents):
            key = f"agent_{i}"
            obs = current_obs.get(key)
            if obs is None:
                continue

            # Basic physical centers
            pos = obs["location"]  # [x, y]
            theta = float(obs["angle"])
            
            # Determine instantaneous velocity vectors
            if "velocity" in obs:
                # Unicycle
                speed = float(obs["velocity"])
            else:
                # Dubins or DubinsFixed: assume speed based on past step or nominal action_max[0]
                # SmokeEnv odom doesn't supply Dubins speed, but agent holds it in its internal robot object.
                speed = float(self.agents_controllers[key].robot.get_state()[2]) if len(self.agents_controllers[key].robot.get_state()) > 3 else float(self.agents_controllers[key].robot.robot_params.action_max[0] / 2.0)

            vx = speed * np.cos(theta)
            vy = speed * np.sin(theta)
            v_center = np.array([vx, vy])

            # Calculate safety point and its velocity
            L = 0.5
            p_safe = pos + L * np.array([np.cos(theta), np.sin(theta)])
            # In steady constant velocity prediction, assume angular velocity = 0 for neighbor extrapolation
            v_safe = v_center 

            # Construct clean 1D raw numpy state array for physical planning alignment
            if "velocity" in obs:
                state_raw = np.array([pos[0], pos[1], theta, speed], dtype=np.float32)
            else:
                state_raw = np.array([pos[0], pos[1], theta], dtype=np.float32)

            agents_physical[key] = {
                'p_center': torch.tensor(pos, dtype=torch.float32),
                'v_center': torch.tensor(v_center, dtype=torch.float32),
                'p_safe': torch.tensor(p_safe, dtype=torch.float32),
                'v_safe': torch.tensor(v_safe, dtype=torch.float32),
                'state_raw': state_raw
            }

        commands = {}

        # Helper function to run decouple safe planning for a single agent in parallel
        def plan_agent(ego_key, ego_ctrl):
            if ego_key not in agents_physical:
                return ego_key, None

            # 1. Update ego robot state object
            ego_ctrl.set_state(agents_physical[ego_key]['state_raw'])

            # 2. Perform local sensing: find neighbors within sensing radius
            neighbors_detected = []
            ego_pos = agents_physical[ego_key]['p_center'].numpy()

            for other_key, other_phys in agents_physical.items():
                if other_key == ego_key:
                    continue
                other_pos = other_phys['p_center'].numpy()
                if np.linalg.norm(ego_pos - other_pos) <= self.r_sense:
                    neighbors_detected.append(other_phys)

            # 3. Instantiate custom CBF evaluator for the perceived neighborhood
            u_min = ego_ctrl.planner.u_min
            u_max = ego_ctrl.planner.u_max
            cbf_handler = AgentAgentCBF(
                robot_type=self.robot_type,
                neighbors_list=neighbors_detected,
                d_safe=self.d_safe,
                k1=self.k1,
                k2=self.k2,
                dt=self.dt,
                u_min=u_min,
                u_max=u_max,
                device=self.device
            )

            # 4. Inject handlers into the ShieldedMPPI planner
            ego_ctrl.planner.cbf_h_function = cbf_handler.h_function
            ego_ctrl.planner.safe_control_function = cbf_handler.safe_control

            # 5. Execute the optimized/shielded planning call
            u_cmd = ego_ctrl.get_command()
            return ego_key, u_cmd

        # Optimization: Execute sequentially if single-agent, otherwise launch threads!
        if self.num_agents == 1:
            for ego_key, ego_ctrl in self.agents_controllers.items():
                k, cmd = plan_agent(ego_key, ego_ctrl)
                if cmd is not None:
                    commands[k] = cmd
        else:
            # Launch decentralized planners concurrently across workers!
            with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
                futures = [
                    executor.submit(plan_agent, ego_key, ego_ctrl)
                    for ego_key, ego_ctrl in self.agents_controllers.items()
                ]
                for future in futures:
                    ego_key, u_cmd = future.result()
                    if u_cmd is not None:
                        commands[ego_key] = u_cmd

        return commands

    def visualize_rollouts(self, ax, draw_samples=False):
        """
        Delegated visualization function. Loops through all decentralized agents,
        assigning unique aesthetic colors to their nominal predictions.
        """
        # Curated list of distinct colors for multi-agent plotting
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        
        for i, (key, ctrl) in enumerate(self.agents_controllers.items()):
            agent_color = colors[i % len(colors)]
            # Draw the nominal MPC rollout for each agent in their designated color
            ctrl.visualize_rollouts(ax, color=agent_color, draw_samples=draw_samples)


if __name__ == "__main__":
    from agents.dubins_robot import DubinsRobot
    from agents.basic_robot import RobotParams
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    # 1. Configure Self-Contained Sim Parameters
    robot_params = RobotParams(
        action_dim=2,
        state_dim=3,
        action_min=[0.0, -4.0],
        action_max=[6.0, 4.0],
        state_min=[0.0, 0.0, 0.0],
        state_max=[50.0, 35.0, 6.28],
        dt=0.1,
        robot_type="dubins2d"
    )

    num_agents = 2
    goal_thresh = 1.0
    device = 'cpu'
    dtype = torch.float32
    dt = 0.1

    # 2. Symmetrical Starting Locations & Heading for Head-on Collision Scenario
    # Agent 0 -> starts on left, goes right
    # Agent 1 -> starts on right, goes left
    state0 = np.array([5.0, 15.0, 0.0])
    state1 = np.array([35.0, 15.0, np.pi])

    # Goals mapping
    goals = {
        "agent_0": np.array([35.0, 15.0]),
        "agent_1": np.array([5.0, 15.0])
    }

    # 3. Instantiate Decentralized Multi-Agent Shielded MPPI Controller
    mppi_params = MPPIControlParams(num_samples=120, horizon=15, lambda_=1.2)
    multi_ctrl = MultiShieldedMPPIControl(
        num_agents=num_agents,
        robot_params=robot_params,
        robot_type="dubins2d",
        goal_thresh=goal_thresh,
        device=device,
        mppi_params=mppi_params,
        dt=dt,
        r_sense=12.0,
        d_safe=1.5,  # Minimum exclusion zone (distance between centers)
        k1=2.5,
        k2=2.5
    )
    multi_ctrl.set_goals(goals)

    # Add an empty risk map to populate cost evaluator structures
    H, W = 35, 50
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing='xy'), axis=-1).reshape(-1, 2)
    dummy_map = np.zeros(len(coords), dtype=np.float32)
    multi_ctrl.set_maps(deque([(coords, dummy_map)] * 15, maxlen=15))

    # 4. Standalone Physical Simulators
    sim0 = DubinsRobot(robot_params)
    sim1 = DubinsRobot(robot_params)
    sim0.reset(state0)
    sim1.reset(state1)

    # 5. Setup Matplotlib Live Window
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 40)
    ax.set_ylim(5, 25)
    ax.grid(True)
    ax.set_title("MultiShieldedMPPIControl Standalone Unit Test: Head-On Collision Avoidance")

    # Render Goal posts (permanent)
    ax.scatter(goals["agent_0"][0], goals["agent_0"][1], color="blue", marker="x", s=100, label="Goal 0")
    ax.scatter(goals["agent_1"][0], goals["agent_1"][1], color="red", marker="x", s=100, label="Goal 1")
    ax.legend()

    hist0, hist1 = [], []

    print("\n🚀 Executing Decentralized Collision Avoidance Test Loop (200 Steps)...")
    
    # 6. Main Simulation Iterations
    for _ in tqdm(range(200)):
        # Read Simulator ground truths
        curr0 = sim0.get_state()
        curr1 = sim1.get_state()
        
        hist0.append(curr0[:2].copy())
        hist1.append(curr1[:2].copy())

        # Pack into composite agent observations dictionary exactly like SmokeEnv
        obs = {
            "agent_0": {"location": curr0[:2].copy(), "angle": float(curr0[2])},
            "agent_1": {"location": curr1[:2].copy(), "angle": float(curr1[2])}
        }

        # Query decentralized multi-agent command orchestrator
        actions_dict = multi_ctrl.get_commands(obs)

        # Format actions to Numpy commands for physics simulator
        u0 = actions_dict["agent_0"].detach().cpu().numpy()
        u1 = actions_dict["agent_1"].detach().cpu().numpy()

        # Step standalone physics engines
        sim0.dynamic_step(u0)
        sim1.dynamic_step(u1)

        # --- Update Render Frame ---
        # Remove previous frame dynamic lines and collections
        for artist in list(ax.lines) + list(ax.collections[2:]):
            artist.remove()

        pts0 = np.array(hist0)
        pts1 = np.array(hist1)
        
        # Re-draw history paths
        ax.plot(pts0[:, 0], pts0[:, 1], color="blue", linestyle='-', alpha=0.8)
        ax.plot(pts1[:, 0], pts1[:, 1], color="red", linestyle='-', alpha=0.8)

        # Re-draw physical body envelopes (d_safe circles)
        env0 = plt.Circle((curr0[0], curr0[1]), radius=1.5/2, color='blue', fill=False, alpha=0.6, linestyle='--')
        env1 = plt.Circle((curr1[0], curr1[1]), radius=1.5/2, color='red', fill=False, alpha=0.6, linestyle='--')
        ax.add_patch(env0)
        ax.add_patch(env1)

        # Draw current vehicle dots
        ax.scatter(curr0[0], curr0[1], color="blue", s=60)
        ax.scatter(curr1[0], curr1[1], color="red", s=60)

        plt.draw()
        plt.pause(0.01)

    print("\n✅ Multi-Agent Unit Test loop finalized.")

