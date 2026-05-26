import io
import logging
from collections import deque
from typing import Any, Dict

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from agents.basic_robot import RobotParams
from controllers.mppi_ctrl import MPPICtrlParams
from controllers.multi_dual_guard_cbf_ctrl import MultiDualGuardCBFCtrl
from envs.smoke_env import SmokeEnv
from experiments.base_experiment import BaseExperiment

log = logging.getLogger(__name__)


class MultiShieldedMPPIExperiment(BaseExperiment):
    """Experiment class for Decentralized Multi-Agent Safe Navigation using Shielded MPPI.
    Incorporates decentralized analytical HOCBF safety exclusion zones inside dynamic environments.
    """

    def __init__(self, cfg: DictConfig, risk_aware: bool = True):
        super().__init__(cfg)
        self.risk_aware = risk_aware
        self.rgb_frames = []

    # ======================================================
    #  1. INITIALIZATION AND SETUP
    # ======================================================
    def setup(self):
        log.info("Setting up Multi-Agent Shielded MPPI Experiment...")

        # 1. Process Robot Configuration
        self.robot_params = RobotParams()
        # In case of overriding bounds from configuration:
        self.robot_params.world_x_size = self.cfg.env.get("world_x_size", 100.0)
        self.robot_params.world_y_size = self.cfg.env.get("world_y_size", 100.0)
        self.robot_params.action_min = np.array(self.cfg.agent.action_min)
        self.robot_params.action_max = np.array(self.cfg.agent.action_max)
        self.robot_params.dt = self.cfg.agent.dt

        # 2. Initialize Multi-Agent Environment using the CORRECT and NEW unified configuration schema
        log.info(
            f"Instantiating Multi-Agent SmokeEnv (num_agents={self.cfg.env.get('num_agents', 1)})"
        )
        self.env = SmokeEnv(cfg=self.cfg.env, robot_params=self.robot_params)

        # Sync dynamic properties if Playback loaded external sizes
        self.robot_params.state_max[0] = self.env.env_params.world_x_size
        self.robot_params.state_max[1] = self.env.env_params.world_y_size

        # 3. Multi-Agent Reset with explicit Initial Locations Injection
        num_agents = self.env.env_params.num_agents
        initial_state_dict = None

        # Retrieve position overrides from environment config
        raw_init_locs = self.cfg.env.get("initial_locations", None)

        if raw_init_locs is not None:
            import ast

            # Gracefully decode command-line string lists like '[[1.0,2.0],...]'
            if isinstance(raw_init_locs, str):
                try:
                    raw_init_locs = ast.literal_eval(raw_init_locs)
                except Exception as e:
                    log.warning(f"Failed to decode initial_locations override string: {e}")
            else:
                # Convert from OmegaConf ListConfig to native Python list/objects
                raw_init_locs = OmegaConf.to_object(raw_init_locs)

            if isinstance(raw_init_locs, (list, tuple)) and len(raw_init_locs) >= num_agents:
                initial_state_dict = {}
                for i in range(num_agents):
                    xy = raw_init_locs[i]
                    goal = self.env.agents[i].goal_location

                    # Autocalculate nominal heading pointing directly toward the goal
                    dx = float(goal[0] - xy[0])
                    dy = float(goal[1] - xy[1])
                    theta_init = np.arctan2(dy, dx)

                    initial_state_dict[f"agent_{i}"] = {
                        "location": np.array(xy, dtype=np.float32),
                        "angle": np.array([theta_init], dtype=np.float32),
                    }
                log.info(f"Successfully configured {num_agents} initial agent overrides.")

        # Run Gymnasium reset applying custom coordinates if available
        episode_idx = self.cfg.experiment.get("episode_idx", None)
        self.state, _ = self.env.reset(initial_state=initial_state_dict, seed=episode_idx)

        # 4. Initialize the Decentralized Shielded MPPI Control Orchestrator
        horizon = self.cfg.experiment.time_horizon
        mppi_params = MPPICtrlParams(
            horizon=horizon,
            num_samples=self.cfg.experiment.get("samples", 120),
            device=self.cfg.experiment.get("device", "cpu"),
            lambda_=self.cfg.experiment.get("lambda_", 1.2),
        )

        # Retrieve local perception parameters from configuration, providing robust defaults
        r_sense = self.cfg.experiment.get("r_sense", 6.0)
        d_safe = self.cfg.experiment.get("d_safe", 1.2)
        k1 = self.cfg.experiment.get("cbf_k1", 2.5)
        k2 = self.cfg.experiment.get("cbf_k2", 2.5)

        log.info(f"Initializing MultiDualGuardCBFCtrl | r_sense={r_sense} | d_safe={d_safe}")
        self.controller = MultiDualGuardCBFCtrl(
            num_agents=self.env.env_params.num_agents,
            robot_params=self.robot_params,
            robot_type=self.env.robot_params.robot_type,
            goal_thresh=self.env.env_params.goal_radius,
            device=self.cfg.experiment.get("device", "cpu"),
            mppi_params=mppi_params,
            dt=self.robot_params.dt,
            r_sense=r_sense,
            d_safe=d_safe,
            k1=k1,
            k2=k2,
        )

        # Assign corresponding individual agent goals
        goals = {
            f"agent_{i}": np.array(self.env.agents[i].goal_location)
            for i in range(self.env.env_params.num_agents)
        }
        self.controller.set_goals(goals)

        # Disable default single-agent renderer to prevent KeyError: 'location'
        self.renderer = None
        self.rgb_frames = []

        # If render_mode is rgb_array, setup Agg backend to run headless
        if self.cfg.env.render == "rgb_array":
            plt.switch_backend("Agg")

        log.info(
            f"MultiShieldedMPPIExperiment READY | risk_aware={self.risk_aware} | "
            f"render={self.cfg.env.render}"
        )

    # ======================================================
    #  2. CORE SIMULATION LOOP
    # ======================================================
    def run_episode(self) -> Dict[str, Any]:
        finished = False
        horizon = self.cfg.experiment.time_horizon
        max_steps = self.env.env_params.max_steps
        render_mode = self.cfg.env.render
        render_save_every = self.cfg.env.get("render_save_every", 1)

        log.info(f"Starting Multi-Agent Episode Execution (Max Steps: {max_steps})...")

        for t in tqdm(range(max_steps + 1)):
            if finished:
                break

            # 1. Incorporate Environmental Risk Maps (if active)
            if self.risk_aware:
                # Extract spatial risk map from first agent's global reading
                # Note: Centralized map logic assumes shared environmental perception
                first_agent_obs = self.state.get("agent_0", list(self.state.values())[0])

                # If obs format is used:
                if "smoke_density" in first_agent_obs and len(first_agent_obs["smoke_density"]) > 0:
                    smoke_map_flat = first_agent_obs["smoke_density"].squeeze().astype(np.float32)
                    coords_flat = first_agent_obs["smoke_density_location"]

                    # Set map deque to all agents
                    self.controller.set_maps(
                        deque([(coords_flat, smoke_map_flat)] * horizon, maxlen=horizon)
                    )

            # 2. Decoupled Safe Multi-Agent Planning (HOCBF Shielded MPPI)
            with self.time_tracker.track("control"):
                # Fetch optimal actions dictionary (contains PyTorch tensors)
                commands_dict = self.controller.get_commands(self.state)

                # Format actions into CPU numpy arrays for SmokeEnv interface
                step_actions = {}
                for key, cmd in commands_dict.items():
                    step_actions[key] = cmd.detach().cpu().numpy()

            # 3. Synchronized Multi-Agent Environment Integration
            with self.time_tracker.track("env"):
                next_state, reward, terminated, truncated, _ = self.env.step(step_actions)

            # 4. Log Decentralized Metrics
            self.log_multi_agent_metrics(self.state, step_actions, reward, t)

            # 5. Global Evaluation and Episode Stepping
            # Episode finishes globally when ALL agents are terminated, or ANY is truncated (timeout)
            all_term = all(terminated.values()) if isinstance(terminated, dict) else terminated
            any_trunc = any(truncated.values()) if isinstance(truncated, dict) else truncated
            mean_reward = np.mean(list(reward.values())) if isinstance(reward, dict) else reward

            # We feed the collective termination variables into parent class logic
            finished = self.check_termination(self.state, mean_reward, all_term, any_trunc)

            # 6. Custom Decentralized Multi-Agent Rendering
            if render_mode and t % render_save_every == 0:
                # In multi-agent, the environment naturally manages rendering multiple agents natively!
                # Pass the composite controller to render nominal paths for ALL agents!
                frame = self.env._render_frame(controller=self.controller)

                # If human render: update window canvas
                if render_mode == "human" and self.env.window.get("fig") is not None:
                    fig = self.env.window["fig"]
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(self.env.env_params.clock)

                # If rgb_array: save current Matplotlib figure into internal memory list
                elif render_mode == "rgb_array" and self.env.window.get("fig") is not None:
                    buf = io.BytesIO()
                    self.env.window["fig"].savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    self.rgb_frames.append(imageio.imread(buf))

            self.state = next_state

        return {"status": "completed"}

    # ======================================================
    #  3. TEARDOWN AND EXPORT
    # ======================================================
    def teardown(self):
        """Extends parent teardown to export custom multi-agent visualizations."""
        # Save captured RGB frames to GIF if active
        if self.cfg.env.render == "rgb_array" and self.rgb_frames:
            gif_path = self.output_dir / "playback.gif"
            log.info(f"Saving multi-agent playback GIF to {gif_path}...")
            try:
                imageio.mimsave(
                    str(gif_path),
                    self.rgb_frames,
                    fps=int(1.0 / self.env.env_params.clock / 3.0),
                    loop=0,
                )
                log.info(
                    f"Multi-agent playback successfully saved ({len(self.rgb_frames)} frames)."
                )
            except Exception as e:
                log.error(f"Failed to write playback GIF: {e}")

        # Clean up any remaining matplotlib figures
        plt.close("all")

        # Invoke generic parent teardown (logs CSV and csv time metrics)
        try:
            super().teardown()
        except Exception as e:
            log.warning(f"Suppressed timing extraction warning from base teardown: {e}")
            # Fallback cleanup in case base teardown partially failed
            if self.env:
                self.env.close()

    # ======================================================
    #  4. CUSTOM LOGGING & METRICS OVERRIDES
    # ======================================================
    def log_multi_agent_metrics(
        self, states: Dict[str, Any], actions: Dict[str, Any], rewards: Dict[str, Any], t: int
    ):
        """Decentralized metrics logger that safely collects positions, deviations,
        risks and inputs for each agent independently.
        """
        # Loop through all active agents
        for key, agent_state in states.items():
            if key not in actions:
                continue

            agent_id = int(key.split("_")[1])
            pos = agent_state["location"]
            act = actions[key]

            # 1. Compute and Log Distance to corresponding individual Goal
            agent_obj = self.env.agents[agent_id]
            dist = np.linalg.norm(pos - np.array(agent_obj.goal_location))
            self.metrics.add_value(f"{key}_dist_to_goal", dist)

            # 2. Log Instantaneous Individual Reward
            r_val = rewards.get(key, 0.0) if isinstance(rewards, dict) else rewards
            self.metrics.add_value(f"{key}_reward", r_val)

            # 3. Log Action Vectors
            for i, u_val in enumerate(act):
                self.metrics.add_value(f"{key}_action_{i}", float(u_val))

            # 4. Log Positions
            self.metrics.add_value(f"{key}_pos_x", float(pos[0]))
            self.metrics.add_value(f"{key}_pos_y", float(pos[1]))

        # Log swarm-level proximity metric (Minimum inter-agent distance)
        positions = [agent_state["location"] for key, agent_state in states.items()]
        if len(positions) > 1:
            from scipy.spatial.distance import pdist

            min_sep = np.min(pdist(positions))
            self.metrics.add_value("swarm_min_separation", min_sep)

            # Real-time physical collision detection logger
            col_dist = float(self.env.env_params.collision_radius * 2.0)
            if min_sep < col_dist:
                log.warning(
                    f"⚠️  [COLLISION ALERT] Step {t}: Minimum drone separation is only "
                    f"{min_sep:.3f}m! (Physical threshold limit: {col_dist:.2f}m)"
                )

        # Log generic execution indexes
        self.metrics.add_value("steps", t)
        self.metrics.add_value("time", t * self.cfg.env.clock)
