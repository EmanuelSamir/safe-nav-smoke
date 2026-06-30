import json
import logging
import os
import time
from collections import deque
from typing import Optional, Dict, Any

import imageio.v2 as imageio
import numpy as np
import torch

from schema import BenchmarkConfig
from src.env.smoke_env import SmokeEnv
from metrics import BenchmarkTracker, EpisodeMetrics

logger = logging.getLogger("evaluator")

def to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)

def _make_mppi_params(cfg_agent, cfg_mppi, device: str):
    from src.controllers.base.schemas import MPPIConfig

    u_min = torch.tensor(cfg_agent.action_min, device=device, dtype=torch.float32)
    u_max = torch.tensor(cfg_agent.action_max, device=device, dtype=torch.float32)
    nu = cfg_agent.action_dim
    horizon = cfg_mppi.horizon

    alpha = float(getattr(cfg_mppi, "alpha_noise_sigma", 10.0))
    action_range = (u_max - u_min).cpu()
    noise_sigma = alpha * torch.diag(action_range).to(device)
    noise_abs_cost = bool(getattr(cfg_mppi, "noise_abs_cost", False))

    v_mid = ((u_min[0] + u_max[0]) / 2.0).item()
    u_init_vec = torch.zeros(nu, device=device)
    u_init_vec[0] = v_mid
    u_init = u_init_vec.unsqueeze(0)

    return MPPIConfig(
        nx=cfg_agent.state_dim,
        noise_sigma=noise_sigma,
        num_samples=cfg_mppi.num_samples,
        horizon=horizon,
        lambda_=cfg_mppi.lambda_,
        device=device,
        u_min=u_min,
        u_max=u_max,
        u_init=u_init,
        noise_abs_cost=noise_abs_cost,
        step_dependent_dynamics=True,
    )

def instantiate_controller(name: str, cfg_controller, robot_params, env_cfg):
    device = robot_params.device

    if name == "nominal":
        from src.controllers.base_multi_agent import BaseMultiAgentController
        from src.controllers.schemas import BaseMultiAgentConfig, SharedSafetyConfig
        mppi_params = _make_mppi_params(robot_params, cfg_controller.mppi, device)
        config = BaseMultiAgentConfig(
            safety=SharedSafetyConfig(),
            mppi=mppi_params,
            mode="filter",
            dt=env_cfg.clock,
        )
        return BaseMultiAgentController(
            num_agents=env_cfg.num_agents,
            robot_config=robot_params,
            config=config,
            goal_thresh=env_cfg.goal_radius,
        )

    elif name.startswith("cbf"):
        from src.controllers.multi_agent_cbf import MultiAgentCBFController
        from src.controllers.schemas import MultiAgentCBFConfig, SharedSafetyConfig
        mppi_params = _make_mppi_params(robot_params, cfg_controller.mppi, device)
        safety_kwargs = cfg_controller.safety.model_dump(exclude_unset=True) if getattr(cfg_controller, "safety", None) else {}
        config = MultiAgentCBFConfig(
            safety=SharedSafetyConfig(
                **safety_kwargs,
                robot_radius=env_cfg.collision_radius,
            ),
            mppi=mppi_params,
            **cfg_controller.model_dump(exclude_unset=True, include={"mode", "dt", "k1", "k2", "rho"}),
        )
        return MultiAgentCBFController(
            num_agents=env_cfg.num_agents,
            robot_config=robot_params,
            config=config,
            goal_thresh=env_cfg.goal_radius,
        )

    elif name.startswith("hj"):
        from src.controllers.base.schemas import RelativeHJSolverConfig
        from src.controllers.multi_agent_hj import MultiAgentHJController
        from src.controllers.schemas import MultiAgentHJConfig, SharedSafetyConfig
        mppi_params = _make_mppi_params(robot_params, cfg_controller.mppi, device)
        sol_kwargs = cfg_controller.solver.model_dump(exclude_unset=True) if getattr(cfg_controller, "solver", None) else {}
        sol_cfg = RelativeHJSolverConfig(**sol_kwargs)
        safety_kwargs = cfg_controller.safety.model_dump(exclude_unset=True) if getattr(cfg_controller, "safety", None) else {}
        
        config = MultiAgentHJConfig(
            safety=SharedSafetyConfig(
                **safety_kwargs,
                robot_radius=env_cfg.collision_radius,
            ),
            mppi=mppi_params,
            solver=sol_cfg,
            **cfg_controller.model_dump(exclude_unset=True, include={"mode", "control_type", "dt", "action_min", "action_max"}),
        )
        controller = MultiAgentHJController(
            num_agents=env_cfg.num_agents,
            robot_config=robot_params,
            config=config,
            goal_thresh=env_cfg.goal_radius,
        )
        if cfg_controller.mode != "online_dual-guard":
            logger.info(f"Precomputing HJI Relative value function for {name}...")
            start_hj = time.time()
            controller.solve_relative()
            logger.info(f"JAX HJI precomputation finished in {time.time() - start_hj:.2f}s")
        return controller

    else:
        raise ValueError(f"Unknown controller: {name}")

class Evaluator:
    def __init__(self, benchmark_cfg: BenchmarkConfig, output_dir: str, device_override: Optional[str] = None):
        self.cfg = benchmark_cfg
        self.output_dir = output_dir
        self.device = device_override or benchmark_cfg.run.device
        
    def evaluate_controller(self, name: str, episodes: int, steps_limit: Optional[int], render_mode: str, test_mode: bool):
        logger.info(f"--- EVALUATING CONTROLLER: {name.upper()} ---")
        self.tracker = BenchmarkTracker(self.output_dir, name)
        
        cfg_controller = self.cfg.controllers[name]
        cfg_env = self.cfg.env
        cfg_agent = self.cfg.agent
        cfg_playback = self.cfg.playback
        
        device = cfg_controller.mppi.device if self.device is None else self.device
        robot_params = cfg_agent.model_copy(update={"device": device})
        
        effective_render = render_mode if render_mode != "none" else cfg_env.render
        if test_mode:
            effective_render = "human"
            
        env_cfg_obj = cfg_env.model_copy(update={
            "render": effective_render,
            "remove_dead_agents": False
        })
        
        controller = instantiate_controller(name, cfg_controller, robot_params, env_cfg_obj)
        
        initial_positions = env_cfg_obj.initial_locations
        goal_locations = env_cfg_obj.goal_locations
        
        goals_dict = {f"agent_{i}": np.array(goal_locations[i]) for i in range(env_cfg_obj.num_agents)}
        controller.set_goals(goals_dict)
        
        initial_headings = []
        for start, goal in zip(initial_positions, goal_locations):
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            initial_headings.append(np.arctan2(dy, dx))
            
        from src.env.simulator.schemas import PlaybackConfig
        simulator_params = PlaybackConfig(data_path=cfg_playback.data_path)
        logger.info(f"Simulator mode: PLAYBACK (data_path={cfg_playback.data_path})")
        
        env = SmokeEnv(
            env_cfg=env_cfg_obj,
            robot_cfg=robot_params,
            sensor_cfg=self.cfg.sensor,
            simulator_cfg=simulator_params,
        )
        
        mppi_horizon = cfg_controller.mppi.horizon if name != "nominal" else 14
        self._warmup_controller(name, controller, initial_positions, initial_headings, mppi_horizon, env_cfg_obj.num_agents)
        
        for ep_idx in range(episodes):
            logger.info(f"[{name}] Running Episode {ep_idx + 1}/{episodes}...")
            env.save_transitions_path = os.path.join(self.output_dir, "transitions", name, f"ep_{ep_idx}")
            
            initial_state_list = [
                {
                    "location": np.array(initial_positions[i], dtype=np.float32),
                    "angle": np.array([initial_headings[i]], dtype=np.float32),
                } for i in range(env_cfg_obj.num_agents)
            ]
            
            obs, _ = env.reset(initial_state=initial_state_list, seed=ep_idx)
            
            metrics = self._run_single_episode(
                name, ep_idx, env, controller, obs, env_cfg_obj, mppi_horizon, goal_locations, 
                steps_limit, effective_render, test_mode
            )
            
            self.tracker.record_episode(metrics)
            
            logger.info(
                f"  Ep {ep_idx + 1} Result: {'SUCCESS' if metrics.success else 'FAILED'} | "
                f"Avg Steps to Goal: {metrics.avg_steps_to_goal:.1f} | Reached: {metrics.reached_drones}/{env_cfg_obj.num_agents} | "
                f"Collided Drones: {metrics.collided_drones}/{env_cfg_obj.num_agents} | Latency: {metrics.avg_planning_latency_ms:.2f} ms"
            )
            
        env.close()

    def _warmup_controller(self, name, controller, initial_positions, initial_headings, mppi_horizon, num_agents):
        logger.info(f"Warming up controller '{name}' to compile JIT/JAX trace...")
        dummy_obs = {}
        for i in range(num_agents):
            dummy_obs[f"agent_{i}"] = {
                "location": np.array(initial_positions[i], dtype=np.float32),
                "angle": np.array([initial_headings[i]], dtype=np.float32),
            }
        if name != "nominal":
            dummy_coords = np.zeros((100, 2), dtype=np.float32)
            dummy_smoke = np.zeros(100, dtype=np.float32)
            controller.set_maps(deque([(dummy_coords, dummy_smoke)] * mppi_horizon, maxlen=mppi_horizon))
        try:
            _ = controller.get_commands(dummy_obs)
            logger.info(f"Warm-up for '{name}' completed successfully.")
        except Exception as e:
            logger.warning(f"Warm-up step failed or skipped: {e}. Benchmarking will continue.")

    def _run_single_episode(self, name, ep_idx, env, controller, obs, env_cfg_obj, mppi_horizon, goal_locations, steps_limit, effective_render, test_mode) -> EpisodeMetrics:
        step_latencies = []
        smoke_densities_ep = []
        collision_occurred = False
        drone_collided_flags = [False] * env_cfg_obj.num_agents
        drone_reached_flags = [False] * env_cfg_obj.num_agents
        steps_to_reach = [env_cfg_obj.max_steps] * env_cfg_obj.num_agents
        min_separation_ep = 1000.0
        omega_sq_sum_ep = 0.0
        omega_count_ep = 0

        episode_trajectories = {
            f"agent_{i}": {"x": [], "y": [], "theta": [], "smoke": []} for i in range(env_cfg_obj.num_agents)
        }
        gif_frames = []

        finished = False
        t = 0
        max_steps_limit = steps_limit if steps_limit is not None else env_cfg_obj.max_steps
        
        while not finished and t < max_steps_limit:
            first_agent_obs = obs.get("agent_0", list(obs.values())[0])
            if "smoke_density" in first_agent_obs and len(first_agent_obs["smoke_density"]) > 0:
                smoke_map_flat = to_numpy(first_agent_obs["smoke_density"]).reshape(-1).astype(np.float32)
                coords_flat = to_numpy(first_agent_obs["smoke_density_location"]).reshape(-1, 2).astype(np.float32)
                controller.set_maps(deque([(coords_flat, smoke_map_flat)] * mppi_horizon, maxlen=mppi_horizon))

            start_plan = time.time()
            commands_dict = controller.get_commands(obs)
            step_latencies.append((time.time() - start_plan) * 1000.0)

            step_actions = {k: to_numpy(cmd) for k, cmd in commands_dict.items()}
            obs, reward, terminated, truncated, _ = env.step(step_actions)

            positions = [obs[f"agent_{i}"]["location"] for i in range(env_cfg_obj.num_agents)]

            for i in range(env_cfg_obj.num_agents):
                for j in range(i + 1, env_cfg_obj.num_agents):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < (2.0 * env_cfg_obj.collision_radius):
                        collision_occurred = True
                        drone_collided_flags[i] = True
                        drone_collided_flags[j] = True
                    if dist < min_separation_ep:
                        min_separation_ep = dist

            for i in range(env_cfg_obj.num_agents):
                dist = np.linalg.norm(positions[i] - goal_locations[i])
                if dist < env_cfg_obj.goal_radius:
                    drone_reached_flags[i] = True
                    if steps_to_reach[i] == env_cfg_obj.max_steps:
                        steps_to_reach[i] = t

            for i in range(env_cfg_obj.num_agents):
                act = step_actions.get(f"agent_{i}")
                if act is not None:
                    omega_sq_sum_ep += float(act[1]) ** 2
                    omega_count_ep += 1

                pos = positions[i]
                heading = float(obs[f"agent_{i}"]["angle"])
                smoke_val = float(np.ravel(env.get_smoke_density_in_robot(i))[0])
                episode_trajectories[f"agent_{i}"]["x"].append(float(pos[0]))
                episode_trajectories[f"agent_{i}"]["y"].append(float(pos[1]))
                episode_trajectories[f"agent_{i}"]["theta"].append(heading)
                episode_trajectories[f"agent_{i}"]["smoke"].append(smoke_val)
                smoke_densities_ep.append(smoke_val)

            if effective_render == "human" and t % 2 == 0:
                env.render(controller=controller)
            elif effective_render == "rgb_array" and t % 2 == 0:
                frame = env.render(controller=controller)
                if frame is not None:
                    gif_frames.append(frame)

            all_term = all(terminated.values()) if isinstance(terminated, dict) else terminated
            any_trunc = any(truncated.values()) if isinstance(truncated, dict) else truncated
            if all_term or any_trunc:
                finished = True

            t += 1

        env._save_transitions()
        env._transition_buffer = [] if env.env_cfg.save_transitions else None

        traj_dir = os.path.join(self.output_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, f"{name}_ep_{ep_idx + 1}_trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(episode_trajectories, f, indent=2)

        if not test_mode and effective_render == "rgb_array" and gif_frames:
            videos_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(videos_dir, exist_ok=True)
            gif_path = os.path.join(videos_dir, f"{name}_ep_{ep_idx + 1}_crossing.gif")
            imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)

        success = all(drone_reached_flags) and not collision_occurred
        
        return EpisodeMetrics(
            controller_name=name,
            episode_idx=ep_idx + 1,
            success=success,
            collision_occurred=collision_occurred,
            reached_drones=sum(1 for flag in drone_reached_flags if flag),
            collided_drones=sum(1 for flag in drone_collided_flags if flag),
            avg_steps_to_goal=float(np.mean(steps_to_reach)),
            min_separation=min_separation_ep,
            smoothness=omega_sq_sum_ep / max(1, omega_count_ep),
            avg_planning_latency_ms=float(np.mean(step_latencies)),
            smoke_q1=float(np.percentile(smoke_densities_ep, 25)) if smoke_densities_ep else 0.0,
            smoke_median=float(np.percentile(smoke_densities_ep, 50)) if smoke_densities_ep else 0.0,
            smoke_q3=float(np.percentile(smoke_densities_ep, 75)) if smoke_densities_ep else 0.0,
            smoke_max=float(np.max(smoke_densities_ep)) if smoke_densities_ep else 0.0,
        )
