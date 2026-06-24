import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from types import SimpleNamespace
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import yaml

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prefect import flow, task
from schema import BenchmarkConfig

from src.agents.basic_robot import RobotParams
from src.env.smoke_env import EnvConfig as LegacyEnvConfig
from src.env.smoke_env import SmokeEnv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark")


def _make_mppi_params(cfg_agent, cfg_mppi, device: str) -> "MPPIParams":
    from src.controllers.base.mppi import MPPIParams

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

    return MPPIParams(
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


def instantiate_controller(
    name: str,
    num_agents: int,
    goal_radius: float,
    clock: float,
    cfg_agent,
    cfg_controller,
    robot_params,
):
    device = cfg_controller.mppi.device

    if name == "nominal":
        from src.controllers.base_multi_agent import BaseMultiAgentController

        mppi_params = _make_mppi_params(cfg_agent, cfg_controller.mppi, device)
        return BaseMultiAgentController(
            num_agents=num_agents,
            robot_params=robot_params,
            mppi_params=mppi_params,
            goal_thresh=goal_radius,
            device=device,
            dt=clock,
        )

    elif name.startswith("cbf"):
        from src.controllers.base.cbf_safety import CBFFilterParams
        from src.controllers.multi_agent_cbf import MultiAgentCBFController

        mppi_params = _make_mppi_params(cfg_agent, cfg_controller.mppi, device)
        cbf_params = CBFFilterParams(
            d_safe=cfg_controller.safety.d_safe,
            k1=cfg_controller.k1,
            k2=cfg_controller.k2,
            dt=cfg_controller.dt,
            r_sense=cfg_controller.safety.r_sense,
            rho=cfg_controller.rho,
            L=cfg_controller.L,
        )
        return MultiAgentCBFController(
            num_agents=num_agents,
            robot_params=robot_params,
            mppi_params=mppi_params,
            cbf_params=cbf_params,
            cbf_mode=cfg_controller.mode,
            goal_thresh=goal_radius,
            device=device,
            dt=clock,
        )

    elif name.startswith("hj"):
        from src.controllers.base.hj import HJSolverConfig
        from src.controllers.base.hj_safety import HJFilterParams
        from src.controllers.multi_agent_hj import MultiAgentHJController

        mppi_params = _make_mppi_params(cfg_agent, cfg_controller.mppi, device)
        hj_params = HJFilterParams(
            d_safe=cfg_controller.safety.d_safe,
            safe_margin=cfg_controller.solver.safe_margin,
            dt=cfg_controller.dt,
            r_sense=cfg_controller.safety.r_sense,
            action_min=torch.tensor(cfg_controller.action_min, device=device),
            action_max=torch.tensor(cfg_controller.action_max, device=device),
        )
        sol_cfg = HJSolverConfig(
            domain_cells=np.array(cfg_controller.solver.domain_cells),
            domain=np.array(cfg_controller.solver.domain),
            accuracy=cfg_controller.solver.accuracy,
            superlevel_set_epsilon=cfg_controller.solver.safe_margin,
        )
        controller = MultiAgentHJController(
            num_agents=num_agents,
            robot_params=robot_params,
            mppi_params=mppi_params,
            hj_params=hj_params,
            hj_config=sol_cfg,
            hj_mode=cfg_controller.mode,
            goal_thresh=goal_radius,
            device=device,
            dt=clock,
        )

        if cfg_controller.mode != "online_rollout":
            logger.info(f"Precomputing HJI Relative value function for {name}...")
            start_hj = time.time()
            controller.solve_relative(
                time=0.0,
                target_time=cfg_controller.solver.target_time,
                dt=cfg_controller.solver.dt,
                epsilon=cfg_controller.solver.epsilon,
            )
            logger.info(f"JAX HJI precomputation finished in {time.time() - start_hj:.2f}s")

        return controller

    else:
        raise ValueError(f"Unknown controller: {name}")


@task(name="Evaluate Safety Controller Task")
def evaluate_controller_task(
    name: str,
    episodes: int,
    device_override: Optional[str],
    output_dir: str,
    steps: Optional[int] = None,
    render_mode: str = "none",
    test_mode: bool = False,
):
    logger.info(f"--- EVALUATING CONTROLLER: {name.upper()} ---")

    config_path = os.path.join(os.path.dirname(__file__), "benchmark_config.yaml")
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    benchmark_cfg = BenchmarkConfig(**yaml_data)
    if name not in benchmark_cfg.controllers:
        raise ValueError(f"Controller {name} not found in benchmark_config.yaml")

    cfg_controller = benchmark_cfg.controllers[name]
    cfg_env = benchmark_cfg.env
    cfg_agent = benchmark_cfg.agent
    cfg_playback = benchmark_cfg.playback

    num_agents = cfg_env.num_agents
    x_size = cfg_env.world_x_size
    y_size = cfg_env.world_y_size
    dt = cfg_env.clock
    max_steps = cfg_env.max_steps
    collision_radius = cfg_env.collision_radius
    goal_radius = cfg_env.goal_radius
    mppi_horizon = cfg_controller.mppi.horizon if name != "nominal" else 14

    initial_positions = cfg_env.initial_locations
    goal_locations = cfg_env.goal_locations

    initial_headings = []
    for start, goal in zip(initial_positions, goal_locations):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        initial_headings.append(np.arctan2(dy, dx))

    device = cfg_controller.mppi.device if device_override is None else device_override
    robot_params = RobotParams(
        name="dubins2d",
        action_dim=cfg_agent.action_dim,
        state_dim=cfg_agent.state_dim,
        action_max=cfg_agent.action_max,
        action_min=cfg_agent.action_min,
        state_max=[x_size, y_size, 2.0 * np.pi],
        state_min=[0.0, 0.0, 0.0],
        dt=dt,
        device=device,
    )

    controller = instantiate_controller(
        name, num_agents, goal_radius, dt, cfg_agent, cfg_controller, robot_params
    )

    goals_dict = {f"agent_{i}": np.array(goal_locations[i]) for i in range(num_agents)}
    controller.set_goals(goals_dict)

    effective_render = render_mode if render_mode != "none" else cfg_env.render
    if test_mode:
        effective_render = "human"

    env_cfg_obj = LegacyEnvConfig(
        world_x_size=x_size,
        world_y_size=y_size,
        max_steps=max_steps,
        clock=dt,
        render=effective_render,
        render_save_every=cfg_env.render_save_every,
        goal_radius=goal_radius,
        num_agents=num_agents,
        collision_radius=collision_radius,
        terminate_on_collision=cfg_env.terminate_on_collision,
        collision_penalty=cfg_env.collision_penalty,
        smoke_density_threshold=cfg_env.smoke_density_threshold,
        initial_locations=initial_positions,
        goal_locations=goal_locations,
        save_transitions=cfg_env.save_transitions,
    )

    from src.env.simulator.playback import PlaybackParams

    simulator_params = PlaybackParams(data_path=cfg_playback.data_path)
    logger.info(f"Simulator mode: PLAYBACK (data_path={cfg_playback.data_path})")

    sensor_params = SimpleNamespace(
        sensor_type=benchmark_cfg.sensor.sensor_type,
        density_reading_per_unit_length=benchmark_cfg.sensor.density_reading_per_unit_length,
        world_x_size=x_size,
        world_y_size=y_size,
    )

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
        controller.set_maps(
            deque([(dummy_coords, dummy_smoke)] * mppi_horizon, maxlen=mppi_horizon)
        )
    try:
        _ = controller.get_commands(dummy_obs)
        logger.info(f"Warm-up for '{name}' completed successfully.")
    except Exception as e:
        logger.warning(f"Warm-up step failed or skipped: {e}. Benchmarking will continue.")

    success_list = []
    collision_list = []
    reached_drones_list = []
    collided_drones_list = []
    steps_to_goal_list = []
    min_separation_list = []
    smoothness_list = []
    smoke_exposure_list = []
    planning_latencies = []

    env = SmokeEnv(
        env_params=env_cfg_obj,
        robot_params=robot_params,
        sensor_params=sensor_params,
        simulator_params=simulator_params,
    )

    for ep_idx in range(episodes):
        logger.info(f"[{name}] Running Episode {ep_idx + 1}/{episodes}...")
        env.save_transitions_path = os.path.join(output_dir, "transitions", name, f"ep_{ep_idx}")

        initial_state_list = []
        for i in range(num_agents):
            initial_state_list.append(
                {
                    "location": np.array(initial_positions[i], dtype=np.float32),
                    "angle": np.array([initial_headings[i]], dtype=np.float32),
                }
            )

        obs, _ = env.reset(initial_state=initial_state_list, seed=ep_idx)

        step_latencies = []
        smoke_densities_ep = []
        collision_occurred = False
        drone_collided_flags = [False] * num_agents
        drone_reached_flags = [False] * num_agents
        steps_to_reach = [max_steps] * num_agents
        min_separation_ep = 1000.0
        omega_sq_sum_ep = 0.0
        omega_count_ep = 0

        episode_trajectories = {
            f"agent_{i}": {"x": [], "y": [], "theta": [], "smoke": []} for i in range(num_agents)
        }
        gif_frames = []

        finished = False
        t = 0

        max_steps_limit = steps if steps is not None else max_steps
        while not finished and t < max_steps_limit:
            first_agent_obs = obs.get("agent_0", list(obs.values())[0])
            if "smoke_density" in first_agent_obs and len(first_agent_obs["smoke_density"]) > 0:
                smoke_density = first_agent_obs["smoke_density"]
                coords_density = first_agent_obs["smoke_density_location"]

                if torch.is_tensor(smoke_density):
                    smoke_map_flat = smoke_density.detach().cpu().numpy()
                else:
                    smoke_map_flat = np.asarray(smoke_density)
                smoke_map_flat = smoke_map_flat.reshape(-1).astype(np.float32)

                if torch.is_tensor(coords_density):
                    coords_flat = coords_density.detach().cpu().numpy()
                else:
                    coords_flat = np.asarray(coords_density)
                coords_flat = coords_flat.reshape(-1, 2).astype(np.float32)

                controller.set_maps(
                    deque([(coords_flat, smoke_map_flat)] * mppi_horizon, maxlen=mppi_horizon)
                )

            start_plan = time.time()
            commands_dict = controller.get_commands(obs)
            step_latencies.append((time.time() - start_plan) * 1000.0)

            step_actions = {}
            for k, cmd in commands_dict.items():
                if torch.is_tensor(cmd):
                    step_actions[k] = cmd.detach().cpu().numpy()
                else:
                    step_actions[k] = np.asarray(cmd)

            obs, reward, terminated, truncated, _ = env.step(step_actions)

            positions = [obs[f"agent_{i}"]["location"] for i in range(num_agents)]

            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < (2.0 * collision_radius):
                        collision_occurred = True
                        drone_collided_flags[i] = True
                        drone_collided_flags[j] = True
                    if dist < min_separation_ep:
                        min_separation_ep = dist

            for i in range(num_agents):
                dist = np.linalg.norm(positions[i] - goal_locations[i])
                if dist < goal_radius:
                    drone_reached_flags[i] = True
                    if steps_to_reach[i] == max_steps:
                        steps_to_reach[i] = t

            for i in range(num_agents):
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
        env._transition_buffer = [] if env.env_params.save_transitions else None

        traj_dir = os.path.join(output_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, f"{name}_ep_{ep_idx + 1}_trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(episode_trajectories, f, indent=2)

        if not test_mode and effective_render == "rgb_array" and gif_frames:
            videos_dir = os.path.join(output_dir, "videos")
            os.makedirs(videos_dir, exist_ok=True)
            gif_path = os.path.join(videos_dir, f"{name}_ep_{ep_idx + 1}_crossing.gif")
            logger.info(f"Saving playback GIF to {gif_path}...")
            imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)

        success = all(drone_reached_flags) and not collision_occurred
        success_list.append(1.0 if success else 0.0)
        collision_list.append(1.0 if collision_occurred else 0.0)
        reached_drones_list.append(sum(1 for flag in drone_reached_flags if flag))
        collided_drones_list.append(sum(1 for flag in drone_collided_flags if flag))
        steps_to_goal_list.append(np.mean(steps_to_reach))
        min_separation_list.append(min_separation_ep)
        smoothness_list.append(omega_sq_sum_ep / max(1, omega_count_ep))
        smoke_exposure_list.extend(smoke_densities_ep)
        planning_latencies.append(np.mean(step_latencies))

        logger.info(
            f"  Ep {ep_idx + 1} Result: {'SUCCESS' if success else 'FAILED'} | "
            f"Avg Steps to Goal: {np.mean(steps_to_reach):.1f} | Reached: {sum(1 for flag in drone_reached_flags if flag)}/6 | "
            f"Collided Drones: {sum(1 for flag in drone_collided_flags if flag)}/6 | Latency: {np.mean(step_latencies):.2f} ms"
        )

    aggregated = {
        "Success Rate (%)": np.mean(success_list) * 100.0,
        "Collision Rate (%)": np.mean(collision_list) * 100.0,
        "Avg Reached Drones": np.mean(reached_drones_list),
        "Avg Collided Drones": np.mean(collided_drones_list),
        "Avg Steps to Goal": np.mean(steps_to_goal_list),
        "Smoke Q1": float(np.percentile(smoke_exposure_list, 25)) if smoke_exposure_list else 0.0,
        "Smoke Median": float(np.percentile(smoke_exposure_list, 50))
        if smoke_exposure_list
        else 0.0,
        "Smoke Q3": float(np.percentile(smoke_exposure_list, 75)) if smoke_exposure_list else 0.0,
        "Smoke Max (Peak)": float(np.max(smoke_exposure_list)) if smoke_exposure_list else 0.0,
        "Min Separation (m)": np.mean(min_separation_list),
        "Control Smoothness": np.mean(smoothness_list),
        "Avg Planning Latency (ms)": np.mean(planning_latencies),
    }

    env.close()

    csv_file = os.path.join(output_dir, f"results_{name}.csv")
    new_row = {"Controller": name, **aggregated}
    df = pd.DataFrame([new_row])
    df.to_csv(csv_file, index=False)

    logger.info(f"Incremental metrics for {name} saved successfully to {csv_file}!")
    return aggregated


@flow(name="Decentralized Controllers Swarm Benchmark")
def benchmark_flow(
    episodes: int,
    device: Optional[str],
    output_dir: str,
    selected_controllers: Optional[List[str]],
    steps: Optional[int] = None,
    render_mode: str = "none",
    test_mode: bool = False,
):
    logger.info(
        f"Running Controller Benchmark Flow over {episodes} episodes (steps limit: {steps})."
    )
    logger.info(f"Device: {device or 'default'} | Output folder: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(os.path.dirname(__file__), "benchmark_config.yaml")
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    benchmark_cfg = BenchmarkConfig(**yaml_data)

    all_controllers = list(benchmark_cfg.controllers.keys())

    if selected_controllers:
        controllers_to_run = [c for c in selected_controllers if c in all_controllers]
    else:
        controllers_to_run = all_controllers

    results = {}
    for name in controllers_to_run:
        results[name] = evaluate_controller_task(
            name=name,
            episodes=episodes,
            device_override=device,
            output_dir=output_dir,
            steps=steps,
            render_mode=render_mode,
            test_mode=test_mode,
        )

    logger.info("Controller Benchmark Flow Completed Successfully!")
    return results


def main():
    parser = argparse.ArgumentParser(description="Prefect decentralized controllers benchmark.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes (default: 10)")
    parser.add_argument(
        "--steps", type=int, default=None, help="Limit maximum steps per episode for testing"
    )
    parser.add_argument("--device", type=str, default=None, help="Planning device (cpu, cuda, mps)")
    parser.add_argument(
        "--controllers", type=str, nargs="+", default=None, help="Specific controllers to benchmark"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--render",
        type=str,
        default="none",
        choices=["none", "rgb_array", "human"],
        help="Rendering mode: 'none' (no rendering), 'rgb_array' (save GIFs), 'human' (live window)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode: visualize 1 episode per controller in 'human' mode, no saving",
    )
    args = parser.parse_args()

    if args.test:
        args.episodes = 1

    if args.output_dir is None:
        import datetime

        project_root = "/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        args.output_dir = os.path.join(project_root, "outputs", "benchmark", timestamp)
    else:
        args.output_dir = os.path.abspath(args.output_dir)

    benchmark_flow(
        episodes=args.episodes,
        device=args.device,
        output_dir=args.output_dir,
        selected_controllers=args.controllers,
        steps=args.steps,
        render_mode=args.render,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()
