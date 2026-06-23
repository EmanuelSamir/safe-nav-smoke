import argparse
import json
import logging
import os
import time
from collections import deque
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from prefect import flow, task

from src.agents.basic_robot import RobotParams
from src.env.smoke_env import EnvConfig, SmokeEnv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark")


def instantiate_controller(name: str, cfg, robot_params):
    """Dynamically instantiates the corresponding multi-agent controller from the Hydra config."""
    if name == "nominal":
        from controllers.base.mppi import MPPIParams
        from src.controllers.base_multi_agent import BaseMultiAgentController

        device = cfg.controller.mppi.device
        mppi_params = MPPIParams(
            nx=cfg.agent.state_dim,
            noise_sigma=0.5 * torch.eye(cfg.agent.action_dim, device=device),
            num_samples=cfg.controller.mppi.num_samples,
            horizon=cfg.controller.mppi.horizon,
            lambda_=cfg.controller.mppi.lambda_,
            device=device,
            u_min=torch.tensor(cfg.agent.action_min, device=device),
            u_max=torch.tensor(cfg.agent.action_max, device=device),
        )
        return BaseMultiAgentController(
            num_agents=cfg.env.num_agents,
            robot_params=robot_params,
            mppi_params=mppi_params,
            goal_thresh=cfg.env.goal_radius,
            device=device,
            dt=cfg.env.clock,
        )

    elif name.startswith("cbf"):
        from controllers.base.cbf_safety import CBFFilterParams
        from controllers.base.mppi import MPPIParams
        from src.controllers.multi_agent_cbf import MultiAgentCBFController

        device = cfg.controller.mppi.device
        mppi_params = MPPIParams(
            nx=cfg.agent.state_dim,
            noise_sigma=0.5 * torch.eye(cfg.agent.action_dim, device=device),
            num_samples=cfg.controller.mppi.num_samples,
            horizon=cfg.controller.mppi.horizon,
            lambda_=cfg.controller.mppi.lambda_,
            device=device,
            u_min=torch.tensor(cfg.agent.action_min, device=device),
            u_max=torch.tensor(cfg.agent.action_max, device=device),
        )
        cbf_params = CBFFilterParams(
            d_safe=cfg.controller.safety.d_safe,
            k1=cfg.controller.k1,
            k2=cfg.controller.k2,
            dt=cfg.controller.dt,
            r_sense=cfg.controller.safety.r_sense,
            rho=cfg.controller.rho,
            L=cfg.controller.L,
        )
        return MultiAgentCBFController(
            num_agents=cfg.env.num_agents,
            robot_params=robot_params,
            mppi_params=mppi_params,
            cbf_params=cbf_params,
            cbf_mode=cfg.controller.mode,
            goal_thresh=cfg.env.goal_radius,
            device=device,
            dt=cfg.env.clock,
        )

    elif name.startswith("hj"):
        from controllers.base.hj import HJSolverConfig
        from controllers.base.hj_safety import HJFilterParams
        from controllers.base.mppi import MPPIParams
        from src.controllers.multi_agent_hj import MultiAgentHJController

        device = cfg.controller.mppi.device
        mppi_params = MPPIParams(
            nx=cfg.agent.state_dim,
            noise_sigma=0.5 * torch.eye(cfg.agent.action_dim, device=device),
            num_samples=cfg.controller.mppi.num_samples,
            horizon=cfg.controller.mppi.horizon,
            lambda_=cfg.controller.mppi.lambda_,
            device=device,
            u_min=torch.tensor(cfg.agent.action_min, device=device),
            u_max=torch.tensor(cfg.agent.action_max, device=device),
        )
        hj_params = HJFilterParams(
            d_safe=cfg.controller.safety.d_safe,
            safe_margin=cfg.controller.solver.safe_margin,
            dt=cfg.controller.dt,
            r_sense=cfg.controller.safety.r_sense,
            action_min=torch.tensor(cfg.controller.action_min, device=device),
            action_max=torch.tensor(cfg.controller.action_max, device=device),
        )
        sol_cfg = HJSolverConfig(
            domain_cells=np.array(cfg.controller.solver.domain_cells),
            domain=np.array(cfg.controller.solver.domain),
            accuracy=cfg.controller.solver.accuracy,
            superlevel_set_epsilon=cfg.controller.solver.safe_margin,
        )
        controller = MultiAgentHJController(
            num_agents=cfg.env.num_agents,
            robot_params=robot_params,
            mppi_params=mppi_params,
            hj_params=hj_params,
            hj_config=sol_cfg,
            hj_mode=cfg.controller.mode,
            goal_thresh=cfg.env.goal_radius,
            device=device,
            dt=cfg.env.clock,
        )

        # Precompute HJI Relative values for non-online modes
        if cfg.controller.mode != "online_rollout":
            logger.info(f"Precomputing HJI Relative value function for {name}...")
            start_hj = time.time()
            controller.solve_relative(
                time=0.0,
                target_time=cfg.controller.solver.target_time,
                dt=cfg.controller.solver.dt,
                epsilon=cfg.controller.solver.epsilon,
            )
            logger.info(f"JAX HJI precomputation finished in {time.time() - start_hj:.2f}s")

        return controller

    else:
        raise ValueError(f"Unknown controller: {name}")


@task(name="Evaluate Safety Controller Task")
def evaluate_controller_task(
    name: str,
    config_override_name: str,
    episodes: int,
    device_override: Optional[str],
    output_dir: str,
    steps: Optional[int] = None,
):
    """Prefect task evaluating a single controller layout over a set of episodes."""
    logger.info(f"--- EVALUATING CONTROLLER: {name.upper()} ---")

    # Initialize Hydra inside the task context to compose the config
    with initialize(version_base=None, config_path="../../../configs"):
        overrides = [
            "experiment=multi/benchmark",
            f"+controller={config_override_name}"
            if config_override_name
            else "+controller=cbf/filter",
        ]
        if device_override:
            overrides.append(f"controller.mppi.device={device_override}")

        cfg = compose(config_name="config", overrides=overrides)

    # Resolve omegaconf values
    OmegaConf.resolve(cfg)

    num_agents = cfg.env.num_agents
    x_size = cfg.env.world_x_size
    y_size = cfg.env.world_y_size
    dt = cfg.env.clock
    max_steps = cfg.env.max_steps
    collision_radius = cfg.env.collision_radius
    goal_radius = cfg.env.goal_radius
    mppi_horizon = cfg.controller.mppi.horizon if name != "nominal" else 14

    # Symmetrical Crossing positions
    initial_positions = cfg.env.initial_locations
    goal_locations = cfg.env.goal_locations

    # Compute headings toward goals
    initial_headings = []
    for start, goal in zip(initial_positions, goal_locations):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        initial_headings.append(np.arctan2(dy, dx))

    # Instantiate Robot Params
    robot_params = RobotParams(
        name="dubins2d",
        action_dim=cfg.agent.action_dim,
        state_dim=cfg.agent.state_dim,
        action_max=cfg.agent.action_max,
        action_min=cfg.agent.action_min,
        state_max=[x_size, y_size, 2.0 * np.pi],
        state_min=[0.0, 0.0, 0.0],
        dt=dt,
        device=cfg.controller.mppi.device if "controller" in cfg else "cpu",
    )

    # Instantiate Controller
    controller = instantiate_controller(name, cfg, robot_params)

    # Setup goals
    goals_dict = {f"agent_{i}": np.array(goal_locations[i]) for i in range(num_agents)}
    controller.set_goals(goals_dict)

    # Convert Env params to EnvConfig object
    env_cfg = EnvConfig(
        world_x_size=x_size,
        world_y_size=y_size,
        max_steps=max_steps,
        clock=dt,
        render=cfg.env.render,
        render_save_every=cfg.env.render_save_every,
        goal_radius=goal_radius,
        num_agents=num_agents,
        collision_radius=collision_radius,
        terminate_on_collision=cfg.env.terminate_on_collision,
        collision_penalty=cfg.env.collision_penalty,
        smoke_density_threshold=cfg.env.smoke_density_threshold,
        initial_locations=initial_positions,
        goal_locations=goal_locations,
        save_transitions=cfg.env.save_transitions,
    )

    # Set playback path manually
    from src.env.simulator.playback import PlaybackParams

    playback_params = PlaybackParams(data_path=cfg.simulator.data_path)

    # Set sensors config dynamically from composed config
    sensor_params = OmegaConf.to_object(cfg.sensor)
    sensor_params.world_x_size = x_size
    sensor_params.world_y_size = y_size

    # Dummy warm-up to compile JIT / trace JAX operations before starting benchmark measurement
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

    # Accumulators for controller statistics
    success_list = []
    collision_list = []
    reached_drones_list = []
    collided_drones_list = []
    steps_to_goal_list = []
    min_separation_list = []
    smoothness_list = []
    smoke_exposure_list = []
    planning_latencies = []

    # Bucle de episodios
    for ep_idx in range(episodes):
        logger.info(f"[{name}] Running Episode {ep_idx + 1}/{episodes}...")

        # Instantiate SmokeEnv
        env = SmokeEnv(
            env_params=env_cfg,
            robot_params=robot_params,
            sensor_params=sensor_params,
            simulator_params=playback_params,
        )

        # Parche de ruta de transiciones para evitar sobreescritura de datasets
        env.save_transitions_path = os.path.join(output_dir, "transitions", name, f"ep_{ep_idx}")

        # Reset and initial state setup
        initial_state_list = []
        for i in range(num_agents):
            initial_state_list.append(
                {
                    "location": np.array(initial_positions[i], dtype=np.float32),
                    "angle": np.array([initial_headings[i]], dtype=np.float32),
                }
            )

        obs, _ = env.reset(initial_state=initial_state_list, seed=ep_idx)

        # State tracking arrays
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

        finished = False
        t = 0

        # Step loop
        max_steps_limit = steps if steps is not None else max_steps
        while not finished and t < max_steps_limit:
            # 1. Extract smoke grid from sensor
            first_agent_obs = obs.get("agent_0", list(obs.values())[0])
            if "smoke_density" in first_agent_obs and len(first_agent_obs["smoke_density"]) > 0:
                smoke_density = first_agent_obs["smoke_density"]
                coords_density = first_agent_obs["smoke_density_location"]

                if torch.is_tensor(smoke_density):
                    smoke_map_flat = (
                        smoke_density.squeeze().detach().cpu().numpy().astype(np.float32)
                    )
                else:
                    smoke_map_flat = np.asarray(smoke_density).squeeze().astype(np.float32)

                if torch.is_tensor(coords_density):
                    coords_flat = coords_density.detach().cpu().numpy()
                else:
                    coords_flat = np.asarray(coords_density)

                controller.set_maps(
                    deque([(coords_flat, smoke_map_flat)] * mppi_horizon, maxlen=mppi_horizon)
                )

            # 2. Get commands
            start_plan = time.time()
            commands_dict = controller.get_commands(obs)
            step_latencies.append((time.time() - start_plan) * 1000.0)  # ms

            # Format action values
            step_actions = {}
            for k, cmd in commands_dict.items():
                if torch.is_tensor(cmd):
                    step_actions[k] = cmd.detach().cpu().numpy()
                else:
                    step_actions[k] = np.asarray(cmd)

            # 3. Environment Step
            obs, reward, terminated, truncated, _ = env.step(step_actions)

            # 4. Compute metrics
            positions = [obs[f"agent_{i}"]["location"] for i in range(num_agents)]

            # Check collisions
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < (2.0 * collision_radius):
                        collision_occurred = True
                        drone_collided_flags[i] = True
                        drone_collided_flags[j] = True
                    if dist < min_separation_ep:
                        min_separation_ep = dist

            # Track goal arrival
            for i in range(num_agents):
                dist = np.linalg.norm(positions[i] - goal_locations[i])
                if dist < goal_radius:
                    drone_reached_flags[i] = True
                    if steps_to_reach[i] == max_steps:
                        steps_to_reach[i] = t

            # Track control effort and trajectories
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

            all_term = all(terminated.values()) if isinstance(terminated, dict) else terminated
            any_trunc = any(truncated.values()) if isinstance(truncated, dict) else truncated
            if all_term or any_trunc:
                finished = True

            t += 1

        env.close()

        # Save episode trajectory incrementally
        traj_dir = os.path.join(output_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, f"{name}_ep_{ep_idx + 1}_trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(episode_trajectories, f, indent=2)

        # Success definition: all agents reached goals and no collisions occurred
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

    # Aggregate controller metrics
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

    # Save/append results incrementally to results.csv
    csv_file = os.path.join(output_dir, "results.csv")
    new_row = {"Controller": name, **aggregated}
    df = pd.DataFrame([new_row])
    if os.path.exists(csv_file):
        df_old = pd.read_csv(csv_file)
        # Drop if controller row already exists to avoid duplicates on re-runs
        df_old = df_old[df_old["Controller"] != name]
        df_combined = pd.concat([df_old, df], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
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
):
    """Main Prefect flow orchestrating the safety controller evaluations."""
    logger.info(
        f"Running Controller Benchmark Flow over {episodes} episodes (steps limit: {steps})."
    )
    logger.info(f"Device: {device or 'default'} | Output folder: {output_dir}")

    # Ensure base output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # All benchmark configurations to evaluate
    controllers_meta = {
        "nominal": "",
        "cbf_filter": "cbf/filter",
        "cbf_rollout": "cbf/rollout",
        "cbf_penalty": "cbf/penalty",
        "hj_filter": "hj/filter",
        "hj_rollout": "hj/rollout",
        "hj_online_rollout": "hj/online_rollout",
        "hj_penalty": "hj/penalty",
    }

    # Filter if user requested specific ones
    if selected_controllers:
        controllers_meta = {k: v for k, v in controllers_meta.items() if k in selected_controllers}

    results = {}
    for name, config_file in controllers_meta.items():
        # Execute each controller evaluation task
        results[name] = evaluate_controller_task(
            name=name,
            config_override_name=config_file,
            episodes=episodes,
            device_override=device,
            output_dir=output_dir,
            steps=steps,
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
    args = parser.parse_args()

    # Auto-resolve output path
    if args.output_dir is None:
        import datetime

        project_root = "/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        args.output_dir = os.path.join(project_root, "outputs", "benchmark", timestamp)
    else:
        args.output_dir = os.path.abspath(args.output_dir)

    # Execute Prefect Flow
    benchmark_flow(
        episodes=args.episodes,
        device=args.device,
        output_dir=args.output_dir,
        selected_controllers=args.controllers,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
