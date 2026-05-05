import logging
import os
import sys

import hydra
import numpy as np
import torch
from datasets import Dataset
from omegaconf import DictConfig
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from controllers.mppi_control_dyn import MPPIControlDyn, MPPIControlParams
from envs.smoke_env import SmokeEnv

logger = logging.getLogger(__name__)

class MPPIFutureSighted(MPPIControlDyn):
    def __init__(self, env, *args, future_horizon=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.future_horizon = future_horizon
    def _compute_risk_cost(self, states: torch.Tensor, t: int) -> torch.Tensor:
        relative_step = int(t)
        states_np = states[:, :2].detach().cpu().numpy()
        densities = self.env.get_future_smoke_density(states_np, relative_step=relative_step)
        return torch.tensor(densities, dtype=self.dtype, device=self.device).squeeze(-1)
    def running_cost(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        w_dist, w_risk = 2.0, 50.0
        dist_cost = torch.norm(states[:, :2] - self._goal, dim=1)
        risk_cost = self._compute_risk_cost(states, t)
        return w_dist * dist_cost + w_risk * risk_cost

def generate_random_start_goal(env, blob_safety_radius=2.0, min_goal_dist=15.0):
    world_x, world_y = env.env_params.world_x_size, env.env_params.world_y_size
    max_attempts = 1000
    for _ in range(max_attempts):
        start_pos = np.array([np.random.uniform(2, world_x - 2), np.random.uniform(2, world_y - 2)])
        check_points = start_pos + np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,-1]]) * blob_safety_radius / 2
        if np.any(env.smoke_simulator.get_smoke_density(check_points) > 0.1): continue
        for _ in range(100):
            goal_pos = np.array([np.random.uniform(2, world_x - 2), np.random.uniform(2, world_y - 2)])
            if np.linalg.norm(start_pos - goal_pos) < min_goal_dist: continue
            check_points_g = goal_pos + np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,-1]]) * blob_safety_radius/2
            if np.any(env.smoke_simulator.get_smoke_density(check_points_g) > 0.1): continue
            return start_pos, goal_pos
    raise RuntimeError("Could not find valid start/goal pair")

@hydra.main(version_base=None, config_path="../configs/experiments", config_name="distillation_data_gen")
def main(cfg: DictConfig):
    output_path = cfg.get("output_path", "data/distillation_v1")
    os.makedirs(output_path, exist_ok=True)
    env = SmokeEnv(cfg.env_cfg_path)
    env.env_params.playback_path = cfg.playback_data_path
    visualize = cfg.get("visualize", False)
    from envs.smoke_env import RenderMode
    env.env_params.render = RenderMode.HUMAN if visualize else None
    if not isinstance(env.smoke_simulator, env.smoke_simulator.__class__): env._setup_simulator()
    mppi_params = MPPIControlParams(num_samples=cfg.get("num_samples", 100), horizon=cfg.get("horizon", 20), lambda_=cfg.get("lambda_", 1.0))
    controller = MPPIFutureSighted(env, robot_params=env.robot_params, robot_type=env.robot_params.robot_type, mppi_params=mppi_params, dt=env.env_params.clock)
    num_episodes = cfg.get("num_episodes", 100)
    def generate_rows():
        for ep_idx in tqdm(range(num_episodes), desc="Episodes"):
            obs, _ = env.reset(options={"episode_idx": ep_idx})
            try: start_pos, goal_pos = generate_random_start_goal(env)
            except: continue
            initial_state = {"location": start_pos, "angle": np.random.uniform(0, 2*np.pi)}
            if "velocity" in obs: initial_state["velocity"] = 0.0
            obs, _ = env.reset(initial_state=initial_state, options={"episode_idx": ep_idx})
            env.env_params.goal_location = tuple(goal_pos)
            controller.set_goal(goal_pos)
            prev_action = np.zeros(env.robot_params.action_dim)
            done, step_count = False, 0
            while not done and step_count < env.env_params.max_steps:
                pose, scan = env.robot.get_state(), obs["smoke_density"]
                local_map = env.get_local_smoke_map(size_meters=env.robot_params.action_max[0] * mppi_params.horizon * env.env_params.clock * 2.0)
                controller.set_state(pose)
                action = controller.get_command().detach().cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                yield {"observation": scan.tolist(), "local_map": local_map.tolist(), "action": action.tolist(), "prev_action": prev_action.tolist(), "state": pose.tolist(), "goal": goal_pos.tolist(), "success": terminated, "episode_id": ep_idx, "step_id": step_count}
                obs, prev_action, done, step_count = next_obs, action, terminated or truncated, step_count + 1
    ds = Dataset.from_generator(generate_rows)
    ds.save_to_disk(output_path)

if __name__ == "__main__":
    main()
