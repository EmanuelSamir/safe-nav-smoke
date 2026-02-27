import os
import abc
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra

from src.utils import LoggerMetrics, dataclass_json_dump
from src.visualization import StandardRenderer
from src.time_tracker import TimeTracker

# Environment & simulator imports (used by subclasses)
from envs.smoke_env_dyn import EnvParams, DynamicSmokeEnv, DynamicSmokeParams
from agents.basic_robot import RobotParams
from simulator.sensor import GlobalSensorParams, DownwardsSensorParams
from simulator.dynamic_smoke import SmokeBlobParams

# Configure global logger
log = logging.getLogger(__name__)

class BaseExperiment(abc.ABC):
    """
    Abstract base class for all experiments.
    Handles configuration, logging, metrics, and experiment life cycle.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        try:
            from hydra.core.hydra_config import HydraConfig
            self.output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            self.output_dir = Path(os.getcwd())
        
        # Initialize utilities
        self.metrics = LoggerMetrics()
        self.time_tracker = TimeTracker()
        self.renderer = None
        
        # Experiment state
        self.env = None
        self.robot = None
        self.controller = None
        self.finished = False
        
        log.info(f"Initializing experiment: {cfg.experiment.name}")
        log.info(f"Output directory: {self.output_dir}")
        
        # Save full configuration
        dump_path = self.output_dir / "config_dump.yaml"
        with open(dump_path, "w") as f:
            OmegaConf.save(cfg, f)

    @abc.abstractmethod
    def setup(self):
        """
        Initialize the environment, controller, and any model components.
        Each subclass is responsible for building env, robot, sensor, and
        controller params from self.cfg.
        """
        pass
        

    @abc.abstractmethod
    def run_episode(self) -> Dict[str, Any]:
        """
        Executes a full simulation episode.
        Must be implemented by subclasses.
        """
        pass

    def setup_renderer(self, has_predictions: bool = False):
        """Initializes the standard renderer."""
        if self.cfg.env.render:
            seq_filepath = str(self.output_dir / "playback.gif") if self.cfg.env.render == "rgb_array" else None
            opts = {
                'render': self.cfg.env.render,
                'env_params': self.env.env_params,
                'seq_filepath': seq_filepath,
                'has_predictions': has_predictions
            }
            self.renderer = StandardRenderer(opts=opts)

    def render_step(self, info: Dict[str, Any], t: int):
        """Passes information to the standard renderer."""
        render_save_every = self.cfg.env.get("render_save_every", 1)
        if self.renderer and t % render_save_every == 0:
            # Need to provide logger_metrics so renderer can read current states
            info['logger_metrics'] = self.metrics
            self.renderer.render(info)
            if self.cfg.env.render == "rgb_array":
                self.renderer.save_frame()

    def teardown(self):
        """Resource cleanup and saving final results."""
        log.info("Finishing experiment and saving results...")
        
        # Save metrics
        metrics_path = self.output_dir / "metrics.csv"
        self.metrics.dump_to_csv(str(metrics_path))
        log.info(f"Metrics saved at {metrics_path}")
        
        # Save timings
        time_path = self.output_dir / "timing.csv"
        time_df = pd.DataFrame(self.time_tracker.as_dict())
        time_df.to_csv(time_path, index=False)
        
        # Close environment and renderer
        if self.env:
            self.env.close()
        
        if self.renderer:
            self.renderer.close() # Ensure videos are saved

    def run(self):
        """Main method to execute the experiment."""
        try:
            self.setup()
            results = self.run_episode()
            return results
        except Exception as e:
            log.error(f"Error during experiment execution: {e}", exc_info=True)
            raise
        finally:
            self.teardown()

    # --- Common Helper Methods ---

    def get_initial_location(self) -> np.ndarray:
        """Selects a random initial location from the configuration."""
        locs = self.cfg.experiment.initial_locations
        idx = np.random.randint(0, len(locs))
        return np.array(locs[idx])

    def get_goal_location(self) -> np.ndarray:
        """Selects a random goal location from the configuration."""
        locs = self.cfg.experiment.goal_locations
        idx = np.random.randint(0, len(locs))
        return np.array(locs[idx])

    def check_termination(self, state, reward, terminated, truncated) -> bool:
        """Checks termination conditions and logs final state."""
        if terminated or truncated:
            status = 'truncated'
            if terminated:
                status = 'reached_goal' if reward > 0.0 else 'crashed'
            
            self.metrics.add_value('status', status)
            log.info(f"Episode finished. Status: {status}")
            return True
        
        self.metrics.add_value('status', 'running')
        return False

    def log_common_metrics(self, state, action, t):
        """Logs standard metrics common to most experiments."""
        # Distance to goal
        if hasattr(self.env, 'env_params') and self.env.env_params.goal_location is not None:
            dist = np.linalg.norm(state["location"] - self.env.env_params.goal_location)
            self.metrics.add_value('dist_to_goal', dist)

        # Smoke at robot location
        if hasattr(self.env, 'get_smoke_density_in_robot'):
            smoke = self.env.get_smoke_density_in_robot()[0]
            self.metrics.add_value('smoke_on_robot', smoke)
            
            # Accumulated
            last_acc = self.metrics.get_last_value('smoke_on_robot_acc')
            self.metrics.add_value('smoke_on_robot_acc', last_acc + smoke)

        # Actions and State
        for i, a in enumerate(action):
            self.metrics.add_value(f'action_{i}', a)
        
        for i, s in enumerate(state["location"]):
            self.metrics.add_value(f'state_{i}', s)
            
        self.metrics.add_value('steps', t)
        self.metrics.add_value('time', t * self.cfg.env.clock)
