import os
import csv
import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

logger = logging.getLogger("metrics")

@dataclass
class EpisodeMetrics:
    controller_name: str
    episode_idx: int
    success: bool
    collision_occurred: bool
    reached_drones: int
    collided_drones: int
    avg_steps_to_goal: float
    min_separation: float
    smoothness: float
    avg_planning_latency_ms: float
    smoke_q1: float
    smoke_median: float
    smoke_q3: float
    smoke_max: float
    
    def to_dict(self):
        return {
            "Controller": self.controller_name,
            "Episode": self.episode_idx,
            "Success": int(self.success),
            "Collision": int(self.collision_occurred),
            "Reached Drones": self.reached_drones,
            "Collided Drones": self.collided_drones,
            "Avg Steps": self.avg_steps_to_goal,
            "Min Separation (m)": self.min_separation,
            "Smoothness": self.smoothness,
            "Avg Latency (ms)": self.avg_planning_latency_ms,
            "Smoke Q1": self.smoke_q1,
            "Smoke Median": self.smoke_median,
            "Smoke Q3": self.smoke_q3,
            "Smoke Max": self.smoke_max,
        }

class BenchmarkTracker:
    def __init__(self, output_dir: str, controller_name: str):
        self.output_dir = output_dir
        self.controller_name = controller_name
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_file = os.path.join(self.output_dir, f"episodes_results_{self.controller_name}.csv")
        self.headers = list(EpisodeMetrics.__annotations__.keys()) # Not exactly, we'll use keys from to_dict
        
        # Initialize file with headers if it doesn't exist
        self._initialize_csv()
        
    def _initialize_csv(self):
        # Dummy instance just to get keys
        dummy = EpisodeMetrics("", 0, False, False, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).to_dict()
        self.headers = list(dummy.keys())
        
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                
    def record_episode(self, metrics: EpisodeMetrics):
        """Append a single episode's results to the CSV incrementally."""
        with open(self.csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(metrics.to_dict())
        logger.debug(f"Saved episode {metrics.episode_idx} for {metrics.controller_name}")
