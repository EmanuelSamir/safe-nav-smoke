"""
Metrics tracking utilities for experiments.
Consolidates LoggerMetrics from multiple experiment files.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class MetricsTracker:
    """
    Unified metrics tracking with optional TensorBoard integration.
    
    Replaces duplicate LoggerMetrics classes across experiments.
    """
    
    def __init__(self, log_dir: Optional[str] = None, use_tensorboard: bool = False):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory for TensorBoard logs (if use_tensorboard=True)
            use_tensorboard: Whether to log to TensorBoard
        """
        self.values = defaultdict(list)
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        
        if use_tensorboard and log_dir:
            self.tb_writer = SummaryWriter(log_dir)
    
    def add_value(self, key: str, value: float, step: Optional[int] = None):
        """
        Add a metric value.
        
        Args:
            key: Metric name (e.g., 'loss', 'reward')
            value: Metric value
            step: Optional step/episode number for TensorBoard
        """
        self.values[key].append(value)
        
        # Log to TensorBoard if enabled
        if self.tb_writer and step is not None:
            self.tb_writer.add_scalar(key, value, step)
    
    def get_last_value(self, key: str) -> float:
        """Get the last value for a metric."""
        if key not in self.values or len(self.values[key]) == 0:
            return 0.0
        return self.values[key][-1]
    
    def get_values(self, key: Optional[str] = None) -> Any:
        """
        Get metric values.
        
        Args:
            key: Metric name. If None, returns all metrics.
            
        Returns:
            List of values for the metric, or dict of all metrics if key is None.
        """
        if key is None:
            return dict(self.values)
        return self.values.get(key, [])
    
    def get_stats(self, key: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            key: Metric name
            
        Returns:
            Dict with mean, std, min, max
        """
        values = self.get_values(key)
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        arr = np.array(values)
        return {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
        }
    
    def dump_to_csv(self, filepath: str):
        """
        Save all metrics to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        df = pd.DataFrame(self.values)
        df.to_csv(filepath, index=False)
    
    def reset(self):
        """Clear all stored metrics."""
        self.values.clear()
    
    def close(self):
        """Close TensorBoard writer if active."""
        if self.tb_writer:
            self.tb_writer.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.close()


# Backward compatibility alias
LoggerMetrics = MetricsTracker
