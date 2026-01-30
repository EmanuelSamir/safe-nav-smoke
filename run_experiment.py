#!/usr/bin/env python3
"""
Main script to run experiments.
Usage: python run_experiment.py experiment=base env=smoke_env agent=dubins
"""

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.experiments.mppi_experiment import MPPIExperiment
from src.experiments.cbf_experiment import CBFExperiment
from src.experiments.vaessm_experiment import VAESSMExperiment

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(f"Running experiment with config: {cfg.experiment.name}")
    
    exp_type = cfg.experiment.get("type")
    
    if exp_type == "cbf":
        experiment = CBFExperiment(cfg)
    elif exp_type == "vaessm":
        experiment = VAESSMExperiment(cfg)
    else:
        # Default to MPPI/Base
        experiment = MPPIExperiment(cfg)
        
    experiment.run()

if __name__ == "__main__":
    main()
