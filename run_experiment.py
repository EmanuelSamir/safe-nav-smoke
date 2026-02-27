#!/usr/bin/env python3
"""
Main script to run experiments.

Usage:
  python run_experiment.py experiment=no_risk
  python run_experiment.py experiment=cbf
  python run_experiment.py experiment=persistent
  python run_experiment.py experiment=behavior_prediction experiment.model_type=fno experiment.checkpoint_path=<path>
"""

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.experiments.mppi_experiment import MPPIExperiment
from src.experiments.cbf_experiment import CBFExperiment
from src.experiments.behavior_prediction_experiment import BehaviorPredictionExperiment


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(f"Running experiment: {cfg.experiment.name}")
    print(OmegaConf.to_yaml(cfg.experiment))

    exp_type = cfg.experiment.get("type")

    if exp_type == "no-risk":
        experiment = MPPIExperiment(cfg, risk_aware=False)
    elif exp_type == "cbf":
        experiment = CBFExperiment(cfg)
    elif exp_type == "persistent":
        experiment = MPPIExperiment(cfg, risk_aware=True)
    elif exp_type == "behavior-prediction":
        experiment = BehaviorPredictionExperiment(cfg)
    else:
        raise ValueError(
            f"Unknown experiment type: '{exp_type}'. "
            "Choose from: no-risk | cbf | persistent | behavior-prediction"
        )

    experiment.run()


if __name__ == "__main__":
    main()
