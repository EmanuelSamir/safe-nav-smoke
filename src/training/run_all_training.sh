#!/bin/bash
set -e  # Exit on error

# Use the specific python environment
PYTHON=/home/emunoz/dev/safe-nav-smoke/.env/bin/python

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/emunoz/dev/safe-nav-smoke

# # 1. RNP
# echo "-------------------------------------------"
# echo "Starting training for RNP (train_rnp.py)..."
# $PYTHON src/training/train_rnp.py training.experiment_name="rnp_base"
# echo "✅ RNP Passed"

# # 2. RNP (No Sampling)
# echo "-------------------------------------------"
# echo "Starting training for RNP without Scheduled Sampling..."
# $PYTHON src/training/train_rnp.py training.experiment_name="rnp_no_sampling" training.sampling.beta_end=0.0
# echo "✅ RNP No Sampling Passed"

# 3. RNP Multistep
echo "-------------------------------------------"
echo "Starting training for RNP Multistep (train_rnp_multistep.py)..."
$PYTHON src/training/train_rnp_multistep.py training.experiment_name="rnp_multistep"
echo "✅ RNP Multistep Passed"
