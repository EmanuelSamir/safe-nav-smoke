#!/bin/bash
set -e  # Exit on error

# Use the specific python environment
PYTHON=/home/emunoz/dev/safe-nav-smoke/.env/bin/python

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=/home/emunoz/dev/safe-nav-smoke

# 1. RNP
echo "-------------------------------------------"
echo "Testing setup for RNP (train_rnp.py)..."
$PYTHON src/training/train_rnp.py
echo "✅ RNP Passed"

# 2. RNP Residual
echo "-------------------------------------------"
echo "Testing setup for RNP Residual (train_rnp_residual.py)..."
$PYTHON src/training/train_rnp_residual.py
echo "✅ RNP Residual Passed"

