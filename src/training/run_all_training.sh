#!/bin/bash
set -e  # Exit on error

# Use the specific python environment
PYTHON=/home/emunoz/dev/safe-nav-smoke/.env/bin/python

export CUDA_VISIBLE_DEVICES=4
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

# # 3. RNP Multistep
# echo "-------------------------------------------"
# echo "Starting training for RNP Multistep (train_rnp_multistep.py)..."
# $PYTHON src/training/train_rnp_multistep.py training.experiment_name="rnp_multistep"
# echo "✅ RNP Multistep Passed"

# # 4. RNP Multistep (No Bias)
# echo "-------------------------------------------"
# echo "Starting training for RNP Multistep (train_rnp_multistep.py)..."
# $PYTHON src/training/train_rnp_multistep.py training.experiment_name="rnp_multistep" training.sampling.beta_end=0.0
# echo "✅ RNP Multistep No Bias Passed"

# echo "-------------------------------------------"
# echo "Starting training for RNP Multistep (train_rnp_multistep.py)..."
# $PYTHON src/training/train_rnp_multistep.py training.experiment_name="rnp_multistep_3" training.model.forecast_horizon=3
# echo "✅ RNP Multistep horizon 3 Passed"

# echo "-------------------------------------------"
# echo "Starting training for RNP Multistep (train_rnp_multistep.py)..."
# $PYTHON src/training/train_rnp_multistep.py training.experiment_name="rnp_multistep_8" training.model.forecast_horizon=8
# echo "✅ RNP Multistep horizon 8 Passed"

# 5. RNP Multistep (Bias)
# echo "-------------------------------------------"
# echo "Starting training for RNP Multistep (train_rnp_multistep.py)..."
# $PYTHON src/training/train_rnp_multistep.py training.experiment_name="rnp_multistep_high_bias"
# echo "✅ RNP Multistep Higher Bias Passed"

# echo "-------------------------------------------"
# echo "1/6  FNO with Scheduled Sampling (bias)..."
# $PYTHON src/training/train_fno.py \
#   training.experiment_name="fno_bias" \
#   training.visualizer.visualize_every=10
# echo "✅ FNO bias Passed"

# echo "-------------------------------------------"
# echo "2/6  FNO without Scheduled Sampling (no bias)..."
# $PYTHON src/training/train_fno.py \
#   training.experiment_name="fno_no_bias" \
#   training.sampling.beta_start=0.0 \
#   training.sampling.beta_end=0.0 \
#   training.visualizer.visualize_every=10
# echo "✅ FNO no bias Passed"

# echo "-------------------------------------------"
# echo "3/6  FNO Multistep H=3..."
# $PYTHON src/training/train_fno_multistep.py \
#   training.experiment_name="fno_multistep_h3" \
#   training.model.forecast_horizon=3 \
#   training.visualizer.visualize_every=10
# echo "✅ FNO Multistep H=3 Passed"

# echo "-------------------------------------------"
# echo "4/6  FNO Multistep H=5..."
# $PYTHON src/training/train_fno_multistep.py \
#   training.experiment_name="fno_multistep_h5" \
#   training.model.forecast_horizon=5 \
#   training.visualizer.visualize_every=10
# echo "✅ FNO Multistep H=5 Passed"

# echo "-------------------------------------------"
# echo "5/6  FNO Multistep H=8..."
# $PYTHON src/training/train_fno_multistep.py \
#   training.experiment_name="fno_multistep_h8" \
#   training.model.forecast_horizon=8 \
#   training.visualizer.visualize_every=10
# echo "✅ FNO Multistep H=8 Passed"

# echo "-------------------------------------------"
# echo "1/6  FNO base (bias + mean feed)"
# $PYTHON src/training/train_fno.py \
#   training.experiment_name="fno_bias" \
#   training.visualizer.visualize_every=10
# echo "✅ FNO bias Passed"

# echo "-------------------------------------------"
# echo "2/6  FNO no scheduled sampling"
# $PYTHON src/training/train_fno.py \
#   training.experiment_name="fno_no_bias" \
#   training.sampling.beta_start=0.0 \
#   training.sampling.beta_end=0.0 \
#   training.visualizer.visualize_every=10
# echo "✅ FNO no bias Passed"

# echo "-------------------------------------------"
# echo "3/6  FNO Multistep H=3"
# $PYTHON src/training/train_fno.py \
#   training.experiment_name="fno_multistep_h3" \
#   training.model.forecast_horizon=3 \
#   training.visualizer.visualize_every=10
# echo "✅ FNO Multistep H=3 Passed"

# echo "-------------------------------------------"
# echo "4/6  FNO Multistep H=5"
# $PYTHON src/training/train_fno.py \
#   training.experiment_name="fno_multistep_h5" \
#   training.model.forecast_horizon=5 \
#   training.visualizer.visualize_every=10
# echo "✅ FNO Multistep H=5 Passed"

# echo "-------------------------------------------"
# echo "5/6  FNO Multistep H=8"
# $PYTHON src/training/train_fno.py \
#   training.experiment_name="fno_multistep_h8" \
#   training.model.forecast_horizon=8 \
#   training.visualizer.visualize_every=10
# echo "✅ FNO Multistep H=8 Passed"

echo "-------------------------------------------"
echo "6/6  FNO Uncertainty Input (2-channel: mean + std)"
$PYTHON src/training/train_fno.py \
  --config-name fno_uncertainty_train \
  training.experiment_name="fno_uncertainty" \
  training.visualizer.visualize_every=10
echo "✅ FNO Uncertainty Passed"
