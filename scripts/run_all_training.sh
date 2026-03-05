#!/bin/bash
# TODO: Hide sh files for blinded review

set -e  # Exit on error

# Use the specific python environment
PYTHON=/home/emunoz/dev/safe-nav-smoke/.env/bin/python

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/emunoz/dev/safe-nav-smoke

echo "-------------------------------------------"
echo "FNO 3D"
$PYTHON src/training/train_fno_3d.py \
  --config-name fno_3d_train \
  training.experiment_name="fno_3d_nll_beta" \
  training.loss.beta=0.5
echo "✅ FNO 3D NLL Beta Passed"

echo "-------------------------------------------"
echo "FNO 3D - Min Std 0.001"
$PYTHON src/training/train_fno_3d.py \
  --config-name fno_3d_train \
  training.experiment_name="fno_3d_min_std_0.001" \
  training.model.min_std=0.001
echo "✅ FNO 3D Min Std 0.001 Passed"

echo "-------------------------------------------"
echo "FNO 3D - NLL Beta 0.5 - Min Std 0.001"
$PYTHON src/training/train_fno_3d.py \
  --config-name fno_3d_train \
  training.experiment_name="fno_3d_nll_beta_min_std_0.001" \
  training.loss.beta=0.5 \
  training.model.min_std=0.001
echo "✅ FNO 3D NLL Beta - Min Std 0.001 Passed"


# echo "-------------------------------------------"
# echo "ConvLSTM Baseline"
# $PYTHON src/training/train_conv_lstm.py \
#   --config-name conv_lstm_train \
#   training.experiment_name="conv_lstm" \
#   training.visualizer.visualize_every=10
# echo "✅ ConvLSTM Passed"

# echo "-------------------------------------------"
# echo "PFNO 3D"
# $PYTHON src/training/train_pfno.py \
#   --config-name pfno_train \
#   training.experiment_name="pfno_3d" 
# echo "✅ PFNO 3D Passed"