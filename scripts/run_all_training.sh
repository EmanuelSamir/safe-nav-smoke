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
  training.experiment_name="fno_3d" \
  training.visualizer.visualize_every=10
echo "✅ FNO 3D Passed"

echo "-------------------------------------------"
echo "ConvLSTM Baseline"
$PYTHON src/training/train_conv_lstm.py \
  --config-name conv_lstm_train \
  training.experiment_name="conv_lstm" \
  training.visualizer.visualize_every=10
echo "✅ ConvLSTM Passed"