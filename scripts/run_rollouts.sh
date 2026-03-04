#!/bin/bash

set -e

PYTHON=/home/emunoz/dev/safe-nav-smoke/.env/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/emunoz/dev/safe-nav-smoke

DATA_PATH="/home/emunoz/dev/safe-nav-smoke/data/playback_data/test_global_source_100_100.npz"

STRIDE=15
HORIZON=15
NUM_SAMPLES=10

CKPT_FNO3D="/home/emunoz/dev/safe-nav-smoke/outputs/2026-02-24/14-10-03_fno_3d/checkpoints/best_model.pt"
CKPT_CONV_LSTM_LAST="/home/emunoz/dev/safe-nav-smoke/outputs/2026-03-02/19-52-54/checkpoints/best_model.pt"

run_rollout() {
    local ckpt=$1
    local model_type=$2
    local out_dir=$3
    local tag=${4:-$model_type}   # optional 4th arg overrides the NPZ key prefix

    if [ -z "$ckpt" ]; then
        echo "  [SKIP] $tag — checkpoint not set"
        return
    fi
    if [ ! -f "$ckpt" ]; then
        echo "  [SKIP] $tag — checkpoint not found: $ckpt"
        return
    fi

    echo ""
    echo "====================================================="
    echo "  Model     : $tag"
    echo "  Type      : $model_type"
    echo "  Checkpoint: $ckpt"
    echo "  Output    : $out_dir"
    echo "====================================================="

    $PYTHON scripts/save_rollouts.py \
        --ckpt        "$ckpt"        \
        --model_type  "$model_type"  \
        --data_path   "$DATA_PATH"   \
        --output_dir  "$out_dir"     \
        --stride      "$STRIDE"      \
        --horizon     "$HORIZON"     \
        --num_samples "$NUM_SAMPLES" \
        --tag         "$tag"
}

# ---- FNO3D -----------------------------------------------------------------
# run_rollout "$CKPT_FNO3D" "fno_3d" "saved_rollouts/fno_3d" "fno_3d"

# ---- CONV_LSTM -------------------------------------------------------------
run_rollout "$CKPT_CONV_LSTM_LAST" "conv_lstm" "saved_rollouts/conv_lstm_last" "conv_lstm_last"


echo ""
echo "All rollouts saved."

