#!/bin/bash

set -e

PYTHON=/home/emunoz/dev/safe-nav-smoke/.env/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/emunoz/dev/safe-nav-smoke

DATA_PATH="/home/emunoz/dev/safe-nav-smoke/data/playback_data/test_global_source_100_100.npz"

STRIDE=20   # 15 for 10s, 30 for 5s
HORIZON=20
NUM_SAMPLES=1
# NUM_EPISODES=4

CKPT_FNO3D="/home/emunoz/dev/safe-nav-smoke/outputs/2026-02-24/14-10-03_fno_3d/checkpoints/best_model.pt"
CKPT_CONV_LSTM_LAST="/home/emunoz/dev/safe-nav-smoke/outputs/2026-03-05/13-19-10/checkpoints/best_model.pt"

CKPT_FNO3D_BETA_NLL="/home/emunoz/dev/safe-nav-smoke/outputs/2026-03-05/03-03-51/checkpoints/best_model.pt"
CKPT_FNO3D_NLL_1e3="/home/emunoz/dev/safe-nav-smoke/outputs/2026-03-05/06-36-19/checkpoints/best_model.pt"

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
        --num_samples "$NUM_SAMPLES" #\
        #--num_episodes "$NUM_EPISODES" 
}

# ---- FNO3D -----------------------------------------------------------------
run_rollout "$CKPT_FNO3D" "fno_3d" "saved_rollouts/fno_3d"
run_rollout "$CKPT_FNO3D_BETA_NLL" "fno_3d" "saved_rollouts/fno_3d_beta_nll"
run_rollout "$CKPT_FNO3D_NLL_1e3" "fno_3d" "saved_rollouts/fno_3d_nll_1e3"


# ---- CONV_LSTM -------------------------------------------------------------
run_rollout "$CKPT_CONV_LSTM_LAST" "conv_lstm" "saved_rollouts/conv_lstm_last" "conv_lstm_last"


echo ""
echo "All rollouts saved."

