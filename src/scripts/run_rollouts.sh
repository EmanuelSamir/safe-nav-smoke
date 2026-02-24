#!/bin/bash
# run_rollouts.sh — Generate rollouts for any trained model (RNP, FNO, etc.)
# Uses the unified save_rollouts.py script.
# Fill in checkpoint paths before running.

set -e

PYTHON=/home/emunoz/dev/safe-nav-smoke/.env/bin/python
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/emunoz/dev/safe-nav-smoke

DATA_PATH="/home/emunoz/dev/safe-nav-smoke/data/playback_data/test_global_source_100_100.npz"

STRIDE=15
HORIZON=15
NUM_SAMPLES=10

# ---- RNP checkpoints -------------------------------------------------------
CKPT_RNP_BIAS="/home/emunoz/dev/safe-nav-smoke/outputs/2026-02-22/17-09-42-bias/checkpoints/best_model.pt"
CKPT_RNP_NO_BIAS="/home/emunoz/dev/safe-nav-smoke/outputs/2026-02-22/19-08-14-no-bias/checkpoints/best_model.pt"
CKPT_MS_BIAS="/home/emunoz/dev/safe-nav-smoke/outputs/2026-02-22/21-24-49-multistep-bias/checkpoints/best_model.pt"
CKPT_MS_NO_BIAS="/home/emunoz/dev/safe-nav-smoke/outputs/2026-02-23/17-12-19_multistep_no_bias/checkpoints/best_model.pt"

# ---- FNO checkpoints (fill in after training) ------------------------------
CKPT_FNO_BIAS=""          # e.g. outputs/2026-02-24/.../checkpoints/best_model.pt
CKPT_FNO_NO_BIAS=""
CKPT_FNO_MS_H3=""
CKPT_FNO_MS_H5=""
CKPT_FNO_MS_H8=""
CKPT_FNO_UNCERTAINTY=""

# ---------------------------------------------------------------------------

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

    $PYTHON src/scripts/save_rollouts.py \
        --ckpt        "$ckpt"        \
        --model_type  "$model_type"  \
        --data_path   "$DATA_PATH"   \
        --output_dir  "$out_dir"     \
        --stride      "$STRIDE"      \
        --horizon     "$HORIZON"     \
        --num_samples "$NUM_SAMPLES" \
        --tag         "$tag"
}

# ---- RNP -------------------------------------------------------------------
# run_rollout "$CKPT_RNP_BIAS"    "rnp"           "saved_rollouts/rnp_bias"
# run_rollout "$CKPT_RNP_NO_BIAS" "rnp"           "saved_rollouts/rnp_no_bias"    "rnp_no_bias"
# run_rollout "$CKPT_MS_BIAS"     "rnp_multistep" "saved_rollouts/ms_bias"        "ms_bias"
# run_rollout "$CKPT_MS_NO_BIAS"  "rnp_multistep" "saved_rollouts/ms_no_bias"     "ms_no_bias"

# ---- FNO -------------------------------------------------------------------
run_rollout "$CKPT_FNO_BIAS"        "fno" "saved_rollouts/fno_bias"        "fno_bias"
run_rollout "$CKPT_FNO_NO_BIAS"     "fno" "saved_rollouts/fno_no_bias"     "fno_no_bias"
run_rollout "$CKPT_FNO_MS_H3"       "fno" "saved_rollouts/fno_ms_h3"       "fno_ms_h3"
run_rollout "$CKPT_FNO_MS_H5"       "fno" "saved_rollouts/fno_ms_h5"       "fno_ms_h5"
run_rollout "$CKPT_FNO_MS_H8"       "fno" "saved_rollouts/fno_ms_h8"       "fno_ms_h8"
run_rollout "$CKPT_FNO_UNCERTAINTY" "fno" "saved_rollouts/fno_uncertainty"  "fno_uncertainty"

echo ""
echo "All rollouts saved."
