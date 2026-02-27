#!/bin/bash
# run_all.sh
# description: Runs all 4 safe navigation smoke experiments across 120 playback episodes sequentially and saves looping GIFs.

# Make sure we fail fast if any experiment crashes
set -e

# Base command variables
PYTHON_EXEC="/Users/emanuelsamir/Documents/dev/cmu/py-envs/dev/bin/python"
SCRIPT="run_experiment.py"
NUM_EPISODES=120

echo "======================================================="
echo " Starting Safe Navigation Smoke Experiments Suite"
echo " Rendering: rgb_array (saving GIFs every 5th step)"
echo " Total Episodes: $NUM_EPISODES"
echo "======================================================="

# Function to run an experiment across all episodes
run_experiment_loop() {
    local exp_name=$1
    local extra_args=$2

    echo "-------------------------------------------------------"
    echo "▶ Running: $exp_name"
    for i in $(seq 0 $((NUM_EPISODES - 1))); do
        echo "  - Episode $i..."
        $PYTHON_EXEC $SCRIPT \
            experiment=${exp_name} \
            +experiment.episode_idx=$i \
            hydra.run.dir=outputs/${exp_name}/ep_${i} \
            env.render=rgb_array \
            env.render_save_every=1 \
            $extra_args
    done
    echo "✔ $exp_name finished."
}

# # 1. No Risk (MPPI without smoke map consideration)
# run_experiment_loop "no_risk" ""

# # 2. CBF (Control Barrier Functions - hardcoded smoke threshold)
# run_experiment_loop "cbf" ""

# # 3. Persistent (MPPI assuming static smoke map at each step)
# run_experiment_loop "persistent" ""

# # 4. Behavior Prediction (MPPI with FNO3D)
# run_experiment_loop "behavior_prediction" "experiment.model_type=fno_3d experiment.name=behavior_prediction_fno_3d"

# 5. Behavior Prediction (MPPI with FNO3D + CVAR)
run_experiment_loop "behavior_prediction" "experiment.model_type=fno_3d experiment.cvar_alpha=0.9 experiment.name=behavior_prediction_fno_3d_cvar"

echo "======================================================="
echo " All experiments finished successfully! 🎉"
echo " Check the outputs/ directory for the generated playback.gif and metrics files."
echo "======================================================="
