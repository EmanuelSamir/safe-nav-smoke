#!/bin/bash
# =====================================================================
# SLURM CONFIGURATION
# =====================================================================
#SBATCH -p pleiades                         
#SBATCH -w pleiades-0-23                     
#SBATCH --job-name=run_cbf_sweep         
#SBATCH --time=12:00:00                     
#SBATCH --cpus-per-task=8                   
#SBATCH --mem=32G                           
#SBATCH --array=0-4

#SBATCH --output=/home/emunoz/slurm_logs/cbf_sweep_%A_%a.log
#SBATCH --error=/home/emunoz/slurm_logs/cbf_sweep_%A_%a.log

# =====================================================================
# ENTORNO DE EJECUCIÓN
# =====================================================================

mkdir -p /home/emunoz/slurm_logs
mkdir -p /data/emunoz/prefect_home
export PREFECT_HOME="/data/emunoz/prefect_home"

cd /home/emunoz/dev/safe-nav-smoke/

# Launch with the specific YAML config and inject the array task ID
# No GPU needed since CBF is primarily CPU bound
apptainer exec --bind /data/emunoz:/data /data/emunoz/imgs/python_full.sif /data/emunoz/envs/dev_env/bin/python projects/single_agent/step_05_cbf_sweep_evaluation/run_sweep.py --config projects/single_agent/step_05_cbf_sweep_evaluation/config_sweep.yaml --sweep_idx $SLURM_ARRAY_TASK_ID
