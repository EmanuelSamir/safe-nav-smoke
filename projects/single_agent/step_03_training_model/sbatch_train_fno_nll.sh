#!/bin/bash
# =====================================================================
# SLURM CONFIGURATION
# =====================================================================
#SBATCH -p pleiades                         
#SBATCH -w pleiades-0-23                     
#SBATCH --job-name=train_fno_nll         
#SBATCH --time=12:00:00                     
#SBATCH --cpus-per-task=8                   
#SBATCH --gpus=1                            
#SBATCH --mem=32G                           

#SBATCH --output=/home/emunoz/slurm_logs/train_fno_nll_%j.log
#SBATCH --error=/home/emunoz/slurm_logs/train_fno_nll_%j.log

# =====================================================================
# ENTORNO DE EJECUCIÓN
# =====================================================================

mkdir -p /home/emunoz/slurm_logs
mkdir -p /data/emunoz/prefect_home
export PREFECT_HOME="/data/emunoz/prefect_home"

cd /home/emunoz/dev/safe-nav-smoke/

# Launch with the specific YAML config
apptainer exec --nv --bind /data/emunoz:/data /data/emunoz/imgs/python_full.sif /data/emunoz/envs/dev_env/bin/python projects/single_agent/step_03_training_model/run_fno.py --config fno_config_nll.yaml
