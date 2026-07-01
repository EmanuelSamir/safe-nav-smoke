#!/bin/bash
# =====================================================================
# SLURM CONFIGURATION
# =====================================================================
#SBATCH -p pleiades                         # general partition
#SBATCH -w pleiades-0-17                    # lightweight node
#SBATCH --job-name=structured_playback        # job name
#SBATCH --time=08:00:00                     # 8 hours limit
#SBATCH --cpus-per-task=8                   # 8 cpu cores
#SBATCH --gpus=1                            # 1 GPU
#SBATCH --mem=16G                           # 16 GB RAM

#SBATCH --output=/home/emunoz/slurm_logs/structured_%j.log
#SBATCH --error=/home/emunoz/slurm_logs/structured_%j.log

# =====================================================================
# ENTORNO DE EJECUCIÓN
# =====================================================================

# 1. Folder creation
mkdir -p /home/emunoz/slurm_logs
mkdir -p /data/emunoz/prefect_home

# 2. Avoid Prefect SQLite locking
export PREFECT_HOME="/data/emunoz/prefect_home"

# 3. Move to the repository directory
cd /home/emunoz/dev/safe-nav-smoke/

# 4. Launch the container with Apptainer and the Python script
apptainer exec --nv --bind /data/emunoz:/data /data/emunoz/imgs/python_full.sif /data/emunoz/envs/dev_env/bin/python projects/single_agent/step_02_structured_sim_collection/data_playback_collection.py