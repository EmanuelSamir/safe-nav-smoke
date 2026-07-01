#!/bin/bash
# =====================================================================
# SLURM CONFIGURATION
# =====================================================================
#SBATCH -p pleiades                         
#SBATCH -w pleiades-0-17                     
#SBATCH --job-name=run_int_cbf         
#SBATCH --time=12:00:00                     
#SBATCH --cpus-per-task=8                         
#SBATCH --mem=32G                           

#SBATCH --output=/home/emunoz/slurm_logs/run_int_cbf_%j.log
#SBATCH --error=/home/emunoz/slurm_logs/run_int_cbf_%j.log

# =====================================================================
# ENTORNO DE EJECUCIÓN
# =====================================================================

mkdir -p /home/emunoz/slurm_logs
mkdir -p /data/emunoz/prefect_home
export PREFECT_HOME="/data/emunoz/prefect_home"

cd /home/emunoz/dev/safe-nav-smoke/

# Launch with the specific YAML config
apptainer exec --nv --bind /data/emunoz:/data /data/emunoz/imgs/python_full.sif /data/emunoz/envs/dev_env/bin/python projects/single_agent/step_06_full_integration/run_integration.py --config config_cbf.yaml
