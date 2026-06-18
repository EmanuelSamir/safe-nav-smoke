#!/bin/bash
# =====================================================================
# CONFIGURACIÓN DE RECURSOS DE SLURM
# =====================================================================
#SBATCH -p pleiades                         # Partición general
#SBATCH -w pleiades-0-23                     # Forzamos el uso del nodo ligero (1080Ti)
#SBATCH --job-name=playback_env              # Nombre del trabajo
#SBATCH --time=12:00:00                     # Límite de tiempo
#SBATCH --cpus-per-task=4                   # 8 núcleos de CPU
#SBATCH --gpus=1                            # 1 GPU GTX 1080Ti
#SBATCH --mem=16G                           # Memoria RAM

# Redirección de logs
#SBATCH --output=/home/emunoz/slurm_logs/playback_env_%j.log
#SBATCH --error=/home/emunoz/slurm_logs/playback_env_%j.log

# =====================================================================
# ENTORNO DE EJECUCIÓN
# =====================================================================

# 1. Asegurar carpetas
mkdir -p /home/emunoz/slurm_logs
mkdir -p /data/emunoz/prefect_home

# 2. Evitar el bloqueo de SQLite de Prefect
export PREFECT_HOME="/data/emunoz/prefect_home"

# 3. Moverse al directorio del repositorio
cd /home/emunoz/dev/safe-nav-smoke/

# 4. Lanzar el contenedor con Apptainer y el script de Python (test=false para no visualizar y guardar)
apptainer exec --nv --bind /data/emunoz:/data /data/emunoz/imgs/python_full.sif /data/emunoz/envs/dev_env/bin/python src/1_data_playback_env_collection.py test=false
