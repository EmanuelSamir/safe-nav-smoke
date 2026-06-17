#!/bin/bash
# =====================================================================
# CONFIGURACIÓN DE RECURSOS DE SLURM
# =====================================================================
#SBATCH -p pleiades                         # Partición general
#SBATCH -w pleiades-0-23                     # Forzamos el uso del nodo ligero (1080Ti)
#SBATCH --job-name=training_nll_loss         # Nombre del trabajo
#SBATCH --time=12:00:00                     # Aumentamos el límite de tiempo a 8 horas por seguridad
#SBATCH --cpus-per-task=8                   # 8 núcleos de CPU para carga de datos
#SBATCH --gpus=1                            # 1 GPU GTX 1080Ti
#SBATCH --mem=32G                           # 16 GB de memoria RAM

# Corregido: SLURM ya no se confundirá con la virgulilla
#SBATCH --output=/home/emunoz/slurm_logs/physics_%j.log
#SBATCH --error=/home/emunoz/slurm_logs/physics_%j.log

# =====================================================================
# ENTORNO DE EJECUCIÓN
# =====================================================================

# 1. Asegurar carpetas
mkdir -p /home/emunoz/slurm_logs
mkdir -p /data/emunoz/prefect_home

# 2. Evitar el bloqueo de SQLite de Prefect
export PREFECT_HOME="/data/emunoz/prefect_home"

# 3. Moverse al directorio del repositorio basado en tu árbol de directorios
cd /home/emunoz/dev/safe-nav-smoke/

# 4. Lanzar el contenedor con Apptainer y el script de Python
apptainer exec --nv --bind /data/emunoz:/data /data/emunoz/imgs/python_full.sif /data/emunoz/envs/dev_env/bin/python src/training/train_fno.py training.loss.name=nll