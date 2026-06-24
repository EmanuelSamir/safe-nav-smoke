#!/bin/bash

# Generar un timestamp único para el output_dir compartido por todos los jobs
TIMESTAMP=$(date +"%Y-%m-%d/%H-%M-%S")
PROJECT_ROOT="/home/emunoz/dev/safe-nav-smoke"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/benchmark/${TIMESTAMP}"

# Crear el directorio base
mkdir -p "$OUTPUT_DIR"
echo "Output directory created: $OUTPUT_DIR"

# Parsear argumentos
if [ "$1" == "--test" ]; then
    echo "Running in TEST MODE: Only evaluating 'nominal' controller for 1 episode."
    CONTROLLERS=("hj_rollout")
    EPISODES=1
else
    # Lista de controladores a evaluar completa
    CONTROLLERS=(
        "nominal"
        "cbf_filter"
        "cbf_rollout"
        "cbf_penalty"
        "hj_filter"
        "hj_rollout"
        "hj_online_rollout"
        "hj_penalty"
    )
    # Número de episodios por defecto
    EPISODES=10
fi

# Enviar un job a SLURM por cada controlador
for CTRL in "${CONTROLLERS[@]}"; do
    echo "Submitting benchmark for controller: $CTRL"
    
    # Creamos un bloque sbatch usando heredoc
    sbatch <<EOF
#!/bin/bash
#SBATCH -p pleiades
#SBATCH -w pleiades-0-23
#SBATCH --job-name=bench_${CTRL}
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --output=/home/emunoz/slurm_logs/bench_${CTRL}_%j.log
#SBATCH --error=/home/emunoz/slurm_logs/bench_${CTRL}_%j.log

export PREFECT_HOME="/data/emunoz/prefect_home"

cd ${PROJECT_ROOT}

apptainer exec --nv --bind /data/emunoz:/data /data/emunoz/imgs/python_full.sif /data/emunoz/envs/dev_env/bin/python projects/4_controller_comparison/benchmark.py --controllers ${CTRL} --output_dir ${OUTPUT_DIR} --episodes ${EPISODES}
EOF

done

echo "================================================================"
echo "All jobs submitted successfully to SLURM!"
echo "Check queue status with: squeue -u emunoz"
echo "Output directory for this benchmark run: $OUTPUT_DIR"
echo "Once all jobs finish, you can analyze the results by running:"
echo "  python projects/4_controller_comparison/analyze.py --output_dir $OUTPUT_DIR"
echo "================================================================"
