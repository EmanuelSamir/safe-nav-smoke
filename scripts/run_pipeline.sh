#!/bin/bash
# =============================================================================
# run_pipeline.sh
# Completo pipeline: genera datos de smoke → corre benchmark de controladores.
#
# Modos:
#   --test      Genera 5 episodios cortos, corre 1 episodio de benchmark por
#               controlador. Ideal para verificar que todo corra sin errores.
#   (sin flag)  Modo overnight: genera 100 ep x 200 steps, corre benchmark
#               completo con todos los controladores.
#
# Uso:
#   bash scripts/run_pipeline.sh           # overnight
#   bash scripts/run_pipeline.sh --test    # test rápido
#   bash scripts/run_pipeline.sh --test --controllers nominal cbf_filter
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/Users/emanuelsamir/Documents/dev/cmu/py-envs/dev/bin/python"

COLLECTION_SCRIPT="${PROJECT_ROOT}/src/1_data_playback_env_collection.py"
BENCHMARK_SCRIPT="${PROJECT_ROOT}/src/experiments/controller_benchmark/benchmark.py"
CONFIG_DIR="${PROJECT_ROOT}/configs"

# ---------------------------------------------------------------------------
# Parse argumentos
# ---------------------------------------------------------------------------
TEST_MODE=false
EXTRA_CONTROLLERS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --controllers)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                EXTRA_CONTROLLERS+=("$1")
                shift
            done
            ;;
        *)
            echo "❌ Unknown argument: $1"
            echo "Usage: $0 [--test] [--controllers ctrl1 ctrl2 ...]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Parámetros según modo
# ---------------------------------------------------------------------------
if [ "$TEST_MODE" = true ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║               🧪  MODO TEST                                 ║"
    echo "║  5 episodios × 20 steps  →  datos en outputs/test/data      ║"
    echo "║  1 episodio de benchmark por controlador                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    NUM_EPISODES=5
    EPISODE_STEPS=20
    DATA_PATH="${PROJECT_ROOT}/outputs/test/data/smoke_env"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/test/benchmark"
    BENCHMARK_EPISODES=1
    BENCHMARK_STEPS=20
    RENDER_MODE="none"
    # En test, evalúa solo nominal + cbf_filter a menos que el usuario pase otros
    if [ ${#EXTRA_CONTROLLERS[@]} -eq 0 ]; then
        CONTROLLERS="nominal cbf_filter"
    else
        CONTROLLERS="${EXTRA_CONTROLLERS[*]}"
    fi
else
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              🌙  MODO OVERNIGHT                             ║"
    echo "║  100 episodios × 200 steps  →  datos en data/               ║"
    echo "║  Benchmark completo con todos los controladores              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    NUM_EPISODES=100
    EPISODE_STEPS=200
    DATA_PATH="${PROJECT_ROOT}/data/smoke_env_100ep"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/benchmark/$(date +%Y-%m-%d/%H-%M-%S)"
    BENCHMARK_EPISODES=100
    BENCHMARK_STEPS=""  # sin límite
    RENDER_MODE="rgb_array"
    if [ ${#EXTRA_CONTROLLERS[@]} -eq 0 ]; then
        CONTROLLERS=""  # todos
    else
        CONTROLLERS="${EXTRA_CONTROLLERS[*]}"
    fi
fi

# ---------------------------------------------------------------------------
# Paso 1: Generación de datos de smoke
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦  PASO 1/2 — Generando datos de smoke"
echo "    Episodios : ${NUM_EPISODES}"
echo "    Steps/ep  : ${EPISODE_STEPS}"
echo "    Output    : ${DATA_PATH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT"

$PYTHON "$COLLECTION_SCRIPT" \
    num_episodes="${NUM_EPISODES}" \
    episode_steps="${EPISODE_STEPS}" \
    output_path="${DATA_PATH}" \
    hydra.run.dir="${OUTPUT_DIR}/hydra/collection"

echo ""
echo "✅  Datos generados correctamente en: ${DATA_PATH}"
echo ""

# ---------------------------------------------------------------------------
# Paso 2: Actualizar config para usar los datos generados (playback)
# ---------------------------------------------------------------------------
# Escribimos un override temporal de playback.yaml apuntando al nuevo path
PLAYBACK_YAML="${CONFIG_DIR}/env/simulator/playback.yaml"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚙️   CONFIGURANDO: apuntando playback → ${DATA_PATH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Backup del playback.yaml original
cp "$PLAYBACK_YAML" "${PLAYBACK_YAML}.bak"

# Sobreescribir con el nuevo path
cat > "$PLAYBACK_YAML" <<EOF
data_path: "${DATA_PATH}"
EOF

echo "    playback.yaml actualizado (backup en .bak)"
echo ""

# Función para restaurar el backup en caso de error
cleanup() {
    if [ -f "${PLAYBACK_YAML}.bak" ]; then
        mv "${PLAYBACK_YAML}.bak" "$PLAYBACK_YAML"
        echo "🔁  playback.yaml restaurado desde backup."
    fi
}
trap cleanup EXIT

# Cambiar config.yaml para apuntar a playback
# (usa sed para modificar solo la línea del simulator)
CONFIG_YAML="${CONFIG_DIR}/config.yaml"
cp "$CONFIG_YAML" "${CONFIG_YAML}.bak"

sed -i.tmp 's|env/simulator@simulator:.*|env/simulator@simulator: playback|' "$CONFIG_YAML"
rm -f "${CONFIG_YAML}.tmp"

echo "    config.yaml actualizado a modo playback"
echo ""

# ---------------------------------------------------------------------------
# Paso 3: Benchmark
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀  PASO 2/2 — Corriendo benchmark de controladores"
echo "    Episodios : ${BENCHMARK_EPISODES}"
echo "    Output    : ${OUTPUT_DIR}"
echo "    Render    : ${RENDER_MODE}"
if [ -n "$CONTROLLERS" ]; then
    echo "    Controllers: ${CONTROLLERS}"
else
    echo "    Controllers: TODOS"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

mkdir -p "$OUTPUT_DIR"

# Construir argumentos del benchmark
BENCHMARK_ARGS=(
    --episodes "${BENCHMARK_EPISODES}"
    --output_dir "${OUTPUT_DIR}"
    --render "${RENDER_MODE}"
)

if [ -n "$BENCHMARK_STEPS" ]; then
    BENCHMARK_ARGS+=(--steps "${BENCHMARK_STEPS}")
fi

if [ -n "$CONTROLLERS" ]; then
    BENCHMARK_ARGS+=(--controllers $CONTROLLERS)
fi

$PYTHON "$BENCHMARK_SCRIPT" "${BENCHMARK_ARGS[@]}"

# ---------------------------------------------------------------------------
# Restaurar configs originales (también lo hace el trap al salir)
# ---------------------------------------------------------------------------
if [ -f "${CONFIG_YAML}.bak" ]; then
    mv "${CONFIG_YAML}.bak" "$CONFIG_YAML"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅  Pipeline completado exitosamente                       ║"
echo "║     Resultados: ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
