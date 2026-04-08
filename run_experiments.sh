#!/bin/bash

# Ruta al virtual env utilizado en este proyecto
PYTHON_BIN="/Users/emanuelsamir/Documents/dev/cmu/py-envs/dev/bin/python"

echo "================================================="
echo "🚀 INICIANDO EXPERIMENTO 1: FNO-3D BASELINE      "
echo "================================================="
# FNO utilizará un sample_ratio pequeño y compensará con 100 epochs
$PYTHON_BIN 2_train_fno_3d.py \
    training.epochs=100 \
    data.sample_ratio=0.05

echo ""
echo "================================================="
echo "🚀 INICIANDO EXPERIMENTO 2: DREAMER V4 LATENT    "
echo "================================================="
# Dreamer iterará velozmente pudiendo asimilar el 100% (1.0) del corpus en el mismo batch size
$PYTHON_BIN 3_train_dreamer_v4.py \
    training.epochs=100 \
    data.sample_ratio=1.0

echo "✅ Pipeline completado."
