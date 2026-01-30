#!/bin/bash

# Setup environment variables
echo "Setting up environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Setup data directory
mkdir -p data

echo "Setting up virtual environment..."
python -m venv .venv
"alias activate_venv='source .venv/bin/activate'" >> ~/.zshrc
activate_venv

echo "Installing dependencies..."
pip install -r requirements.txt
