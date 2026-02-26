#!/bin/bash

# Tip-Adapter Conda Environment Setup Script
# Based on: https://github.com/gaopengcuhk/Tip-Adapter

set -e

ENV_NAME="${1:-tip_adapter}"
PYTHON_VERSION="3.10"
CUDA_VERSION="${2:-12.1}"

echo "=========================================="
echo "Tip-Adapter Environment Setup"
echo "=========================================="
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "CUDA version: ${CUDA_VERSION}"
echo "=========================================="

# Create conda environment
echo "[1/4] Creating conda environment..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "[2/4] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install PyTorch with CUDA support
echo "[3/4] Installing PyTorch..."
if [ "${CUDA_VERSION}" == "12.1" ]; then
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
elif [ "${CUDA_VERSION}" == "12.4" ]; then
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
elif [ "${CUDA_VERSION}" == "11.8" ]; then
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
elif [ "${CUDA_VERSION}" == "cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    echo "Warning: Unknown CUDA version ${CUDA_VERSION}, trying cu121..."
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
fi

# Install requirements
echo "[4/4] Installing dependencies from requirements.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/../requirements.txt"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run Tip-Adapter on ImageNet:"
echo "  CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet.yaml"
echo ""
echo "To run on other datasets:"
echo "  CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dataset.yaml"
echo ""
