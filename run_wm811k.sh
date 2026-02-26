#!/bin/bash
#SBATCH --job-name=Tip-Adapter-WM811k
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --account=MST114475
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/u8686038/log/Tip-Adapter-WM811k_%j.log
#SBATCH --error=/work/u8686038/log/Tip-Adapter-WM811k_%j.log

# Run Tip-Adapter on WM811k Wafer Map dataset
# Dataset: andyqmongo/IVL_OOD_WM811k
# Train split: 8_shot
# Test split: test

set -e

echo "=========================================="
echo "Running Tip-Adapter on WM811k Dataset"
echo "=========================================="
echo ""
echo "Dataset: andyqmongo/IVL_OOD_WM811k"
echo "Train split: 8_shot"
echo "Test split: test"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tip_adapter || {
    echo "Error: tip_adapter environment not found."
    exit 1
}

# Install datasets library if needed
pip install datasets huggingface_hub -q

# Create data directory
mkdir -p ./data

# Run Tip-Adapter
cd /work/u8686038/Tip-Adapter
CUDA_VISIBLE_DEVICES=${1:-0} python main.py --config configs/wm811k.yaml

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
