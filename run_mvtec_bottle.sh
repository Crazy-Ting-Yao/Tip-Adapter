#!/bin/bash
#SBATCH --job-name=Tip-Adapter-MVTec-Bottle
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --account=MST113264
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/u8686038/log/Tip-Adapter-MVTec-Bottle_%j.log
#SBATCH --error=/work/u8686038/log/Tip-Adapter-MVTec-Bottle_%j.log

# Run Tip-Adapter on MVTec Bottle dataset
# Dataset: andyqmongo/IVL_OOD_MVTec_bottle
# Train split: 1_shot
# Test split: test

set -e

echo "=========================================="
echo "Running Tip-Adapter on MVTec Bottle Dataset"
echo "=========================================="
echo ""
echo "Dataset: andyqmongo/IVL_OOD_MVTec_bottle"
echo "Train split: 1_shot"
echo "Test split: test"
echo ""

# Activate conda environment
ml load miniconda3
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
CUDA_VISIBLE_DEVICES=${1:-0} python main.py --config configs/mvtec_bottle.yaml

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
