#!/bin/bash
#SBATCH --job-name=Tip-Adapter-Pokemon
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --account=MST114475
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/u8686038/log/Tip-Adapter-Pokemon_%j.log
#SBATCH --error=/work/u8686038/log/Tip-Adapter-Pokemon_%j.log

# Run Tip-Adapter on Pokemon dataset
# Train: andyqmongo/IVL_CLS_pokemon_1_shot
# Test: andyqmongo/pokemon_eval_standard

set -e

echo "=========================================="
echo "Running Tip-Adapter on Pokemon Dataset"
echo "=========================================="
echo ""
echo "Train dataset: andyqmongo/IVL_CLS_pokemon_1_shot"
echo "Test dataset: andyqmongo/pokemon_eval_standard"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tip_adapter || {
    echo "Error: tip_adapter environment not found. Creating it..."
    bash /work/u8686038/Tip-Adapter/setup/install_conda.sh tip_adapter 12.1
    conda activate tip_adapter
}

# Install datasets library if needed
pip install datasets huggingface_hub -q

# Create data directory
mkdir -p ./data

# Run Tip-Adapter
cd /work/u8686038/Tip-Adapter
CUDA_VISIBLE_DEVICES=${1:-0} python main.py --config configs/pokemon.yaml

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
