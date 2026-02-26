#!/bin/bash
#SBATCH --job-name=Tip-Pokemon-Qwen
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --account=MST114475
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/u8686038/log/Tip-Adapter-Pokemon-Qwen_%j.log
#SBATCH --error=/work/u8686038/log/Tip-Adapter-Pokemon-Qwen_%j.log

# Run Tip-Adapter on Pokemon with Qwen2.5-VL-7B-Instruct backbone
# Train: andyqmongo/IVL_CLS_pokemon_1_shot, Test: andyqmongo/pokemon_eval_standard

set -e

echo "=========================================="
echo "Tip-Adapter on Pokemon (Qwen2.5-VL-7B)"
echo "=========================================="
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate tip_adapter || {
    echo "Error: tip_adapter environment not found."
    exit 1
}

export HF_TOKEN=${HF_TOKEN:-""}
pip install datasets huggingface_hub transformers accelerate -q
mkdir -p ./data

cd /work/u8686038/Tip-Adapter
CUDA_VISIBLE_DEVICES=${1:-0} python main.py --config configs/pokemon_qwen.yaml

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
