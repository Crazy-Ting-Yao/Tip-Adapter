#!/bin/bash
#SBATCH --job-name=Tip-Retinal-LLaVA-8s
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --account=MST113264
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/u8686038/log/Tip-Adapter-Retinal-LLaVA-8shot_%j.log
#SBATCH --error=/work/u8686038/log/Tip-Adapter-Retinal-LLaVA-8shot_%j.log

# Run Tip-Adapter on Retinal (8-shot) with LLaVA v1.6 Mistral 7B backbone

set -e

echo "=========================================="
echo "Tip-Adapter on Retinal (LLaVA v1.6 Mistral 7B, 8-shot)"
echo "=========================================="
echo ""

ml load miniconda3
conda activate tip_adapter || { echo "Error: tip_adapter environment not found."; exit 1; }

# Use HF_TOKEN from environment (set before sbatch for private HuggingFace datasets)
export HF_TOKEN=${HF_TOKEN:-""}
pip install datasets huggingface_hub transformers accelerate -q
mkdir -p ./data

cd /work/u8686038/Tip-Adapter
CUDA_VISIBLE_DEVICES=${1:-0} python main.py --config configs/retinal_llava_8shot.yaml

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
