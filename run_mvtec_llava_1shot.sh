#!/bin/bash
#SBATCH --job-name=Tip-MVTec-LLaVA-1s
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --account=MST113264
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/u8686038/log/Tip-Adapter-MVTec-LLaVA-1shot_%j.log
#SBATCH --error=/work/u8686038/log/Tip-Adapter-MVTec-LLaVA-1shot_%j.log

# Run Tip-Adapter on MVTec AD (1-shot) with LLaVA v1.6 Mistral 7B backbone
# Usage: sbatch run_mvtec_llava_1shot.sh [object_name] [gpu_id]
# Objects: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

set -e

OBJ=${1:-bottle}
VALID="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
if [[ ! " $VALID " =~ " $OBJ " ]]; then
    echo "Invalid object: $OBJ. Choose from: $VALID"
    exit 1
fi

echo "=========================================="
echo "Tip-Adapter on MVTec ($OBJ) (LLaVA v1.6 Mistral 7B, 1-shot)"
echo "=========================================="
echo ""

ml load miniconda3
conda activate tip_adapter || { echo "Error: tip_adapter environment not found."; exit 1; }

# Use HF_TOKEN from environment (set before sbatch for private HuggingFace datasets)
export HF_TOKEN=${HF_TOKEN:-""}
pip install datasets huggingface_hub transformers accelerate -q
mkdir -p ./data

cd /work/u8686038/Tip-Adapter
CONFIG="configs/mvtec_${OBJ}_llava_1shot.yaml"
CUDA_VISIBLE_DEVICES=${2:-0} python main.py --config "$CONFIG"

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
