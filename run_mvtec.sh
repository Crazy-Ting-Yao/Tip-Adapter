#!/bin/bash
#SBATCH --job-name=Tip-Adapter-MVTec
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --account=MST114475
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/u8686038/log/Tip-Adapter-MVTec_%j.log
#SBATCH --error=/work/u8686038/log/Tip-Adapter-MVTec_%j.log

# Run Tip-Adapter on MVTec AD dataset
# Usage: sbatch run_mvtec.sh [object_name]
# Objects: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
# Dataset: andyqmongo/IVL_OOD_MVTec_{object}
# Train split: 8_shot, Test split: test

set -e

OBJ=${1:-bottle}
VALID="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
if [[ ! " $VALID " =~ " $OBJ " ]]; then
    echo "Invalid object: $OBJ. Choose from: $VALID"
    exit 1
fi

echo "=========================================="
echo "Running Tip-Adapter on MVTec ($OBJ)"
echo "=========================================="
echo ""
echo "Dataset: andyqmongo/IVL_OOD_MVTec_$OBJ"
echo "Train split: 1_shot"
echo "Test split: test"
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate tip_adapter || { echo "Error: tip_adapter environment not found."; exit 1; }
pip install datasets huggingface_hub -q
mkdir -p ./data

cd /work/u8686038/Tip-Adapter
CONFIG="configs/mvtec_${OBJ}.yaml"
CUDA_VISIBLE_DEVICES=${2:-0} python main.py --config "$CONFIG"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
