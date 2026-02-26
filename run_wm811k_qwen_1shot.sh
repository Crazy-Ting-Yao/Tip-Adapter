#!/bin/bash
#SBATCH --job-name=Tip-WM811k-Qwen-1s
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --account=MST114475
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/u8686038/log/Tip-Adapter-WM811k-Qwen-1shot_%j.log
#SBATCH --error=/work/u8686038/log/Tip-Adapter-WM811k-Qwen-1shot_%j.log

# Run Tip-Adapter on WM811k (1-shot) with Qwen2.5-VL-7B-Instruct backbone
# Dataset: andyqmongo/IVL_OOD_WM811k (8_shot base, few-shot sampler to 1-shot)

echo "=========================================="
echo "Tip-Adapter on WM811k (Qwen2.5-VL-7B, 1-shot)"
echo "=========================================="
echo ""

source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate tip_adapter || {
    echo "Error: tip_adapter environment not found."
    exit 1
}

export HF_TOKEN=""
pip install datasets huggingface_hub transformers accelerate -q
mkdir -p ./data

cd /work/u8686038/Tip-Adapter
CUDA_VISIBLE_DEVICES=printf '%s
' "#!/bin/bash" "#SBATCH --job-name=Tip-WM811k-Qwen-1s" "#SBATCH --partition=dev" "#SBATCH --time=02:00:00" "#SBATCH --account=MST114475" "#SBATCH --nodes=1" "#SBATCH --gpus-per-node=1" "#SBATCH --cpus-per-task=4" "#SBATCH --output=/work/u8686038/log/Tip-Adapter-WM811k-Qwen-1shot_%j.log" "#SBATCH --error=/work/u8686038/log/Tip-Adapter-WM811k-Qwen-1shot_%j.log" "" "# Run Tip-Adapter on WM811k (1-shot) with Qwen2.5-VL-7B-Instruct backbone" "# Dataset: andyqmongo/IVL_OOD_WM811k (8_shot base, few-shot sampler to 1-shot)" "" "set -e" "" "echo \"==========================================\"" "echo \"Tip-Adapter on WM811k (Qwen2.5-VL-7B, 1-shot)\"" "echo \"==========================================\"" "echo \"\"" "" "source $(conda info --base)/etc/profile.d/conda.sh" "conda activate tip_adapter || {" "    echo \"Error: tip_adapter environment not found.\"" "    exit 1" "}" "" "export HF_TOKEN=${HF_TOKEN:-\"\"}" "pip install datasets huggingface_hub transformers accelerate -q" "mkdir -p ./data" "" "cd /work/u8686038/Tip-Adapter" "CUDA_VISIBLE_DEVICES=${1:-0} python main.py --config configs/wm811k_qwen_1shot.yaml" "" "echo \"\"" "echo \"==========================================\"" "echo \"Done.\"" "echo \"==========================================\"" > /work/u8686038/Tip-Adapter/run_wm811k_qwen_1shot.sh python main.py --config configs/wm811k_qwen_1shot.yaml

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
