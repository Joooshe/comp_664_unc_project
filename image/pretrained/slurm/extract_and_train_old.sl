#!/bin/bash
# Frozen ViT features + linear head for ONE backbone.
#
# Submit from the project folder:
#   cd ~/comp_664_unc_project
#   sbatch slurm/extract_and_train.sl                                     # default: ViT-L
#   sbatch slurm/extract_and_train.sl google/vit-base-patch16-224-in21k   # pick a model
#
# Or submit all three sizes at once: bash slurm/submit_all.sh
#
#SBATCH --job-name=vit_informative
#SBATCH --partition=l40-gpu          # 48 GB GPUs; a100-gpu or rtx6k_gpu (--qos=gpu_rtx6k) also work
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # matches --workers 8 for image decoding
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out      # %x = job name, %j = job id

set -euo pipefail                    # stop at the first error
cd $HOME/comp_664_unc_project
source env.sh
export HF_HUB_OFFLINE=1              # use cached models/data only; never download on compute nodes
export HF_DATASETS_OFFLINE=1

MODEL=${1:-google/vit-large-patch16-224-in21k}
POOL=cls+mean
TAG=$(basename $MODEL)

# ---- record what this run used (reproducibility) ----
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "model $MODEL  pool $POOL"
echo "git commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run --frozen python -c "import torch, transformers, datasets; print('torch', torch.__version__, '| transformers', transformers.__version__, '| datasets', datasets.__version__)"

# ---- stage 1: frozen ViT -> saved features ----
uv run --frozen python scripts/extract_features.py --model $MODEL --pool $POOL --workers 8

# ---- stage 2: linear head only ----
uv run --frozen python scripts/train_head.py --prefix informative_${TAG}_${POOL}

echo "done at $(date)"
