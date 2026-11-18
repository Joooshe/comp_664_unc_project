#!/bin/bash
# Submit one job per ViT size. Run from the project folder:
#   cd ~/comp_664_unc_project && bash slurm/submit_all.sh
set -euo pipefail
mkdir -p logs

sbatch --job-name=vitB_informative slurm/extract_and_train.sl google/vit-base-patch16-224-in21k
# sbatch --job-name=vitL_informative slurm/extract_and_train.sl google/vit-large-patch16-224-in21k
# Huge is ~2x slower, so give it more time
# sbatch --job-name=vitH_informative --time=04:00:00 slurm/extract_and_train.sl google/vit-huge-patch14-224-in21k

squeue -u $USER
