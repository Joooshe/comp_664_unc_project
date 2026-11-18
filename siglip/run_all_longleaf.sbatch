#!/bin/bash
#SBATCH --job-name=crisismmd-all
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=crisismmd-all-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory with sbatch}"

module load python/3.12.4
python=.venv/bin/python
if [[ ! -x "$python" || ! -f train_multimodal.py ]]; then
    echo "Expected .venv/bin/python and train_multimodal.py in $PWD" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

"$python" -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"; print(torch.cuda.get_device_name(0))'

failed=()
for model in clip bridgetower blip flava visualbert; do
    echo "[$(date)] Starting $model"
    if "$python" -u train_multimodal.py --model "$model" --device cuda "$@"; then
        echo "[$(date)] Finished $model"
    else
        failed+=("$model")
        echo "[$(date)] Failed $model" >&2
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo "Failed models: ${failed[*]}" >&2
    exit 1
fi
