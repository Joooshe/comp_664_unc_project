# Project environment variables. Sourced by your Longleaf shell (~/.bashrc) and by every SLURM job.

# repo root = the folder this file lives in, wherever the repo is cloned
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=$PROJ/hf_cache          # Hugging Face model + dataset downloads
export UV_CACHE_DIR=$PROJ/uv_cache     # uv's package cache (torch is big)
export PATH=$HOME/.local/bin:$PATH     # where the uv installer puts the `uv` binary
EOF