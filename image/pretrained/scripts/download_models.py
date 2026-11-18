
"""
Download the three ViT backbones into $HF_HOME so SLURM jobs can run offline.
Run once on a LOGIN node (it has internet; compute nodes may not):
 
    uv run python image/pretrained/scripts/download_models.py
"""
from huggingface_hub import snapshot_download
 
MODELS = [
    "google/vit-base-patch16-224-in21k",   # ViT-B/16,  ~86M params
    "google/vit-large-patch16-224-in21k",  # ViT-L/16, ~307M params
    "google/vit-huge-patch14-224-in21k",   # ViT-H/14, ~632M params
]
 
for m in MODELS:
    path = snapshot_download(m)
    print(f"{m} -> {path}")
 