
"""
STEP 2: Extract image features with a FROZEN Vision Transformer.
Task: CrisisMMD informative (agreed labels, event-based split from make_split.py)
 
Big picture
-----------
In the feature-based (frozen backbone) approach the pretrained ViT is never updated,
so every image always maps to the same embedding. We run the ViT ONCE, save each
image's embedding, and later train a tiny linear classifier on those (train_head.py).
 
    image --> [frozen ViT] --> feature vector --> saved to image/pretrained/results/feats/*.pt
 
Backbones (ImageNet-21k pretrained, same recipe, different sizes):
    google/vit-base-patch16-224-in21k    ~86M   hidden 768
    google/vit-large-patch16-224-in21k   ~307M  hidden 1024
    google/vit-huge-patch14-224-in21k    ~632M  hidden 1280
 
Usage
-----
    uv run python image/pretrained/scripts/extract_features.py --model google/vit-large-patch16-224-in21k
"""
import argparse
import json
import os
import time
 
import torch
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
 
 
# ---------------------------------------------------------------------------
# Dataset: wraps one HF split and returns (pixel tensor, label string, image id)
# ---------------------------------------------------------------------------
class HFImages(Dataset):
    def __init__(self, hf_split, processor):
        self.ds = hf_split
        # The processor applies the preprocessing the ViT was trained with
        # (resize, crop, normalize), so our inputs match what it saw in pretraining.
        self.proc = processor
 
    def __len__(self):
        return len(self.ds)
 
    def __getitem__(self, i):
        ex = self.ds[i]                      # decodes just this one image
        img = ex["image"]
        if img is None:                      # guard against missing images
            img = Image.new("RGB", (224, 224))
            print(f"[warn] missing image for {ex['image_id']}")
        img = img.convert("RGB")             # some images are grayscale/RGBA; force 3 channels
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]   # (3, 224, 224)
        return px, ex["label"], ex["image_id"]
 
 
# ---------------------------------------------------------------------------
# Feature extraction loop
# ---------------------------------------------------------------------------
@torch.no_grad()  # no gradients: the backbone is frozen. Saves a lot of memory.
def extract(model, loader, device, pool):
    feats, labels, ids = [], [], []
    for step, (px, y, iid) in enumerate(loader):
        px = px.to(device, dtype=model.dtype, non_blocking=True)
        h = model(pixel_values=px).last_hidden_state
        # h has shape (B, 1 + num_patches, D):
        #   h[:, 0]  = [CLS] token, a learned summary of the whole image
        #   h[:, 1:] = one vector per image patch
        cls = h[:, 0]
        if pool == "cls":
            f = cls                                              # (B, D)
        else:
            # CLS + average of patch tokens: adds more spatial detail.  (B, 2D)
            f = torch.cat([cls, h[:, 1:].mean(dim=1)], dim=-1)
        feats.append(f.float().cpu())
        labels += list(y)
        ids += list(iid)
        if step % 20 == 0:
            print(f"  batch {step}/{len(loader)}", flush=True)
    return torch.cat(feats), labels, ids
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/informative_agreed_eventsplit")
    ap.add_argument("--model", default="google/vit-large-patch16-224-in21k")
    ap.add_argument("--pool", default="cls+mean", choices=["cls", "cls+mean"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8, help="CPU processes decoding/resizing images")
    ap.add_argument("--out_dir", default="image/pretrained/results/feats")
    ap.add_argument("--limit", type=int, default=None,
                    help="only use the first N images per split (quick local tests)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    args = ap.parse_args()
 
    # cuda on Longleaf; mps on Apple-silicon laptops; otherwise cpu
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
 
    # --- Load the dataset made by make_split.py (works offline) --------------
    dd = load_from_disk(args.data_dir)
    if args.limit:  # keep only the first N rows of each split
        dd = DatasetDict({s: dd[s].select(range(min(args.limit, len(dd[s])))) for s in dd})
    print("splits:", dd.num_rows)
 
    # --- Load the pretrained backbone ------------------------------------------
    # AutoModel = the ViT WITHOUT a classification head; it just outputs hidden states.
    # float16 on GPU halves memory and is fine for inference.
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.eval().to(device)                   # eval() disables dropout -> deterministic features
 
    # --- Freeze it (and verify) ----------------------------------------------------
    for p in model.parameters():
        p.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model}: {total/1e6:.0f}M params, {trainable} trainable")
 
    # --- Extract + save each split ---------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.model.split("/")[-1]
    prefix = f"informative_{tag}_{args.pool}"
    timing, t_all = {}, time.time()
    for split in ["train", "dev", "test"]:
        dl = DataLoader(HFImages(dd[split], processor), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=(device == "cuda"))
        print(f"{split}: {len(dl.dataset)} images")
        t0 = time.time()
        X, y, ids = extract(model, dl, device, args.pool)
        secs = time.time() - t0
        timing[split] = {"images": len(y), "seconds": secs, "images_per_second": len(y) / secs}
        out = os.path.join(args.out_dir, f"{prefix}_{split}.pt")
        torch.save({"X": X, "y": y, "ids": ids, "model": args.model, "pool": args.pool}, out)
        print(f"{split}: features {tuple(X.shape)} in {secs:.1f}s -> {out}")
 
    # Save extraction details next to the features; train_head.py copies these
    # into its metrics json so the write-up has backbone size + runtime in one place.
    meta = {
        "model": args.model, "pool": args.pool, "backbone_params": total,
        "backbone_trainable_params": trainable, "feature_dim": X.shape[1],
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "dtype": str(model.dtype), "batch_size": args.batch_size, "limit": args.limit,
        "splits": timing, "total_seconds": time.time() - t_all,
    }
    with open(os.path.join(args.out_dir, f"{prefix}_extract_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"total extraction time: {meta['total_seconds']:.1f}s")
 
 
if __name__ == "__main__":
    main()
 