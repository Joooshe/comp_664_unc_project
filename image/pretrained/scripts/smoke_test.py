
"""
Local smoke test: run on your LAPTOP before submitting anything to Longleaf.
Uses the smallest model (ViT-Base, 86M) and a handful of real images. Takes a few
minutes on CPU or an Apple-silicon Mac.
 
Prerequisite (once, on the laptop, needs internet + a few GB of disk):
    uv run python common/make_split.py
 
Run from the repo root:
    uv run python image/pretrained/smoke_test.py
 
What it checks
--------------
1. The model loads and the frozen backbone reports 0 trainable params.
2. FREEZING REALLY WORKS: we attach a linear head to the backbone, run a few real
   optimizer steps end to end, and confirm the backbone weights are bit-for-bit
   unchanged, have no gradients, while the head's weights DID change.
3. Features are deterministic (same image -> same vector) and have the right shape.
4. The real pipeline runs: extract_features.py on 32 images/split, then
   train_head.py for a few epochs, producing a metrics json.
Every check prints PASS or raises an error.
"""
import os
import subprocess
import sys
import tempfile
 
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoImageProcessor, AutoModel
 
MODEL = "google/vit-base-patch16-224-in21k"
DATA_DIR = "data/informative_agreed_eventsplit"
HERE = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
 
 
def ok(msg):
    print(f"  PASS  {msg}")
 
 
def main():
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"{DATA_DIR} not found. Run `uv run python common/make_split.py` first "
                 f"(from the repo root).")
    print(f"device: {device}\n")
 
    dd = load_from_disk(DATA_DIR)
    imgs = [dd["train"][i]["image"].convert("RGB") for i in range(8)]
    labels = torch.tensor([0 if dd["train"][i]["label"] == "informative" else 1 for i in range(8)])
 
    # ---- 1. Load + freeze --------------------------------------------------------------
    print("1. load + freeze")
    proc = AutoImageProcessor.from_pretrained(MODEL)
    backbone = AutoModel.from_pretrained(MODEL).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    n_total = sum(p.numel() for p in backbone.parameters())
    n_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    assert n_train == 0, f"{n_train} backbone params still trainable"
    ok(f"{n_total/1e6:.0f}M backbone params, 0 trainable")
 
    # ---- 2. Freezing holds during real training steps --------------------------------------
    print("2. freezing holds during training")
    px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
    head = nn.Linear(backbone.config.hidden_size, 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)   # optimizer only sees the head
 
    before_backbone = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
    before_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
 
    backbone.eval()
    for _ in range(3):  # a few steps through the FULL model: image -> backbone -> head
        logits = head(backbone(pixel_values=px).last_hidden_state[:, 0])
        loss = nn.functional.cross_entropy(logits, labels.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
 
    assert all(p.grad is None for p in backbone.parameters()), "backbone received gradients"
    ok("backbone has no gradients")
    changed = [k for k, v in backbone.state_dict().items() if not torch.equal(v, before_backbone[k])]
    assert not changed, f"backbone weights changed: {changed[:3]}"
    ok("backbone weights identical after training steps")
    assert any(not torch.equal(v, before_head[k]) for k, v in head.state_dict().items()), "head did not update"
    ok("head weights changed (it is learning)")
 
    # ---- 3. Deterministic features, right shape -------------------------------------------
    print("3. features")
    with torch.no_grad():
        h1 = backbone(pixel_values=px[:2]).last_hidden_state
        h2 = backbone(pixel_values=px[:2]).last_hidden_state
    assert torch.allclose(h1, h2), "features differ between runs (dropout on? forgot eval()?)"
    ok("same image -> same features (eval mode)")
    n_tokens = 1 + (224 // backbone.config.patch_size) ** 2
    assert h1.shape == (2, n_tokens, backbone.config.hidden_size), h1.shape
    ok(f"hidden states shape {tuple(h1.shape)} = (batch, 1 CLS + {n_tokens-1} patches, {backbone.config.hidden_size})")
 
    # ---- 4. Real scripts, tiny run ----------------------------------------------------------
    print("4. real pipeline on 32 images per split")
    tag = MODEL.split("/")[-1]
    with tempfile.TemporaryDirectory() as tmp:
        feats, res = os.path.join(tmp, "feats"), os.path.join(tmp, "res")
        subprocess.run([sys.executable, os.path.join(HERE, "extract_features.py"),
                        "--model", MODEL, "--limit", "32", "--batch_size", "16",
                        "--workers", "0", "--out_dir", feats, "--device", device], check=True)
        ok("extract_features.py")
        subprocess.run([sys.executable, os.path.join(HERE, "train_head.py"),
                        "--feat_dir", feats, "--results_dir", res,
                        "--prefix", f"informative_{tag}_cls+mean",
                        "--epochs", "5", "--lrs", "1e-3", "--wds", "0"], check=True)
        assert os.path.exists(os.path.join(res, "metrics", f"informative_{tag}_cls+mean.json"))
        ok("train_head.py wrote a metrics json")
 
    print("\nAll checks passed. Safe to push and run the full jobs on Longleaf.")
 
 
if __name__ == "__main__":
    main()
 