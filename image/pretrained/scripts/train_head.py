
"""
STEP 3: Train ONLY a linear classification layer on the saved ViT features.
Task: CrisisMMD informative (binary: informative vs not_informative)
 
Big picture
-----------
The whole model we train is one linear layer:
 
    logits = W @ x + b       W: (2 x feature_dim)
 
This is a "linear probe": it measures how linearly separable the classes already
are in the frozen ViT's feature space. Each run takes seconds, so we sweep a few
learning rates and weight decays.
 
Model selection uses DEV (earthquakes) only. TEST (California wildfires) is
evaluated exactly once at the end with the chosen settings.
 
Outputs (everything you need for the write-up)
----------------------------------------------
  results/heads/<prefix>_head.pt        BEST head (best dev macro-F1) + LAST-epoch head
  results/predictions/<prefix>_{dev,test}.csv   one row per image: true, predicted, prob
  results/metrics/<prefix>.json         test/dev scores, per-class report, confusion matrix,
                                        loss function + optimizer settings, timing,
                                        and the full per-epoch training history
  -> run summarize.py to turn these into a results table + training-curve plots
 
Usage
-----
    uv run python image/pretrained/scripts/train_head.py --prefix informative_vit-large-patch16-224-in21k_cls+mean
"""
import argparse
import json
import os
import platform
import time
 
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
 
# Fixed class order so index 0/1 always means the same thing across runs.
CLASSES = ["informative", "not_informative"]
C2I = {c: i for i, c in enumerate(CLASSES)}
 
 
def load_split(feat_dir, prefix, split):
    """Load saved features; turn string labels into integer ids (0 or 1); keep image ids."""
    d = torch.load(os.path.join(feat_dir, f"{prefix}_{split}.pt"))
    return d["X"], torch.tensor([C2I[c] for c in d["y"]]), d["ids"]
 
 
def save_predictions(head, X, y, ids, path, model_name, device):
    """One row per image, in the SHARED format every model in the group writes:
        image_id, true_label, pred_label, prob_informative, correct, model
    Joining these files on image_id lets you compare image / text / multimodal per post."""
    import pandas as pd
    head.eval()
    with torch.no_grad():
        p = head(X.to(device)).softmax(dim=1).cpu()
    pred = p.argmax(dim=1)
    pd.DataFrame({
        "image_id": ids,
        "true_label": [CLASSES[i] for i in y.tolist()],
        "pred_label": [CLASSES[i] for i in pred.tolist()],
        "prob_informative": p[:, C2I["informative"]].numpy().round(4),
        "correct": (pred == y).numpy(),
        "model": model_name,
    }).to_csv(path, index=False)
 
 
def evaluate(head, X, y, device, loss_fn=None):
    head.eval()
    with torch.no_grad():
        logits = head(X.to(device))
        pred = logits.argmax(dim=1).cpu()               # class with the higher logit
        loss = loss_fn(logits, y.to(device)).item() if loss_fn else None
    return {
        "pred": pred,
        "loss": loss,
        "acc": accuracy_score(y, pred),
        # macro-F1: each class counts equally -> is the minority class actually learned?
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
        # weighted-F1: weighted by class frequency -> what most CrisisMMD papers report
        "weighted_f1": f1_score(y, pred, average="weighted", labels=[0, 1], zero_division=0),
    }
 
 
def train_one(Xtr, ytr, Xdv, ydv, lr, wd, epochs, bs, class_weight, device, verbose=False):
    """Train a fresh linear head for one (lr, wd) setting.
    Returns the best-dev weights, the last-epoch weights, and the per-epoch history."""
    torch.manual_seed(0)  # same init for every config so comparisons are fair
    t0 = time.time()
 
    # THE ONLY TRAINABLE LAYER. No softmax: CrossEntropyLoss applies it internally.
    head = nn.Linear(Xtr.shape[1], len(CLASSES)).to(device)
 
    # AdamW = Adam + decoupled weight decay (regularization on W).
    # Cosine schedule lowers the learning rate smoothly to ~0 by the last epoch.
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
 
    # Optional: weight the rarer class more, so mistakes on it cost more.
    weight = None
    if class_weight:
        counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
        weight = (counts.sum() / (len(CLASSES) * counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
 
    # The whole train set fits in memory, so slice mini-batches by hand.
    Xg, yg = Xtr.to(device), ytr.to(device)
    best = {"macro_f1": -1.0}
    history = []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xg), device=device)       # reshuffle each epoch
        total, n = 0.0, 0
        for i in range(0, len(Xg), bs):
            idx = perm[i:i + bs]
            loss = loss_fn(head(Xg[idx]), yg[idx])
            opt.zero_grad()
            loss.backward()          # gradients only reach the linear layer
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        sched.step()
 
        # Average train loss over the WHOLE epoch (smoother than the last batch),
        # plus dev loss and dev scores, logged every epoch for the training curves.
        m = evaluate(head, Xdv, ydv, device, loss_fn)
        history.append({"epoch": ep, "train_loss": total / n, "dev_loss": m["loss"],
                        "dev_acc": m["acc"], "dev_macro_f1": m["macro_f1"]})
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  train loss {total/n:.4f}  dev loss {m['loss']:.4f}  "
                  f"dev acc {m['acc']:.4f}  dev macroF1 {m['macro_f1']:.4f}", flush=True)
 
        # keep the checkpoint with the best dev score (early stopping by checkpoint)
        if m["macro_f1"] > best["macro_f1"]:
            best = {**m, "epoch": ep,
                    "state": {k: v.detach().clone().cpu() for k, v in head.state_dict().items()}}
 
    best["last_state"] = {k: v.detach().clone().cpu() for k, v in head.state_dict().items()}
    best["history"] = history
    best["train_seconds"] = time.time() - t0
    return best
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="image/pretrained/results/feats")
    ap.add_argument("--prefix", required=True, help="e.g. informative_vit-large-patch16-224-in21k_cls+mean")
    ap.add_argument("--results_dir", default="image/pretrained/results")
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    ap.add_argument("--wds", type=float, nargs="+", default=[0.0, 1e-4, 1e-2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--class_weight", action="store_true",
                    help="reweight loss by inverse class frequency (try if macro-F1 << accuracy)")
    ap.add_argument("--verbose", action="store_true", help="print loss / dev scores every 10 epochs")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
 
    # ---- 1. Load cached features ---------------------------------------------
    Xtr, ytr, _ = load_split(args.feat_dir, args.prefix, "train")
    Xdv, ydv, ids_dv = load_split(args.feat_dir, args.prefix, "dev")
    Xte, yte, ids_te = load_split(args.feat_dir, args.prefix, "test")
    print(f"train {tuple(Xtr.shape)}  dev {tuple(Xdv.shape)}  test {tuple(Xte.shape)}")
 
    # ---- 2. Majority-class baseline: any real model must beat this ------------
    counts = torch.bincount(ytr, minlength=2)
    print("train class counts:", dict(zip(CLASSES, counts.tolist())))
    majority = counts.argmax().item()
    baseline_acc = (yte == majority).float().mean().item()
    print(f"majority-class baseline test acc: {baseline_acc:.4f}")
 
    # ---- 3. Standardize features (stats from TRAIN only, no leakage) -----------
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xdv, Xte = (Xtr - mu) / sd, (Xdv - mu) / sd, (Xte - mu) / sd
 
    # ---- 4. Hyperparameter sweep, selected on DEV -------------------------------
    sweep, best = [], {"macro_f1": -1.0}
    for lr in args.lrs:
        for wd in args.wds:
            r = train_one(Xtr, ytr, Xdv, ydv, lr, wd, args.epochs,
                          args.batch_size, args.class_weight, device, verbose=args.verbose)
            print(f"lr={lr:<7g} wd={wd:<7g} best dev: acc={r['acc']:.4f} "
                  f"macroF1={r['macro_f1']:.4f} (epoch {r['epoch']}, {r['train_seconds']:.1f}s)")
            sweep.append({"lr": lr, "wd": wd, "dev_acc": r["acc"], "dev_macro_f1": r["macro_f1"],
                          "best_epoch": r["epoch"], "train_seconds": r["train_seconds"]})
            if r["macro_f1"] > best["macro_f1"]:
                best = {**r, "lr": lr, "wd": wd}
    print(f"\nchosen: lr={best['lr']:g}, wd={best['wd']:g}, epoch {best['epoch']}, "
          f"dev macroF1={best['macro_f1']:.4f}")
 
    # Convergence hint for the write-up / for deciding whether to train longer
    if best["epoch"] >= 0.9 * args.epochs:
        print("[hint] best epoch is near the end: dev score was still improving, try more --epochs")
    elif best["epoch"] <= 0.05 * args.epochs:
        print("[hint] best epoch is very early: lr may be too high or the head overfits quickly")
 
    # ---- 5. Final evaluation on TEST (once) --------------------------------------
    head = nn.Linear(Xtr.shape[1], len(CLASSES)).to(device)
    head.load_state_dict(best["state"])
    m = evaluate(head, Xte, yte, device)
    print(f"\nTEST  acc={m['acc']:.4f}  macroF1={m['macro_f1']:.4f}  weightedF1={m['weighted_f1']:.4f}")
    print(classification_report(yte, m["pred"], labels=[0, 1], target_names=CLASSES,
                                digits=4, zero_division=0))
    cm = confusion_matrix(yte, m["pred"], labels=[0, 1])
    print("confusion matrix [rows=true, cols=pred]:\n", cm)
    per_class = classification_report(yte, m["pred"], labels=[0, 1], target_names=CLASSES,
                                      output_dict=True, zero_division=0)
 
    # ---- 6. Save heads ------------------------------------------------------------
    # "state" = best-dev checkpoint (what you report). "last_state" = final epoch, kept for comparison.
    for sub in ("heads", "metrics", "predictions"):
        os.makedirs(os.path.join(args.results_dir, sub), exist_ok=True)
    torch.save({"state": best["state"], "last_state": best["last_state"], "classes": CLASSES,
                "mu": mu, "sd": sd, "lr": best["lr"], "wd": best["wd"], "epoch": best["epoch"]},
               os.path.join(args.results_dir, "heads", f"{args.prefix}_head.pt"))
 
    # Per-image predictions for dev + test (for error analysis / comparing models)
    for split, X_, y_, ids_ in [("dev", Xdv, ydv, ids_dv), ("test", Xte, yte, ids_te)]:
        out_p = os.path.join(args.results_dir, "predictions", f"{args.prefix}_{split}.csv")
        save_predictions(head, X_, y_, ids_, out_p, f"image_pretrained:{args.prefix}", device)
        print("saved predictions ->", out_p)
 
    # ---- 7. Save a metrics file with everything the write-up needs -------------------
    meta_path = os.path.join(args.feat_dir, f"{args.prefix}_extract_meta.json")
    extraction = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    metrics = {
        "prefix": args.prefix,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "data": {"n_train": len(ytr), "n_dev": len(ydv), "n_test": len(yte),
                 "train_class_counts": dict(zip(CLASSES, counts.tolist())), "feature_dim": Xtr.shape[1]},
        "setup": {
            "model": "linear head (nn.Linear) on frozen features",
            "trainable_params": Xtr.shape[1] * len(CLASSES) + len(CLASSES),
            "loss": "cross-entropy (softmax)" + (" with inverse-frequency class weights" if args.class_weight else ""),
            "optimizer": "AdamW", "lr_schedule": "cosine annealing to 0",
            "epochs": args.epochs, "batch_size": args.batch_size,
            "feature_standardization": "z-score using train mean/std",
            "selection": "best dev macro-F1 over epochs and the (lr, wd) grid",
            "grid": {"lr": args.lrs, "wd": args.wds},
        },
        "chosen": {"lr": best["lr"], "wd": best["wd"], "best_epoch": best["epoch"]},
        "dev": {"acc": best["acc"], "macro_f1": best["macro_f1"], "weighted_f1": best["weighted_f1"],
                "loss": best["loss"]},
        "test": {"acc": m["acc"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                 "majority_baseline_acc": baseline_acc, "confusion_matrix": cm.tolist(),
                 "per_class": {c: per_class[c] for c in CLASSES}},
        "timing_seconds": {
            "feature_extraction": extraction["total_seconds"] if extraction else None,
            "head_training_chosen_config": best["train_seconds"],
            "head_training_full_sweep": sum(s["train_seconds"] for s in sweep),
            "train_head_script_total": time.time() - t_start,
        },
        "extraction": extraction,
        "history_chosen_config": best["history"],
        "sweep": sweep,
    }
    out = os.path.join(args.results_dir, "metrics", f"{args.prefix}.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved metrics ->", out)
    print(f"total time: {time.time() - t_start:.1f}s")
 
 
if __name__ == "__main__":
    main()
 