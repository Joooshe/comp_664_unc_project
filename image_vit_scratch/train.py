"""
ViT-B/16 trained FROM SCRATCH (no pre-trained weights) on CrisisMMD v2.0,
binary informative vs. not-informative, image modality only, agreed-label pairs only,
event split: train = hurricanes + Sri Lanka floods, val = earthquakes,
test = California wildfires (held-out event).

Recipe (DeiT-style, adapted for ~10k images): AdamW, linear warmup + cosine decay,
RandAugment, Mixup/CutMix, label smoothing, random erasing, stochastic depth,
grad clipping, bf16/fp16 autocast. Model selection on validation macro-F1.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

import timm
from timm.data import Mixup, create_transform
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2

import data as D


import timm.data.auto_augment as _AA
from PIL import ImageOps as _ImageOps

_orig_posterize = _AA.posterize


def _safe_posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return _ImageOps.posterize(img, max(1, int(bits_to_keep)))


_AA.posterize = _safe_posterize
for _k, _v in list(_AA.NAME_TO_OP.items()):
    if _v is _orig_posterize:
        _AA.NAME_TO_OP[_k] = _safe_posterize


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agreed_dir", required=True, help="unzipped crisismmd_datasplit_agreed_label folder")
    p.add_argument("--cache_dir", default="./cache_256", help="where resized images are cached")
    p.add_argument("--image_root", default=None,
                   help="optional folder that already contains CrisisMMD's data_image/ tree "
                        "(e.g. extracted CrisisMMD_v2.0.tar.gz); otherwise downloaded from Hugging Face")
    p.add_argument("--out_dir", default="./runs/vit_b16_scratch")
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--cache_short_side", type=int, default=256)
    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128, help="per-step batch that goes on the GPU")
    p.add_argument("--accum_steps", type=int, default=1,
                   help="gradient accumulation; effective batch = batch_size * accum_steps")
    p.add_argument("--lr", type=float, default=5e-4, help="peak LR at batch 512; scaled linearly to batch_size")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--drop_path", type=float, default=0.1)
    # augmentation / regularisation
    p.add_argument("--randaug", default="rand-m9-mstd0.5-inc1", help="timm auto_augment string; 'none' to disable")
    p.add_argument("--crop_scale_min", type=float, default=0.25)
    p.add_argument("--reprob", type=float, default=0.25, help="random erasing probability")
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--smoothing", type=float, default=0.1)
    # system
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="2 epochs on a 512-image subset to check the pipeline")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageDS(Dataset):
    def __init__(self, df, tf):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i], i


def metrics(y, pred):
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "f1_informative": float(f1_score(y, pred, pos_label=1, average="binary", zero_division=0)),
        "f1_not_informative": float(f1_score(y, pred, pos_label=0, average="binary", zero_division=0)),
        "frac_informative": float(np.mean(y)) if len(y) else float("nan"),
    }


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    probs = np.zeros((len(loader.dataset), 2), dtype=np.float32)
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y, idx in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        logits = logits.float()
        loss_sum += ce(logits, y).item()
        n += len(y)
        probs[idx.numpy()] = logits.softmax(-1).cpu().numpy()
    return probs, loss_sum / max(n, 1)


def main():
    args = get_args()
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("no CUDA device found")

    # ---------------- data ----------------
    df = D.load_agreed_pairs(args.agreed_dir)
    df = D.assign_event_split(df)
    df = D.cache_images(df, args.cache_dir, short_side=args.cache_short_side, image_root=args.image_root)
    print("[data] agreed pairs per split / event / class:\n" + D.describe(df))
    df.to_csv(os.path.join(args.out_dir, "data_manifest.csv"), index=False)

    tr_df = df[df.split == "train"].reset_index(drop=True)
    va_df = df[df.split == "val"].reset_index(drop=True)
    te_df = df[df.split == "test"].reset_index(drop=True)
    if args.smoke:
        tr_df = tr_df.sample(min(512, len(tr_df)), random_state=0).reset_index(drop=True)
        va_df = va_df.sample(min(256, len(va_df)), random_state=0).reset_index(drop=True)
        te_df = te_df.sample(min(256, len(te_df)), random_state=0).reset_index(drop=True)
        args.epochs, args.warmup_epochs = 2, 1

    train_tf = create_transform(
        input_size=args.img_size, is_training=True,
        scale=(args.crop_scale_min, 1.0), hflip=0.5,
        auto_augment=None if args.randaug == "none" else args.randaug,
        interpolation="bicubic", re_prob=args.reprob, re_mode="pixel", re_count=1,
    )
    eval_tf = create_transform(input_size=args.img_size, is_training=False,
                               interpolation="bicubic", crop_pct=0.9)

    use_mix = args.mixup > 0 or args.cutmix > 0
    loader_kw = dict(num_workers=args.workers, pin_memory=True)
    train_dl = DataLoader(ImageDS(tr_df, train_tf), batch_size=args.batch_size, shuffle=True,
                          drop_last=True, persistent_workers=args.workers > 0,
                          **loader_kw)  # drop_last: Mixup needs even batches
    val_dl = DataLoader(ImageDS(va_df, eval_tf), batch_size=args.batch_size * 2, **loader_kw)
    test_dl = DataLoader(ImageDS(te_df, eval_tf), batch_size=args.batch_size * 2, **loader_kw)

    # ---------------- model ----------------
    model = timm.create_model(args.model, pretrained=False, num_classes=2,
                              img_size=args.img_size, drop_path_rate=args.drop_path)
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = n_params - sum(p.numel() for p in model.get_classifier().parameters())
    print(f"[model] {args.model} from scratch: {n_params/1e6:.2f}M params "
          f"({n_backbone/1e6:.2f}M backbone + 2-class head; 86.57M with the 1000-class ImageNet head)")
    model.to(device)
    fwd_model = torch.compile(model) if args.compile else model

    eff_batch = args.batch_size * args.accum_steps
    lr = args.lr * eff_batch / 512
    print(f"[optim] effective batch {eff_batch} ({args.batch_size} x {args.accum_steps} accum), peak LR {lr:.2e}")
    opt = create_optimizer_v2(model, "adamw", lr=lr, weight_decay=args.weight_decay)  # no WD on norms/bias/pos/cls
    steps_per_epoch = len(train_dl) // args.accum_steps
    if steps_per_epoch == 0:
        raise SystemExit("Not enough training batches for one optimizer step; lower --accum_steps")
    total, warm = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return (args.min_lr + (lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * t))) / lr

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.smoothing,
                     num_classes=2) if use_mix else None
    criterion = SoftTargetCrossEntropy() if use_mix else (
        LabelSmoothingCrossEntropy(args.smoothing) if args.smoothing > 0 else nn.CrossEntropyLoss())

    amp_dtype = None
    if args.amp == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.amp in ("fp16", "bf16") and device.type == "cuda":
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "scaled_lr": lr, "n_params": n_params,
                   "n_train": len(tr_df), "n_val": len(va_df), "n_test": len(te_df)}, f, indent=2)

    # ---------------- train ----------------
    best_f1, best_epoch, history = -1.0, -1, []
    ckpt_path = os.path.join(args.out_dir, "best.pt")
    for epoch in range(args.epochs):
        fwd_model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        n_used = steps_per_epoch * args.accum_steps  # drop trailing micro-batches that can't fill a step
        for it, (x, y, _) in enumerate(train_dl):
            if it >= n_used:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = criterion(fwd_model(x), y)
            scaler.scale(loss / args.accum_steps).backward()
            run_loss += loss.item() * len(x)
            seen += len(x)
            if (it + 1) % args.accum_steps == 0:
                if args.clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        probs, val_loss = predict(fwd_model, val_dl, device, amp_dtype)
        m = metrics(va_df["label"].values, probs.argmax(1))
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "train_loss": run_loss / max(seen, 1),
               "val_loss": val_loss, "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
               "sec": time.time() - t0}
        history.append(row)
        improved = m["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = m["macro_f1"], epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "args": vars(args)}, ckpt_path)
        print(f"ep {epoch+1:3d}/{args.epochs} | lr {row['lr']:.2e} | train {row['train_loss']:.4f} | "
              f"val loss {val_loss:.4f} acc {m['accuracy']:.4f} mF1 {m['macro_f1']:.4f}"
              f"{'  *best*' if improved else ''} | {row['sec']:.0f}s", flush=True)
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

    # ---------------- final evaluation with best-val checkpoint ----------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    results, per_event = {"best_epoch": best_epoch}, []
    for name, sdf, dl in (("val", va_df, val_dl), ("test", te_df, test_dl)):
        probs, _ = predict(model, dl, device, amp_dtype)
        pred = probs.argmax(1)
        y = sdf["label"].values
        results[name] = metrics(y, pred)
        results[name]["confusion_matrix[true][pred]"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        for ev, g in sdf.assign(pred=pred).groupby("event_name"):
            per_event.append({"split": name, "event_name": ev, **metrics(g["label"].values, g["pred"].values)})
    per_event = pd.DataFrame(per_event)
    per_event.to_csv(os.path.join(args.out_dir, "per_event_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== best epoch {best_epoch} (val macro-F1 {best_f1:.4f}) ===")
    for name in ("val", "test"):
        r = results[name]
        print(f"{name:5s} n={r['n']:5d}  acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}")
    cols = ["split", "event_name", "n", "accuracy", "macro_f1", "f1_informative", "f1_not_informative",
            "frac_informative"]
    print("\nper-event:\n" + per_event[cols].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
