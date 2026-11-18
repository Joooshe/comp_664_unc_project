"""
STEP 4: Turn all metrics jsons into write-up material.

Reads every results/metrics/*.json made by train_head.py and produces:
  results/summary.csv               one row per model: params, test acc / macro-F1 / weighted-F1,
                                    dev macro-F1, chosen lr/wd/epoch, extraction + training time
  results/summary.md                the same table in Markdown (paste into your report)
  results/plots/<prefix>_curves.png training curves for the chosen config:
                                    train vs dev loss, and dev acc / macro-F1 per epoch

Usage (from the repo root, after at least one train_head.py run):
    uv run python image/pretrained/scripts/summarize.py
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display needed (works on Longleaf too)
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="image/pretrained/results")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "metrics", "*.json")))
    if not files:
        raise SystemExit(f"no metrics found in {args.results_dir}/metrics; run train_head.py first")

    rows = []
    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)
    for path in files:
        m = json.load(open(path))
        ex = m.get("extraction") or {}
        t = m.get("timing_seconds", {})
        rows.append({
            "model": ex.get("model", m["prefix"]),
            "backbone_params_M": round(ex["backbone_params"] / 1e6) if ex.get("backbone_params") else None,
            "feature_dim": m.get("data", {}).get("feature_dim"),
            "test_acc": m["test"]["acc"],
            "test_macro_f1": m["test"]["macro_f1"],
            "test_weighted_f1": m["test"]["weighted_f1"],
            "majority_baseline_acc": m["test"]["majority_baseline_acc"],
            "dev_macro_f1": m.get("dev", {}).get("macro_f1", m.get("dev_macro_f1")),
            "lr": m["chosen"]["lr"], "wd": m["chosen"]["wd"],
            "best_epoch": m["chosen"].get("best_epoch", m["chosen"].get("epoch")),
            "extract_min": t.get("feature_extraction") and t["feature_extraction"] / 60,
            "head_train_s": t.get("head_training_chosen_config"),
            "sweep_s": t.get("head_training_full_sweep"),
            "device": m.get("device"),
        })

        # ---- training curves for the chosen config ----
        h = m.get("history_chosen_config")
        if h:
            hdf = pd.DataFrame(h)
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
            a1.plot(hdf.epoch, hdf.train_loss, label="train loss")
            a1.plot(hdf.epoch, hdf.dev_loss, label="dev loss")
            a1.axhline(0.693, color="gray", ls=":", lw=1, label="50/50 guessing (0.693)")
            a1.set_xlabel("epoch"); a1.set_ylabel("cross-entropy"); a1.legend()
            a2.plot(hdf.epoch, hdf.dev_acc, label="dev accuracy")
            a2.plot(hdf.epoch, hdf.dev_macro_f1, label="dev macro-F1")
            a2.axvline(m["chosen"].get("best_epoch", 0), color="gray", ls="--", lw=1, label="chosen epoch")
            a2.set_xlabel("epoch"); a2.legend()
            fig.suptitle(f"{m['prefix']}  (lr={m['chosen']['lr']:g}, wd={m['chosen']['wd']:g})", fontsize=10)
            fig.tight_layout()
            out = os.path.join(args.results_dir, "plots", f"{m['prefix']}_curves.png")
            fig.savefig(out, dpi=150); plt.close(fig)
            print("saved plot ->", out)

    df = pd.DataFrame(rows).sort_values("backbone_params_M", na_position="last")
    df.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
    try:
        md = df.to_markdown(index=False, floatfmt=".4f")   # needs the `tabulate` package
    except ImportError:
        md = df.to_string(index=False)                     # fallback: plain-text table
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(md + "\n")
    pd.set_option("display.width", 200)
    print("\n", df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsaved -> {args.results_dir}/summary.csv and summary.md")


if __name__ == "__main__":
    main()
