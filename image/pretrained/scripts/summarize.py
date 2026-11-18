
"""
STEP 4: Turn all metrics + prediction files into write-up material, including a
comparison ACROSS MODEL SIZES (ViT-B 86M / ViT-L 307M / ViT-H 632M).
 
Reads results/metrics/*.json and results/predictions/*_test.csv from train_head.py and writes:
 
  results/summary.csv / summary.md        one row per model: size, test acc / macro-F1 / weighted-F1,
                                          per-class F1 + recall, change vs the smallest model,
                                          chosen hyperparameters, extraction speed + timing
  results/plots/<prefix>_curves.png       training curves for each model
  results/plots/size_comparison.png       how scores, per-class recall and cost change with size
  results/plots/curves_all_sizes.png      all models' training curves overlaid
  results/size_agreement.csv              % of test images on which each pair of sizes agrees
  results/size_transitions.csv            per image: which sizes got it right, and whether the
                                          bigger model FIXED or BROKE it (for qualitative examples)
 
Usage (from the repo root):
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
 
# Fallback sizes if a metrics file has no extraction info
KNOWN_PARAMS_M = {"vit-base": 86, "vit-large": 307, "vit-huge": 632}
 
# Colors: model size is ORDERED, so sizes use one hue light -> dark (small -> big).
SIZE_RAMP = ["#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
# Different metrics are different things, so they use distinct categorical hues in fixed order.
METRIC_COLORS = {"test_acc": "#2a78d6", "test_macro_f1": "#eb6834", "test_weighted_f1": "#1baf7a"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e6"
 
plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
 
 
def short_name(prefix_or_model):
    for k in KNOWN_PARAMS_M:
        if k in prefix_or_model:
            return {"vit-base": "ViT-B", "vit-large": "ViT-L", "vit-huge": "ViT-H"}[k]
    return prefix_or_model.split("/")[-1]
 
 
def params_m(m):
    ex = m.get("extraction") or {}
    if ex.get("backbone_params"):
        return round(ex["backbone_params"] / 1e6)
    for k, v in KNOWN_PARAMS_M.items():
        if k in m["prefix"]:
            return v
    return None
 
 
def plot_curves_one(m, out):
    h = pd.DataFrame(m["history_chosen_config"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    a1.plot(h.epoch, h.train_loss, lw=2, color="#2a78d6", label="train loss")
    a1.plot(h.epoch, h.dev_loss, lw=2, color="#eb6834", label="dev loss")
    a1.axhline(0.693, color=MUTED, ls=":", lw=1, label="50/50 guessing (0.693)")
    a1.set_xlabel("epoch"); a1.set_ylabel("cross-entropy"); a1.legend(frameon=False)
    a2.plot(h.epoch, h.dev_acc, lw=2, color="#2a78d6", label="dev accuracy")
    a2.plot(h.epoch, h.dev_macro_f1, lw=2, color="#eb6834", label="dev macro-F1")
    a2.axvline(m["chosen"].get("best_epoch", 0), color=MUTED, ls="--", lw=1, label="chosen epoch")
    a2.set_xlabel("epoch"); a2.legend(frameon=False)
    fig.suptitle(f"{m['prefix']}  (lr={m['chosen']['lr']:g}, wd={m['chosen']['wd']:g})", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
 
 
def plot_size_comparison(df, out):
    """Three small multiples (no dual axes): scores, per-class recall, cost, each vs model size."""
    df = df.dropna(subset=["backbone_params_M"])
    x = df["backbone_params_M"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
 
    # (a) overall test scores vs size
    ax = axes[0]
    for col, label in [("test_acc", "accuracy"), ("test_macro_f1", "macro-F1"), ("test_weighted_f1", "weighted-F1")]:
        ax.plot(x, df[col], lw=2, marker="o", ms=8, color=METRIC_COLORS[col], label=label)
        ax.annotate(f"{df[col].iloc[-1]:.3f}", (x.iloc[-1], df[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.axhline(df["majority_baseline_acc"].iloc[0], color=MUTED, ls=":", lw=1, label="majority-class baseline")
    ax.set_title("Test scores vs model size", color=INK, loc="left")
    ax.set_ylabel("score")
 
    # (b) per-class recall vs size: which class does a bigger model help?
    ax = axes[1]
    for col, label, color in [("recall_informative", "informative", "#2a78d6"),
                              ("recall_not_informative", "not informative", "#eb6834")]:
        ax.plot(x, df[col], lw=2, marker="o", ms=8, color=color, label=label)
    ax.set_title("Test recall per class", color=INK, loc="left")
    ax.set_ylabel("recall")
 
    # (c) cost: feature-extraction throughput vs size
    ax = axes[2]
    if df["extract_images_per_s"].notna().any():
        ax.plot(x, df["extract_images_per_s"], lw=2, marker="o", ms=8, color="#2a78d6")
        for xi, yi in zip(x, df["extract_images_per_s"]):
            if pd.notna(yi):
                ax.annotate(f"{yi:.0f}/s", (xi, yi), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_title("Feature extraction speed", color=INK, loc="left")
    ax.set_ylabel("images per second")
 
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(x); ax.set_xticklabels([f"{n}\n{p}M" for n, p in zip(df["name"], x)])
        ax.minorticks_off()
        ax.set_xlabel("backbone (frozen params)")
    axes[0].legend(frameon=False, fontsize=8); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
 
 
def plot_curves_all(metrics, out):
    """Overlay every model's training curves; one ordered blue per size, small -> big."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    ends = []
    for i, m in enumerate(metrics):
        h = pd.DataFrame(m["history_chosen_config"])
        c = SIZE_RAMP[min(i, len(SIZE_RAMP) - 1)]
        name = short_name(m["prefix"])
        a1.plot(h.epoch, h.train_loss, lw=2, color=c, label=name)
        a2.plot(h.epoch, h.dev_macro_f1, lw=2, color=c, label=name)
        ends.append([h.dev_macro_f1.iloc[-1], h.epoch.iloc[-1], name])
 
    # direct labels at the line ends, nudged apart so close curves don't overlap
    lo, hi = a2.get_ylim()
    gap = 0.06 * (hi - lo)
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y, xe, name in ends:
        a2.annotate(name, (xe, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK, annotation_clip=False)
    a1.axhline(0.693, color=MUTED, ls=":", lw=1)
    a1.set_title("Train loss (chosen settings)", color=INK, loc="left"); a1.set_xlabel("epoch"); a1.set_ylabel("cross-entropy")
    a2.set_title("Dev macro-F1", color=INK, loc="left"); a2.set_xlabel("epoch")
    a1.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
 
 
def compare_predictions(results_dir, order):
    """Per-image comparison between sizes on the test set."""
    files = {short_name(os.path.basename(f)): f
             for f in glob.glob(os.path.join(results_dir, "predictions", "*_test.csv"))}
    names = [n for n in order if n in files]
    if len(names) < 2:
        return
    preds = {n: pd.read_csv(files[n], dtype={"image_id": str}).set_index("image_id") for n in names}
    common = sorted(set.intersection(*(set(p.index) for p in preds.values())))
 
    # pairwise agreement: how often two sizes give the same prediction
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            agree.loc[a, b] = (preds[a].loc[common, "pred_label"] == preds[b].loc[common, "pred_label"]).mean()
    agree.round(4).to_csv(os.path.join(results_dir, "size_agreement.csv"))
    print("\nprediction agreement between sizes (test):\n", agree.round(3).to_string())
 
    # per image: correct or not for each size, and what happened going up one size
    t = pd.DataFrame({"true_label": preds[names[0]].loc[common, "true_label"]})
    for n in names:
        t[f"{n}_correct"] = preds[n].loc[common, "correct"].astype(bool)
        t[f"{n}_prob_informative"] = preds[n].loc[common, "prob_informative"]
    print("\nwhat changes when the model gets bigger (test images):")
    for small, big in zip(names, names[1:]):
        s, b = t[f"{small}_correct"], t[f"{big}_correct"]
        col = f"{small}_to_{big}"
        t[col] = "same"
        t.loc[~s & b, col] = "fixed"     # small model wrong, bigger model right
        t.loc[s & ~b, col] = "broke"     # small model right, bigger model wrong
        counts = t[col].value_counts()
        print(f"  {small} -> {big}: fixed {counts.get('fixed', 0)}, broke {counts.get('broke', 0)}, "
              f"net {counts.get('fixed', 0) - counts.get('broke', 0):+d}")
    t.reset_index().to_csv(os.path.join(results_dir, "size_transitions.csv"), index=False)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="image/pretrained/results")
    args = ap.parse_args()
 
    files = sorted(glob.glob(os.path.join(args.results_dir, "metrics", "*.json")))
    if not files:
        raise SystemExit(f"no metrics found in {args.results_dir}/metrics; run train_head.py first")
    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)
 
    metrics = [json.load(open(f)) for f in files]
    metrics.sort(key=lambda m: params_m(m) or 1e9)   # small -> big
 
    rows = []
    for m in metrics:
        ex = m.get("extraction") or {}
        t = m.get("timing_seconds", {})
        pc = m["test"].get("per_class", {})
        n_img = sum(s["images"] for s in ex.get("splits", {}).values()) if ex.get("splits") else None
        rows.append({
            "name": short_name(m["prefix"]),
            "model": ex.get("model", m["prefix"]),
            "backbone_params_M": params_m(m),
            "feature_dim": m.get("data", {}).get("feature_dim"),
            "test_acc": m["test"]["acc"],
            "test_macro_f1": m["test"]["macro_f1"],
            "test_weighted_f1": m["test"]["weighted_f1"],
            "f1_informative": pc.get("informative", {}).get("f1-score"),
            "f1_not_informative": pc.get("not_informative", {}).get("f1-score"),
            "recall_informative": pc.get("informative", {}).get("recall"),
            "recall_not_informative": pc.get("not_informative", {}).get("recall"),
            "majority_baseline_acc": m["test"]["majority_baseline_acc"],
            "dev_macro_f1": m.get("dev", {}).get("macro_f1", m.get("dev_macro_f1")),
            "lr": m["chosen"]["lr"], "wd": m["chosen"]["wd"],
            "best_epoch": m["chosen"].get("best_epoch", m["chosen"].get("epoch")),
            "extract_min": t.get("feature_extraction") and t["feature_extraction"] / 60,
            "extract_images_per_s": (n_img / t["feature_extraction"]) if (n_img and t.get("feature_extraction")) else None,
            "head_train_s": t.get("head_training_chosen_config"),
            "device": m.get("device"),
        })
        if m.get("history_chosen_config"):
            out = os.path.join(args.results_dir, "plots", f"{m['prefix']}_curves.png")
            plot_curves_one(m, out); print("saved plot ->", out)
 
    df = pd.DataFrame(rows)
    # change vs the smallest model: "what does going bigger buy you?"
    for col in ["test_acc", "test_macro_f1", "test_weighted_f1"]:
        df[f"delta_{col}_vs_smallest"] = df[col] - df[col].iloc[0]
 
    df.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
    try:
        md = df.to_markdown(index=False, floatfmt=".4f")   # needs the `tabulate` package
    except ImportError:
        md = df.to_string(index=False)                     # fallback: plain-text table
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(md + "\n")
    pd.set_option("display.width", 250)
    show = ["name", "backbone_params_M", "test_acc", "test_macro_f1", "test_weighted_f1",
            "delta_test_macro_f1_vs_smallest", "recall_informative", "recall_not_informative",
            "best_epoch", "extract_min", "head_train_s"]
    print("\n", df[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsaved -> {args.results_dir}/summary.csv and summary.md")
 
    if len(df) >= 2:
        plot_size_comparison(df, os.path.join(args.results_dir, "plots", "size_comparison.png"))
        if all(m.get("history_chosen_config") for m in metrics):
            plot_curves_all(metrics, os.path.join(args.results_dir, "plots", "curves_all_sizes.png"))
        print("saved -> plots/size_comparison.png, plots/curves_all_sizes.png")
        compare_predictions(args.results_dir, list(df["name"]))
 
 
if __name__ == "__main__":
    main()
 