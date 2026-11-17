"""
Data preparation for the CrisisMMD informative / not-informative image task.

* Agreed pairs (text label == image label) come from the official CrisisMMD v2.0
  "agreed label" data split (crisismmd_datasplit_agreed_label.zip). The Hugging Face
  copy (QCRI/CrisisMMD) carries a single `label` column, so it cannot tell us which
  pairs agree; we use it only as the image source.
* All agreed pairs (train+dev+test of the released split) are pooled and re-split
  by event, following the team's project update:
      test  = california_wildfires
      val   = mexico_earthquake, iraq_iran_earthquake
      train = hurricane_harvey, hurricane_irma, hurricane_maria, srilanka_floods
* Images are resized once (shorter side = --cache_short_side) and cached as JPEGs so
  training does not decode full-size tweet images every epoch.
"""

from __future__ import annotations

import glob
import os
import sys

import pandas as pd

CLASSES = ["not_informative", "informative"]  # index 0, 1
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}

TEST_EVENTS = ["california_wildfires"]
VAL_EVENTS = ["mexico_earthquake", "iraq_iran_earthquake"]
TRAIN_EVENTS = ["hurricane_harvey", "hurricane_irma", "hurricane_maria", "srilanka_floods"]

_DROP_LABELS = {"dont_know_or_cant_judge", "dont_know", "cant_judge", ""}


def _norm_label(x) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def load_agreed_pairs(agreed_dir: str) -> pd.DataFrame:
    """Read every informative-task TSV in `agreed_dir` (searched recursively) and
    return one row per agreed (tweet, image) pair:
        event_name, tweet_id, image_id, label (0/1), label_name
    """
    files = sorted(
        f for f in glob.glob(os.path.join(agreed_dir, "**", "*.tsv"), recursive=True)
        if "informative" in os.path.basename(f).lower()
    )
    if not files:
        sys.exit(
            f"No informative-task .tsv files found under {agreed_dir}. Unzip "
            "crisismmd_datasplit_agreed_label.zip there; you should see files like "
            "task_informative_text_img_agreed_lab_train.tsv"
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        df.columns = [c.strip().lower() for c in df.columns]
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    for col in ("event_name", "image_id"):
        if col not in df.columns:
            sys.exit(f"Expected column '{col}' in agreed-label TSVs; got {list(df.columns)}")

    # if separate text/image labels are present, enforce equality 
    if {"label_text", "label_image"} <= set(df.columns):
        lt = df["label_text"].map(_norm_label)
        li = df["label_image"].map(_norm_label)
        n_before = len(df)
        df = df[lt == li].copy()
        df["label_name"] = li[lt == li]
        n_dis = n_before - len(df)
        if n_dis:
            print(f"[data] dropped {n_dis} rows whose text and image labels disagree")
    elif "label" in df.columns:
        df["label_name"] = df["label"].map(_norm_label)
    else:
        sys.exit(f"No label column found in agreed-label TSVs; got {list(df.columns)}")

    df = df[~df["label_name"].isin(_DROP_LABELS)]
    unknown = set(df["label_name"]) - set(LABEL_TO_ID)
    if unknown:
        sys.exit(f"Unexpected label values {unknown}; expected {CLASSES}")

    df["event_name"] = df["event_name"].map(lambda s: str(s).strip().lower())
    df["image_id"] = df["image_id"].map(lambda s: str(s).strip())
    df["label"] = df["label_name"].map(LABEL_TO_ID).astype(int)

    # released split should be duplicate-free, guard anyway
    dup = df.duplicated("image_id", keep=False)
    if dup.any():
        conflicting = df[dup].groupby("image_id")["label"].nunique()
        n_conf = int((conflicting > 1).sum())
        print(f"[data] {int(dup.sum())} duplicate image_id rows "
              f"({n_conf} with conflicting labels, dropped)")
        df = df[~df["image_id"].isin(conflicting[conflicting > 1].index)]
        df = df.drop_duplicates("image_id")

    keep = ["event_name", "tweet_id", "image_id", "label", "label_name"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def assign_event_split(df: pd.DataFrame) -> pd.DataFrame:
    known = set(TRAIN_EVENTS + VAL_EVENTS + TEST_EVENTS)
    unknown = set(df["event_name"]) - known
    if unknown:
        sys.exit(f"Unrecognised event names {unknown}")

    def which(e):
        if e in TEST_EVENTS:
            return "test"
        if e in VAL_EVENTS:
            return "val"
        return "train"

    df = df.copy()
    df["split"] = df["event_name"].map(which)
    return df


IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def _index_images(root: str) -> dict:
    """Map image_id (file stem) -> full path for every image file under root."""
    idx = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS:
                idx[stem] = os.path.join(dirpath, f)
    return idx


def cache_images(df: pd.DataFrame, cache_dir: str, short_side: int = 256,
                 image_root: str | None = None, hf_name: str = "QCRI/CrisisMMD") -> pd.DataFrame:
    """Resize the images we need (shorter side = short_side) into cache_dir.

    Source images come from `image_root` if given (any folder containing CrisisMMD's
    data_image/ tree, e.g. the extracted CrisisMMD_v2.0.tar.gz), otherwise the
    data_image/ folder of the Hugging Face repo is downloaded into cache_dir/_raw.
    The HF json files only hold image paths, so load_dataset() alone yields no pixels.
    Returns df with a `path` column; pairs whose image can't be found are dropped."""
    from PIL import Image

    os.makedirs(cache_dir, exist_ok=True)
    df = df.copy()
    df["path"] = df["image_id"].map(lambda i: os.path.join(cache_dir, f"{i}.jpg"))
    need = set(df.loc[~df["path"].map(os.path.exists), "image_id"])

    if need:
        if image_root is None:
            from huggingface_hub import snapshot_download

            image_root = os.path.join(cache_dir, "_raw")
            print(f"[data] downloading {hf_name} data_image/ (~18k files, a few GB) into {image_root}\n"
                  "       this is resumable: if it stops, just rerun the same command")
            snapshot_download(hf_name, repo_type="dataset", allow_patterns=["data_image/*"],
                              local_dir=image_root, max_workers=8)

        src_index = _index_images(image_root)
        print(f"[data] found {len(src_index)} source images under {image_root}; "
              f"resizing {len(need)} into {cache_dir} ...")
        n_bad, not_found = 0, []
        for k, iid in enumerate(sorted(need)):
            src = src_index.get(iid)
            if src is None:
                not_found.append(iid)
                continue
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    s = short_side / min(w, h)
                    if s < 1:
                        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
                    im.save(os.path.join(cache_dir, f"{iid}.jpg"), quality=95)
            except Exception as e:  # corrupt / unreadable image
                n_bad += 1
                print(f"[data] could not read {src}: {e}")
            if (k + 1) % 2000 == 0:
                print(f"[data]   {k + 1}/{len(need)}", flush=True)
        if not_found:
            print(f"[data] {len(not_found)} image_ids not found under {image_root}, "
                  f"e.g. {not_found[:3]}")
        if n_bad:
            print(f"[data] {n_bad} images unreadable")

    missing = ~df["path"].map(os.path.exists)
    if missing.any():
        print(f"[data] WARNING: {int(missing.sum())} of {len(df)} agreed pairs have no image; dropped")
        df = df[~missing]
    if df.empty:
        sys.exit("[data] no images available; check the download / --image_root path")
    return df.reset_index(drop=True)


def describe(df: pd.DataFrame) -> str:
    t = (df.groupby(["split", "event_name", "label_name"]).size()
           .unstack("label_name", fill_value=0))
    t["total"] = t.sum(axis=1)
    return t.to_string()
