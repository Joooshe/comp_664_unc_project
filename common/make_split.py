"""
STEP 1: Build the shared dataset every model in the group uses.

What this does
--------------
1. Loads the HF CrisisMMD "informative" config and combines train/dev/test into one pool.
2. Loads the ORIGINAL per-event annotation files (files_individual_events/*_final_data.tsv),
   which have separate text and image labels (text_info / image_info).
3. Keeps only posts where text_info == image_info ("agreed" labels), following
   Ofli et al. 2020. On those rows the label is unambiguous, so we take it from the
   annotations rather than the HF `label` column.
4. Re-splits by EVENT:
       test  = california_wildfires
       dev   = all earthquake events (iraq_iran_earthquake, mexico_earthquake)
       train = everything else (hurricanes harvey/irma/maria, srilanka_floods)
5. Saves:
       data/informative_agreed_eventsplit/                  HF DatasetDict WITH images (for the image model)
       data/informative_agreed_eventsplit_manifest.csv      one row per post, no images (for everyone, commit this)

Run once on a LOGIN node (needs internet to download):
    uv run python common/make_split.py
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset
from datasets import Image as HFImage
from huggingface_hub import hf_hub_download

REPO = "QCRI/CrisisMMD"
EVENTS = [
    "california_wildfires", "hurricane_harvey", "hurricane_irma", "hurricane_maria",
    "iraq_iran_earthquake", "mexico_earthquake", "srilanka_floods",
]
CLASSES = {"informative", "not_informative"}


def split_of(event):
    """Event-based split rule. Change this function to try other splits."""
    if event == "california_wildfires":
        return "test"
    if "earthquake" in event:
        return "dev"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. Combine all HF splits into one pool ------------------------------
    hf = load_dataset(REPO, "informative")
    print("HF splits:", hf.num_rows)
    full = concatenate_datasets([hf[s] for s in hf])

    # Guard against the same image appearing in more than one original split.
    seen, keep = set(), []
    for i, iid in enumerate(full["image_id"]):
        if iid not in seen:
            seen.add(iid)
            keep.append(i)
    if len(keep) < len(full):
        print(f"[note] dropped {len(full) - len(keep)} duplicate image_ids")
    full = full.select(keep)
    print("combined rows:", len(full))

    # ---- 2. Load the original text + image annotations ------------------------
    # QUOTE_NONE: tweets contain stray quote characters that otherwise break parsing.
    ann = pd.concat([
        pd.read_csv(
            hf_hub_download(REPO, f"files_individual_events/{e}_final_data.tsv", repo_type="dataset"),
            sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
        )
        for e in EVENTS
    ], ignore_index=True)
    ann = ann.drop_duplicates("image_id")
    print("annotation rows:", len(ann))
    print(ann[["text_info", "image_info"]].value_counts().to_string(), "\n")

    # ---- 3. Keep only agreed rows ---------------------------------------------
    agreed = ann[ann["text_info"] == ann["image_info"]]
    agreed_label = dict(zip(agreed["image_id"], agreed["image_info"]))

    # input_columns="image_id" -> only that column is read, so no images get decoded here
    full = full.filter(lambda iid: iid in agreed_label, input_columns="image_id")
    print(f"agreed rows: {len(full)}")

    # Replace HF's single `label` with the agreed annotation label (a plain string)
    labels = [agreed_label[i] for i in full["image_id"]]
    assert set(labels) <= CLASSES, f"unexpected labels: {set(labels) - CLASSES}"
    full = full.remove_columns("label").add_column("label", labels)

    # ---- 3b. Attach the actual image files ---------------------------------------
    # The HF `image` column can come back empty (None), so we don't rely on it.
    # Instead we download ONLY the agreed rows' image files (into $HF_HOME, cached,
    # so reruns are instant) and rebuild the image column from each row's `image_path`.
    def fetch(rel_path):
        try:
            return hf_hub_download(REPO, rel_path, repo_type="dataset")
        except Exception as e:  # missing file on the hub, network hiccup, etc.
            print(f"[warn] could not download {rel_path}: {type(e).__name__}")
            return None

    rel_paths = full["image_path"]
    print(f"downloading {len(rel_paths)} images (cached after the first run)...")
    with ThreadPoolExecutor(max_workers=16) as pool:   # parallel downloads
        local = []
        for n, p in enumerate(pool.map(fetch, rel_paths), 1):
            local.append(p)
            if n % 1000 == 0:
                print(f"  {n}/{len(rel_paths)}", flush=True)

    keep = [i for i, p in enumerate(local) if p is not None]
    if len(keep) < len(full):
        print(f"[note] dropping {len(full) - len(keep)} rows whose image could not be downloaded")
    full = full.select(keep)
    paths = [local[i] for i in keep]
    full = full.remove_columns("image").add_column("image", paths).cast_column("image", HFImage())
    print(f"rows with images: {len(full)}")

    # ---- 4. Event-based split ---------------------------------------------------
    full = full.add_column("split", [split_of(e) for e in full["event_name"]])
    dd = DatasetDict({
        s: full.filter(lambda x, s=s: x == s, input_columns="split")
        for s in ["train", "dev", "test"]
    })

    # ---- 5. Sanity checks -------------------------------------------------------
    ids = {s: set(dd[s]["image_id"]) for s in dd}
    assert not (ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"])
    assert set(dd["test"]["event_name"]) == {"california_wildfires"}
    assert all("earthquake" in e for e in dd["dev"]["event_name"])

    manifest = full.remove_columns("image").to_pandas()
    # the same tweet (with several images) must never be in two splits
    assert manifest.groupby("tweet_id")["split"].nunique().max() == 1

    print("\nrows per split:", dd.num_rows)
    print(pd.crosstab(manifest["split"], manifest["label"], margins=True).to_string())

    # ---- 6. Save ------------------------------------------------------------------
    ds_path = os.path.join(args.out_dir, "informative_agreed_eventsplit")
    csv_path = os.path.join(args.out_dir, "informative_agreed_eventsplit_manifest.csv")
    dd.save_to_disk(ds_path)
    manifest.to_csv(csv_path, index=False)
    print(f"\nsaved dataset  -> {ds_path}")
    print(f"saved manifest -> {csv_path}  (commit this so the whole group uses the same split)")


if __name__ == "__main__":
    main()