"""Frozen image+text baselines for CrisisMMD Task 1."""

import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATASET = "QCRI/CrisisMMD"
AGREEMENT_URL = "https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip"
LABELS = ["informative", "not_informative"]
CHECKPOINTS = {
    "visualbert": "uclanlp/visualbert-vqa-coco-pre",
    "bridgetower": "BridgeTower/bridgetower-base",
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-large-patch16-384",
    "flava": "facebook/flava-full",
    "blip": "Salesforce/blip-itm-base-coco",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, default="clip")
    parser.add_argument("--split", choices=("event", "released"), default="event")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Seeded sample size per split for smoke runs")
    parser.add_argument("--skip-test", action="store_true", help="Run training and validation without scoring the test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-root", type=Path, help="Local directory containing data_image/")
    parser.add_argument("--agreement-zip", type=Path, help="Local copy of QCRI's annotation ZIP")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--data-only", action="store_true", help="Print split counts and exit")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 2):
        parser.error("--batch-size must be positive and --limit must be at least 2")
    return args


def released_annotations(local_zip):
    if local_zip:
        archive = local_zip.read_bytes()
    else:
        with urlopen(AGREEMENT_URL, timeout=60) as response:
            archive = response.read()
    annotations = {}
    with ZipFile(io.BytesIO(archive)) as files:
        for split in ("train", "dev", "test"):
            name, = [
                name for name in files.namelist()
                if Path(name).name == f"task_informative_text_img_agreed_lab_{split}.tsv"
                and not name.startswith("__MACOSX/")
            ]
            with files.open(name) as source:
                rows = csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig"), delimiter="\t")
                for row in rows:
                    image_path = Path(row["image"].strip()).as_posix()
                    annotations[image_path] = split
    return annotations


def load_splits(args):
    from datasets import load_dataset

    annotations = released_annotations(args.agreement_zip)
    hub = load_dataset(DATASET, "informative")
    splits = {"train": [], "validation": [], "test": []}
    matched = set()
    for split in hub.values():
        columns = ["tweet_id", "image_id", "image_path", "event_name", "tweet_text", "label"]
        for row in split.select_columns(columns):
            path = Path(row["image_path"]).as_posix()
            if path not in annotations or path in matched or not (row["tweet_text"] or "").strip():
                continue
            matched.add(path)
            released = annotations[path]
            event = row["event_name"]
            if args.split == "released":
                target = "validation" if released == "dev" else released
            elif event == "california_wildfires":
                target = "test"
            elif event.endswith("_earthquake"):
                target = "validation"
            else:
                target = "train"
            if row["label"] not in (0, 1):
                raise ValueError(f"Unexpected informativeness label for {path}: {row['label']}")
            splits[target].append(row)
    print(f"Matched {len(matched):,} of {len(annotations):,} agreed-label pairs.", flush=True)
    if len(matched) != len(annotations):
        raise ValueError("Some agreed-label pairs could not be joined to the Hub dataset by image path.")
    return splits


def load_image(row, root):
    from PIL import Image
    from huggingface_hub import hf_hub_download

    relative_path = row["image_path"]
    path = root / relative_path if root else hf_hub_download(DATASET, relative_path, repo_type="dataset")
    with Image.open(path) as image:
        return image.convert("RGB")


class Encoder:
    def __init__(self, name, device):
        import torch
        import transformers as hf

        self.name = name
        self.device = device
        self.torch = torch
        checkpoint = CHECKPOINTS[name]
        classes = {
            "visualbert": hf.VisualBertModel,
            "bridgetower": hf.BridgeTowerModel,
            "clip": hf.CLIPModel,
            "siglip": hf.SiglipModel,
            "flava": hf.FlavaModel,
            "blip": hf.BlipForImageTextRetrieval,
        }
        self.model = classes[name].from_pretrained(checkpoint).to(device).eval()
        if name == "visualbert":
            from torchvision.models import ResNet50_Weights, resnet50
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

            self.processor = hf.AutoTokenizer.from_pretrained("bert-base-uncased")
            detector_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.detector = fasterrcnn_resnet50_fpn(weights=detector_weights).to(device).eval()
            self.detector_transform = detector_weights.transforms()
            region_weights = ResNet50_Weights.DEFAULT
            self.region_transform = region_weights.transforms()
            self.region_encoder = resnet50(weights=region_weights)
            self.region_encoder.fc = torch.nn.Identity()
            self.region_encoder.to(device).eval()
        else:
            self.processor = hf.AutoProcessor.from_pretrained(checkpoint)

    def regions(self, images):
        torch = self.torch
        regions = []
        for image in images:
            detection = self.detector([self.detector_transform(image).to(self.device)])[0]
            crops = [image]
            for box in detection["boxes"][:12].cpu().tolist():
                if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
                    crops.append(image.crop(box))
            pixels = torch.stack([self.region_transform(crop) for crop in crops]).to(self.device)
            regions.append(self.region_encoder(pixels))
        features = torch.nn.utils.rnn.pad_sequence(regions, batch_first=True)
        lengths = torch.tensor([len(region) for region in regions], device=self.device)
        mask = torch.arange(features.shape[1], device=self.device)[None, :] < lengths[:, None]
        return features, mask.long()

    def encode(self, texts, images):
        torch = self.torch
        with torch.inference_mode():
            token_options = dict(padding=True, truncation=True, max_length=77, return_tensors="pt")
            if self.name == "visualbert":
                tokens = self.processor(texts, **token_options).to(self.device)
                embeddings, mask = self.regions(images)
                output = self.model(**tokens, visual_embeds=embeddings, visual_attention_mask=mask)
                return output.pooler_output.float().cpu().numpy()

            if self.name == "siglip":
                token_options.update(padding="max_length", max_length=64)
            inputs = self.processor(images=images, text=texts, **token_options).to(self.device)
            if self.name == "blip":
                vision = self.model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state
                output = self.model.text_encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    encoder_hidden_states=vision,
                    encoder_attention_mask=torch.ones(
                        vision.shape[:2], dtype=torch.long, device=vision.device
                    ),
                )
                features = output.last_hidden_state[:, 0]
            elif self.name == "flava":
                output = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                )
                features = output.multimodal_embeddings[:, 0]
            else:
                output = self.model(**inputs)
                features = (
                    torch.cat([output.image_embeds, output.text_embeds], dim=-1)
                    if self.name in ("clip", "siglip") else output.pooler_output
                )
            return features.float().cpu().numpy()


def extract(encoder, rows, args):
    import numpy as np
    from tqdm.auto import tqdm

    features = []
    for start in tqdm(range(0, len(rows), args.batch_size), unit="batch"):
        batch = rows[start:start + args.batch_size]
        images = [load_image(row, args.image_root) for row in batch]
        features.append(encoder.encode([row["tweet_text"] for row in batch], images))
    return np.concatenate(features)


def write_test_comparison(path, rows, predictions, probabilities):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_id", "Model", "True_label", "pred_label", "prob_informative", "correct"])
        for row, prediction, probability in zip(rows, predictions, probabilities, strict=True):
            true_label = row["label"]
            writer.writerow([
                row["image_id"], "text+image", LABELS[true_label], LABELS[prediction],
                float(probability), int(true_label == prediction),
            ])


def write_team_results(output, rows, predictions, probabilities, metrics):
    model = f"multimodal_joint_{metrics['backbone_parameters']}"
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "model", "true_label", "pred_label", "prob_informative", "correct"])
        for row, prediction, probability in zip(rows, predictions, probabilities, strict=True):
            label = row["label"]
            writer.writerow([
                row["image_id"], model, LABELS[label], LABELS[prediction], float(probability),
                str(label == prediction).lower(),
            ])
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["macro_f1_score", "accuracy", "model"])
        writer.writerow([metrics["test"]["macro_f1"], metrics["test"]["accuracy"], model])


def main():
    from collections import Counter
    from random import Random

    args = arguments()
    splits = load_splits(args)
    if args.skip_test:
        del splits["test"]
    for split, rows in splits.items():
        if args.limit:
            # Include both classes when present so a tiny sample can still fit a classifier.
            rng = Random(args.seed)
            rng.shuffle(rows)
            representatives = {}
            for index, row in enumerate(rows):
                representatives.setdefault(row["label"], index)
            chosen = set(representatives.values())
            remaining = [index for index in range(len(rows)) if index not in chosen]
            chosen.update(remaining[:args.limit - len(chosen)])
            rows = splits[split] = [rows[index] for index in sorted(chosen)]
        if not rows:
            raise ValueError(f"{split} has no examples")
        print(f"{split}: {len(rows):,} pairs, {dict(Counter(row['event_name'] for row in rows))}", flush=True)
    if args.data_only:
        return
    if {row["label"] for row in splits["train"]} != {0, 1}:
        raise ValueError("Training requires examples from both informativeness classes.")

    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder(args.model, device)
    features = {}
    targets = {}
    for split, rows in splits.items():
        print(f"Encoding {split} with {args.model} on {device}", flush=True)
        features[split] = extract(encoder, rows, args)
        targets[split] = [row["label"] for row in rows]

    train_x, val_x = features["train"], features["validation"]
    train_y, val_y = targets["train"], targets["validation"]
    # Search fits each head on training rows only. Keep the winning training-only fit.
    predefined = PredefinedSplit([-1] * len(train_y) + [0] * len(val_y))
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1_macro", cv=predefined, refit=False, error_score="raise",
    )
    search.fit(np.concatenate([train_x, val_x]), train_y + val_y)
    classifier = search.estimator.set_params(**search.best_params_).fit(train_x, train_y)
    run_name = "full" if args.limit is None else f"smoke-{args.limit}"
    output = args.output or Path("results") / args.split / args.model / run_name
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output / "classifier.joblib")

    metrics = {
        "checkpoint": CHECKPOINTS[args.model],
        "split_protocol": args.split,
        "seed": args.seed,
        "limit": args.limit,
        "classifier_C": search.best_params_["logisticregression__C"],
        "backbone_parameters": sum(parameter.numel() for parameter in encoder.model.parameters()),
        "train_size": len(train_y),
    }
    for split in ("validation",) if args.skip_test else ("validation", "test"):
        predicted = classifier.predict(features[split])
        metrics[split] = {
            "size": len(predicted),
            "accuracy": accuracy_score(targets[split], predicted),
            "macro_f1": f1_score(targets[split], predicted, labels=[0, 1], average="macro", zero_division=0),
        }
        with (output / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["tweet_id", "image_id", "event_name", "label", "prediction"])
            for row, guess in zip(splits[split], predicted):
                writer.writerow([row["tweet_id"], row["image_id"], row["event_name"],
                                 LABELS[row["label"]], LABELS[guess]])
        if split == "test":
            informative_column = list(classifier.classes_).index(LABELS.index("informative"))
            probabilities = classifier.predict_proba(features[split])[:, informative_column]
            write_test_comparison(output / "test_comparison.csv", splits[split], predicted, probabilities)
            write_team_results(output, splits[split], predicted, probabilities, metrics)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved results in {output}")


if __name__ == "__main__":
    main()
