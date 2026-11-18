"""Compare test IDs and score missing wildfire examples with a saved SigLIP classifier."""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from train_multimodal import Encoder, LABELS, extract


def read_rows(path, delimiter=","):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        return reader.fieldnames, list(reader)


def index_rows(rows):
    indexed = {row["image_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate image IDs in CSV or TSV")
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    args = parser.parse_args()
    scoring = (args.annotations, args.classifier, args.output)
    if any(scoring) and not all(scoring):
        parser.error("Scoring requires --annotations, --classifier, and --output")

    columns, existing = read_rows(args.predictions)
    current = index_rows(existing)
    reference = index_rows(read_rows(args.reference)[1])
    if current.keys() - reference.keys():
        raise ValueError("Some existing image IDs are outside the reference set")
    if any(row["true_label"] != reference[key]["true_label"] for key, row in current.items()):
        raise ValueError("True labels differ on shared image IDs")
    missing = sorted(reference.keys() - current.keys())
    print(f"{len(current)} existing, {len(reference)} reference, {len(missing)} missing", flush=True)
    for image_id in missing:
        print(image_id, reference[image_id]["true_label"], flush=True)
    if not args.output:
        return
    if args.output.exists():
        raise FileExistsError(args.output)

    import joblib
    from sklearn.metrics import accuracy_score, f1_score

    models = {row["model"] for row in existing}
    if len(models) != 1 or any(not row["prob_informative"] for row in existing):
        raise ValueError("Expected one model name and probabilities in the existing CSV")
    model, = models
    annotations = index_rows(read_rows(args.annotations, "\t")[1])
    examples = [annotations[image_id] for image_id in missing]
    for row in examples:
        label = reference[row["image_id"]]["true_label"]
        if row["text_info"] != label or row["image_info"] != label or not row["tweet_text"].strip():
            raise ValueError(f"Invalid annotation for {row['image_id']}")

    if examples:
        classifier = joblib.load(args.classifier)
        encoder = Encoder("siglip", args.device)
        features = extract(encoder, examples, SimpleNamespace(batch_size=8, image_root=args.image_root))
        predictions = classifier.predict(features)
        informative = list(classifier.classes_).index(LABELS.index("informative"))
        probabilities = classifier.predict_proba(features)[:, informative]
        for row, prediction, probability in zip(examples, predictions, probabilities, strict=True):
            label = row["text_info"]
            predicted = LABELS[prediction]
            existing.append({
                "image_id": row["image_id"],
                "true_label": label,
                "pred_label": predicted,
                "prob_informative": f"{probability:.4f}",
                "correct": str(label == predicted),
                "model": model,
            })

    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(existing)
    true = [row["true_label"] for row in existing]
    predicted = [row["pred_label"] for row in existing]
    print(f"Wrote {len(existing)} rows to {args.output}")
    print(f"Accuracy {accuracy_score(true, predicted):.4f}")
    print(f"Macro-F1 {f1_score(true, predicted, labels=LABELS, average='macro', zero_division=0):.4f}")


if __name__ == "__main__":
    main()
