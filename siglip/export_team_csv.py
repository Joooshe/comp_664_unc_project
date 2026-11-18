"""Export team comparison CSVs from completed multimodal test runs."""

import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Directories containing full test results")
    parser.add_argument("--output", type=Path, default=Path("team_export"))
    args = parser.parse_args()

    predictions = []
    metrics = []
    for run in args.runs:
        try:
            scores = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            pairs = read_csv(run / "test_predictions.csv")
            comparison = read_csv(run / "test_comparison.csv")
        except FileNotFoundError as error:
            parser.error(f"{run} is missing {Path(error.filename).name}. Wait for the run to finish.")

        model = f"multimodal_joint_{scores['backbone_parameters']}"
        test = scores["test"]
        if not (len(pairs) == len(comparison) == test["size"]):
            parser.error(f"{run} has inconsistent test row counts")

        for pair, row in zip(pairs, comparison, strict=True):
            if (pair["image_id"], pair["label"], pair["prediction"]) != (
                row["Image_id"], row["True_label"], row["pred_label"]
            ):
                parser.error(f"{run} has mismatched prediction rows")
            predictions.append([
                row["Image_id"], model, row["True_label"], row["pred_label"],
                row["prob_informative"],
                str(row["True_label"] == row["pred_label"]).lower(),
            ])
        metrics.append([test["macro_f1"], test["accuracy"], model])

    args.output.mkdir(parents=True, exist_ok=True)
    exports = {
        "predictions.csv": (
            ["image_id", "model", "true_label", "pred_label", "prob_informative", "correct"],
            predictions,
        ),
        "metrics.csv": (["macro_f1_score", "accuracy", "model"], metrics),
    }
    for filename, (header, rows) in exports.items():
        path = args.output / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
        print(path)


if __name__ == "__main__":
    main()
