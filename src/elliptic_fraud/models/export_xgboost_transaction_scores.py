"""Export transaction-level XGBoost prediction scores for diagnostics."""

import csv
from pathlib import Path

import numpy as np

from analyze_xgboost_temporal import (
    load_data,
    predict_in_batches,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = ROOT / "artifacts" / "xgboost_transaction_scores.csv"


def main():
    model, features, labels, time_steps, splits, rounds = load_data()

    print("\nGenerating first-tree predictions...")
    first_scores = predict_in_batches(model, features, rounds=1)

    print("\nGenerating all-tree predictions...")
    all_scores = predict_in_batches(model, features, rounds=rounds)

    if not (
        len(labels) == len(time_steps) == len(splits)
        == len(first_scores) == len(all_scores)
    ):
        raise ValueError("Prediction and graph dimensions do not match.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting transaction-level scores to: {OUTPUT_PATH}")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "row_index",
            "time_step",
            "split",
            "label",
            "first_tree_score",
            "all_51_trees_score",
        ])

        for index in range(len(labels)):
            writer.writerow([
                index,
                int(time_steps[index]),
                int(splits[index]),
                int(labels[index]),
                f"{first_scores[index]:.10f}",
                f"{all_scores[index]:.10f}",
            ])

    print("\nEXPORT COMPLETED")
    print("Transactions exported:", len(labels))
    print("Boosting rounds:", rounds)
    print("Output:", OUTPUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
