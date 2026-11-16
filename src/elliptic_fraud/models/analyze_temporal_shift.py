
"""
Elliptic Sentinel
Temporal Distribution Shift Diagnostic

Compares time steps 35-49, with special attention to
the transition between steps 42 and 43.

Diagnostic only:
- Does not retrain the model.
- Does not modify the dataset.
- Does not select a model using test results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import ks_2samp
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

GRAPH_PATH = ROOT / "data/processed/elliptic/transaction_graph.pt"
FEATURE_PATH = ROOT / "data/processed/elliptic/scaled_features.pt"
MODEL_PATH = ROOT / "artifacts/xgboost_baseline.json"

OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 10_000


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data():

    print("\nLoading dataset...")

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=True
    )

    features = torch.load(
        FEATURE_PATH,
        map_location="cpu",
        weights_only=True
    )

    if isinstance(features, dict):
        features = features["x"]

    features = features.cpu().numpy()
    labels = graph["y"].cpu().numpy()
    steps = graph["time_step"].cpu().numpy()
    splits = graph["split"].cpu().numpy()

    print("Dataset loaded.")
    print("Transactions:", len(labels))
    print("Features:", features.shape[1])

    return features, labels, steps, splits


# --------------------------------------------------
# 3. CLASS DISTRIBUTION
# --------------------------------------------------

def analyze_class_distribution(labels, steps, splits):

    print("\n" + "=" * 75)
    print("CLASS DISTRIBUTION BY TIME STEP")
    print("=" * 75)

    records = []

    for step in range(35, 50):

        mask = (
            (steps == step)
            & (splits == 2)
            & (labels >= 0)
        )

        y = labels[mask]

        total = len(y)
        illicit = int(np.sum(y == 1))
        licit = int(np.sum(y == 0))

        prevalence = (
            illicit / total * 100
            if total > 0 else 0
        )

        records.append({
            "time_step": step,
            "labelled_transactions": total,
            "licit": licit,
            "illicit": illicit,
            "illicit_percentage": prevalence
        })

        print(
            f"Step {step:2d} | "
            f"Total: {total:5d} | "
            f"Licit: {licit:5d} | "
            f"Illicit: {illicit:4d} | "
            f"Illicit %: {prevalence:7.3f}%"
        )

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_DIR / "temporal_class_distribution.csv",
        index=False
    )

    return result


# --------------------------------------------------
# 4. FEATURE SHIFT: STEP 42 VS STEP 43
# --------------------------------------------------

def compare_feature_distributions(
    features,
    labels,
    steps,
    splits
):

    print("\n" + "=" * 75)
    print("FEATURE DISTRIBUTION: STEP 42 VS STEP 43")
    print("=" * 75)

    records = []

    # Compare all nodes, including unlabelled nodes.
    # This is useful because unlabelled nodes can also
    # contribute to graph message passing.

    mask_42 = steps == 42
    mask_43 = steps == 43

    x42 = features[mask_42]
    x43 = features[mask_43]

    for i in range(features.shape[1]):

        a = x42[:, i]
        b = x43[:, i]

        ks = ks_2samp(a, b)

        records.append({
            "feature": f"feature_{i + 1:03d}",
            "step42_mean": float(np.mean(a)),
            "step43_mean": float(np.mean(b)),
            "mean_difference": float(np.mean(b) - np.mean(a)),
            "step42_std": float(np.std(a)),
            "step43_std": float(np.std(b)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue)
        })

    result = pd.DataFrame(records)

    result = result.sort_values(
        "ks_statistic",
        ascending=False
    )

    result.to_csv(
        OUTPUT_DIR / "temporal_feature_shift_42_vs_43.csv",
        index=False
    )

    print("\nTop 15 features with the largest distribution changes:")

    print(
        result.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Class-conditional comparison for labelled nodes.
    print("\n" + "-" * 75)
    print("CLASS-CONDITIONAL FEATURE COMPARISON")
    print("-" * 75)

    class_records = []

    for class_value, class_name in [(0, "licit"), (1, "illicit")]:

        mask42 = (steps == 42) & (labels == class_value)
        mask43 = (steps == 43) & (labels == class_value)

        if mask42.sum() == 0 or mask43.sum() == 0:
            print(f"Insufficient {class_name} samples.")
            continue

        a = features[mask42]
        b = features[mask43]

        for i in range(features.shape[1]):

            ks = ks_2samp(a[:, i], b[:, i])

            class_records.append({
                "class": class_name,
                "feature": f"feature_{i + 1:03d}",
                "step42_mean": float(np.mean(a[:, i])),
                "step43_mean": float(np.mean(b[:, i])),
                "mean_difference": float(
                    np.mean(b[:, i]) - np.mean(a[:, i])
                ),
                "ks_statistic": float(ks.statistic)
            })

    class_result = pd.DataFrame(class_records)

    class_result.to_csv(
        OUTPUT_DIR / "temporal_class_conditional_shift.csv",
        index=False
    )

    for class_name in ["licit", "illicit"]:

        subset = class_result[
            class_result["class"] == class_name
        ].sort_values(
            "ks_statistic",
            ascending=False
        )

        print(f"\nTop 10 features for {class_name} transactions:")

        print(
            subset.head(10).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )


# --------------------------------------------------
# 5. MODEL SCORE DISTRIBUTION
# --------------------------------------------------

def analyze_prediction_scores(features, labels, steps):

    print("\n" + "=" * 75)
    print("XGBOOST PREDICTION SCORE DISTRIBUTION")
    print("=" * 75)

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    rounds = model.get_booster().num_boosted_rounds()

    print("Saved boosting rounds:", rounds)
    print("Best iteration:", model.best_iteration)

    first_scores = []
    all_scores = []

    for start in range(0, len(features), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(features))
        batch = features[start:end]

        first = model.predict_proba(
            batch,
            iteration_range=(0, 1)
        )[:, 1]

        all_tree = model.predict_proba(
            batch,
            iteration_range=(0, rounds)
        )[:, 1]

        first_scores.append(first)
        all_scores.append(all_tree)

        print(f"Processed {end:,}/{len(features):,}", flush=True)

    first_scores = np.concatenate(first_scores)
    all_scores = np.concatenate(all_scores)

    records = []

    for step in range(35, 50):

        for class_value, class_name in [
            (0, "licit"),
            (1, "illicit")
        ]:

            mask = (
                (steps == step)
                & (labels == class_value)
            )

            if not np.any(mask):
                continue

            for model_name, scores in [
                ("first_tree", first_scores),
                ("all_trees", all_scores)
            ]:

                selected = scores[mask]

                records.append({
                    "time_step": step,
                    "class": class_name,
                    "model": model_name,
                    "count": len(selected),
                    "mean_score": float(np.mean(selected)),
                    "median_score": float(np.median(selected)),
                    "min_score": float(np.min(selected)),
                    "max_score": float(np.max(selected))
                })

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_DIR / "temporal_prediction_scores.csv",
        index=False
    )

    print("\nPrediction score summary:")

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------

def main():

    features, labels, steps, splits = load_data()

    analyze_class_distribution(
        labels,
        steps,
        splits
    )

    compare_feature_distributions(
        features,
        labels,
        steps,
        splits
    )

    analyze_prediction_scores(
        features,
        labels,
        steps
    )

    print("\n" + "=" * 75)
    print("TEMPORAL SHIFT DIAGNOSTIC COMPLETED")
    print("=" * 75)

    print("\nReports saved inside:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
