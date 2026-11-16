
"""
Elliptic Sentinel
XGBoost Temporal Performance Diagnostic

Purpose:
Compare the first boosting tree against all saved boosting
trees across validation and test time steps.

This is a diagnostic script only.
It does not retrain or modify the saved model.
Test results are exploratory and must not be used for
model selection or hyperparameter tuning.
"""

from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------
# 1. PATH CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "artifacts" / "xgboost_baseline.json"
GRAPH_PATH = (
    ROOT / "data" / "processed" / "elliptic"
    / "transaction_graph.pt"
)
FEATURE_PATH = (
    ROOT / "data" / "processed" / "elliptic"
    / "scaled_features.pt"
)

BATCH_SIZE = 10_000


# ---------------------------------------------------------
# 2. LOAD MODEL AND DATA
# ---------------------------------------------------------

def load_data():
    print("\nLoading XGBoost model and graph data...")

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

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

    features = (
        features.cpu().numpy()
        if isinstance(features, torch.Tensor)
        else np.asarray(features)
    )

    labels = graph["y"].cpu().numpy()
    time_steps = graph["time_step"].cpu().numpy()
    splits = graph["split"].cpu().numpy()

    rounds = model.get_booster().num_boosted_rounds()

    print("\nModel information")
    print("-" * 50)
    print(f"Total boosting rounds: {rounds}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Feature shape: {features.shape}")
    print(f"Total transactions: {len(features):,}")

    return model, features, labels, time_steps, splits, rounds


# ---------------------------------------------------------
# 3. BATCH PREDICTION
# ---------------------------------------------------------

def predict_in_batches(model, features, rounds):
    """
    Generate probabilities in smaller batches to reduce
    native-library resource pressure.
    """

    predictions = []

    total = len(features)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch = features[start:end]

        batch_predictions = model.predict_proba(
            batch,
            iteration_range=(0, rounds)
        )[:, 1]

        predictions.append(batch_predictions)

        print(
            f"Processed {end:,}/{total:,} transactions",
            flush=True
        )

    return np.concatenate(predictions)


# ---------------------------------------------------------
# 4. TEMPORAL PERFORMANCE ANALYSIS
# ---------------------------------------------------------

def report_period(
    name,
    split_code,
    start_step,
    end_step,
    labels,
    time_steps,
    splits,
    first_scores,
    all_scores
):

    print("\n" + "=" * 80)
    print(f"{name}: TIME STEPS {start_step}-{end_step}")
    print("=" * 80)

    print(
        f"{'Step':<10}"
        f"{'Nodes':<12}"
        f"{'Illicit':<12}"
        f"{'First AP':<14}"
        f"{'All-tree AP':<14}"
        f"{'First ROC':<14}"
        f"{'All ROC':<14}"
    )

    first_aps = []
    all_aps = []

    first_rocs = []
    all_rocs = []

    for step in range(start_step, end_step + 1):

        mask = (
            (splits == split_code)
            & (time_steps == step)
            & (labels >= 0)
        )

        step_labels = labels[mask].astype(int)

        if len(step_labels) == 0:
            print(f"{step:<10}No labelled transactions")
            continue

        if len(np.unique(step_labels)) < 2:
            print(
                f"{step:<10}{len(step_labels):<12}"
                "Only one class present"
            )
            continue

        first_ap = average_precision_score(
            step_labels,
            first_scores[mask]
        )

        all_ap = average_precision_score(
            step_labels,
            all_scores[mask]
        )

        first_roc = roc_auc_score(
            step_labels,
            first_scores[mask]
        )

        all_roc = roc_auc_score(
            step_labels,
            all_scores[mask]
        )

        first_aps.append(first_ap)
        all_aps.append(all_ap)

        first_rocs.append(first_roc)
        all_rocs.append(all_roc)

        illicit_count = int(np.sum(step_labels == 1))

        print(
            f"{step:<10}"
            f"{len(step_labels):<12,}"
            f"{illicit_count:<12,}"
            f"{first_ap:<14.4f}"
            f"{all_ap:<14.4f}"
            f"{first_roc:<14.4f}"
            f"{all_roc:<14.4f}"
        )

    print("-" * 80)

    if first_aps:

        print(
            f"Mean per-step AP (first tree): "
            f"{np.mean(first_aps):.4f}"
        )

        print(
            f"Mean per-step AP (all trees):  "
            f"{np.mean(all_aps):.4f}"
        )

        print(
            f"Mean per-step ROC-AUC (first): "
            f"{np.mean(first_rocs):.4f}"
        )

        print(
            f"Mean per-step ROC-AUC (all):   "
            f"{np.mean(all_rocs):.4f}"
        )


# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------

def main():

    (
        model,
        features,
        labels,
        time_steps,
        splits,
        rounds
    ) = load_data()

    # First boosting tree
    print("\nGenerating first-tree predictions...")

    first_scores = predict_in_batches(
        model,
        features,
        rounds=1
    )

    # All saved boosting trees
    print("\nGenerating all-tree predictions...")

    all_scores = predict_in_batches(
        model,
        features,
        rounds=rounds
    )

    # Validation: steps 30-34
    report_period(
        name="VALIDATION",
        split_code=1,
        start_step=30,
        end_step=34,
        labels=labels,
        time_steps=time_steps,
        splits=splits,
        first_scores=first_scores,
        all_scores=all_scores
    )

    # Test: steps 35-49
    # Diagnostic only. Do not use these results to select models.
    report_period(
        name="TEST - EXPLORATORY DIAGNOSTIC ONLY",
        split_code=2,
        start_step=35,
        end_step=49,
        labels=labels,
        time_steps=time_steps,
        splits=splits,
        first_scores=first_scores,
        all_scores=all_scores
    )

    print("\n" + "=" * 80)
    print("TEMPORAL DIAGNOSTIC COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
