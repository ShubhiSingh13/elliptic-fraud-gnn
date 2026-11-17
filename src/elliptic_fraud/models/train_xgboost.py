
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = PROJECT_ROOT / "params.yaml"

GRAPH_PATH = (
    PROJECT_ROOT
    / "data/processed/elliptic/transaction_graph.pt"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data/processed/elliptic/scaled_features.pt"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    return config


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

def load_data():

    print("\nLoading configuration...")
    config = load_config()

    print("Loading graph...")

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=True,
    )

    print("Loading scaled features...")

    features = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(features, torch.Tensor):
        raise TypeError(
            "Expected scaled_features.pt to contain a tensor."
        )

    if features.ndim != 2:
        raise ValueError(
            "Feature tensor must be two-dimensional."
        )

    if not torch.isfinite(features).all():
        raise ValueError(
            "Feature tensor contains NaN or infinite values."
        )

    X = features.numpy()

    y = graph["y"].numpy()
    split = graph["split"].numpy()

    # --------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------

    if X.shape[0] != len(y) or len(y) != len(split):
        raise ValueError(
            "Feature, label and split dimensions do not match."
        )

    if X.shape[1] != config["data"]["total_feature_count"]:
        raise ValueError(
            "Unexpected number of input features."
        )

    # --------------------------------------------------
    # Validate labels and split codes
    # --------------------------------------------------

    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError(
            "Unexpected label values found."
        )

    if not np.isin(split, [0, 1, 2]).all():
        raise ValueError(
            "Unexpected split codes found."
        )

    return config, X, y, split


# --------------------------------------------------
# 4. CREATE LABELLED TEMPORAL SPLITS
# --------------------------------------------------

def create_splits(X, y, split):

    # Split codes:
    # 0 = train
    # 1 = validation
    # 2 = test
    #
    # Label -1 = unknown.
    # Unknown labels are excluded from supervised targets.

    train_mask = (
        (split == 0)
        & (y >= 0)
    )

    val_mask = (
        (split == 1)
        & (y >= 0)
    )

    test_mask = (
        (split == 2)
        & (y >= 0)
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # --------------------------------------------------
    # Validate non-empty splits
    # --------------------------------------------------

    if (
        len(X_train) == 0
        or len(X_val) == 0
        or len(X_test) == 0
    ):
        raise ValueError(
            "One or more labelled data splits are empty."
        )

    # --------------------------------------------------
    # Validate class presence
    # --------------------------------------------------

    expected_classes = {0, 1}

    if set(np.unique(y_train)) != expected_classes:
        raise ValueError(
            "Training data must contain both classes."
        )

    if set(np.unique(y_val)) != expected_classes:
        raise ValueError(
            "Validation data must contain both classes."
        )

    if set(np.unique(y_test)) != expected_classes:
        raise ValueError(
            "Test data must contain both classes."
        )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\nDATASET SPLIT SUMMARY")
    print("-" * 40)

    print(f"Training samples:   {len(y_train):,}")
    print(f"Validation samples: {len(y_val):,}")
    print(f"Test samples:       {len(y_test):,}")

    print("\nTRAINING CLASS DISTRIBUTION")
    print("-" * 40)

    print(f"Licit (0):   {(y_train == 0).sum():,}")
    print(f"Illicit (1): {(y_train == 1).sum():,}")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# --------------------------------------------------
# 5. SELECT THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

def select_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to calculate validation thresholds."
        )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(f1_scores))

    best_threshold = float(
        thresholds[best_index]
    )

    best_f1 = float(
        f1_scores[best_index]
    )

    print("\nVALIDATION THRESHOLD SELECTION")
    print("-" * 40)

    print(
        f"Selected threshold: {best_threshold:.6f}"
    )

    print(
        f"Validation F1:      {best_f1:.6f}"
    )

    return best_threshold, best_f1


# --------------------------------------------------
# 6. EVALUATE MODEL
# --------------------------------------------------

def evaluate_model(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = {

        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),

        "threshold": float(threshold),
    }

    return results


# --------------------------------------------------
# 7. MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    config, X, y, split = load_data()

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = create_splits(
        X,
        y,
        split,
    )

    model_config = config["tabular_model"]["xgboost"]
    training_config = config["training"]

    # --------------------------------------------------
    # Calculate class imbalance from training labels only
    # --------------------------------------------------

    negative_count = int(
        np.sum(y_train == 0)
    )

    positive_count = int(
        np.sum(y_train == 1)
    )

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"\nTraining class weight: {scale_pos_weight:.6f}"
    )

    # --------------------------------------------------
    # Model configuration
    # --------------------------------------------------

    random_seed = int(
        training_config["random_seed"]
    )

    early_stopping_rounds = 50

    model = XGBClassifier(

        objective=model_config["objective"],

        eval_metric=model_config["eval_metric"],

        n_estimators=model_config["n_estimators"],

        max_depth=model_config["max_depth"],

        learning_rate=model_config["learning_rate"],

        subsample=model_config["subsample"],

        colsample_bytree=model_config["colsample_bytree"],

        random_state=random_seed,

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",

        device="cpu",

        early_stopping_rounds=early_stopping_rounds,

        n_jobs=1,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nSTARTING XGBOOST TRAINING")
    print("-" * 40)

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        verbose=25,
    )

    best_iteration = int(
        model.best_iteration
    )

    print("\nTraining completed.")

    print(
        f"Best iteration: {best_iteration}"
    )

    # --------------------------------------------------
    # Validation predictions
    # --------------------------------------------------

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    validation_ap = float(
        average_precision_score(
            y_val,
            val_probabilities,
        )
    )

    threshold, validation_f1 = select_threshold(
        y_val,
        val_probabilities,
    )

    # --------------------------------------------------
    # Test evaluation
    # --------------------------------------------------

    print("\nRUNNING TEST EVALUATION")
    print("-" * 40)

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_results = evaluate_model(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\nTEST RESULTS")
    print("-" * 40)

    for metric, value in test_results.items():
        print(f"{metric}: {value}")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = (
        ARTIFACTS_DIR
        / "xgboost_baseline.json"
    )

    model.save_model(model_path)

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------

    metadata = {

        "model": "xgboost",

        "random_seed": random_seed,

        "feature_count": int(X.shape[1]),

        "feature_source": str(
            FEATURES_PATH.relative_to(PROJECT_ROOT)
        ),

        "training_samples": int(len(y_train)),

        "validation_samples": int(len(y_val)),

        "test_samples": int(len(y_test)),

        "training_positive_count": positive_count,

        "training_negative_count": negative_count,

        "scale_pos_weight": float(scale_pos_weight),

        "best_iteration": best_iteration,

        "early_stopping_rounds": early_stopping_rounds,

        "validation_average_precision": validation_ap,

        "validation_selected_f1": validation_f1,

        "validation_selected_threshold": threshold,

        "test_status": "exploratory",

        "hyperparameters": {
            "objective": model_config["objective"],
            "eval_metric": model_config["eval_metric"],
            "n_estimators": model_config["n_estimators"],
            "max_depth": model_config["max_depth"],
            "learning_rate": model_config["learning_rate"],
            "subsample": model_config["subsample"],
            "colsample_bytree": model_config["colsample_bytree"],
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": 1,
        },
    }

    metadata_path = (
        ARTIFACTS_DIR
        / "xgboost_baseline_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # Save test metrics
    # --------------------------------------------------

    results_path = (
        ARTIFACTS_DIR
        / "xgboost_baseline_metrics.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            test_results,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # Save feature importance
    # --------------------------------------------------

    importance = model.feature_importances_

    importance_data = {

        f"feature_{index + 1:03d}": float(value)

        for index, value in enumerate(importance)
    }

    importance_path = (
        ARTIFACTS_DIR
        / "xgboost_feature_importance.json"
    )

    with open(
        importance_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            importance_data,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\nSAVED ARTIFACTS")
    print("-" * 40)

    print(model_path.relative_to(PROJECT_ROOT))
    print(metadata_path.relative_to(PROJECT_ROOT))
    print(results_path.relative_to(PROJECT_ROOT))
    print(importance_path.relative_to(PROJECT_ROOT))

    print("\nXGBOOST BASELINE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
