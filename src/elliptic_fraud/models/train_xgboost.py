
from __future__ import annotations

import json
from pathlib import Path

import joblib
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
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = PROJECT_ROOT / "params.yaml"
GRAPH_PATH = (
    PROJECT_ROOT / "data/processed/elliptic/transaction_graph.pt"
)
FEATURES_PATH = (
    PROJECT_ROOT / "data/processed/elliptic/scaled_features.pt"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# --------------------------------------------------
# 2. Load graph and preprocessed features
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

    if isinstance(features, dict):
        raise ValueError(
            "Expected scaled_features.pt to contain a feature tensor."
        )

    X = features.numpy()
    y = graph["y"].numpy()
    split = graph["split"].numpy()

    if X.shape[0] != len(y) or len(y) != len(split):
        raise ValueError("Feature, label and split dimensions do not match.")

    if X.shape[1] != config["data"]["total_feature_count"]:
        raise ValueError("Unexpected number of input features.")

    return config, X, y, split


# --------------------------------------------------
# 3. Create labelled training, validation and test sets
# --------------------------------------------------

def create_splits(X, y, split):
    # Split codes:
    # 0 = train, 1 = validation, 2 = test
    # Label -1 = unknown; exclude from supervised targets.

    train_mask = (split == 0) & (y >= 0)
    val_mask = (split == 1) & (y >= 0)
    test_mask = (split == 2) & (y >= 0)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("One or more labelled data splits are empty.")

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training data must contain both classes.")

    print("\nDataset split summary")
    print("---------------------")
    print(f"Training samples:   {len(y_train):,}")
    print(f"Validation samples: {len(y_val):,}")
    print(f"Test samples:       {len(y_test):,}")

    print("\nTraining class distribution")
    print("---------------------------")
    print(f"Licit (0):   {np.sum(y_train == 0):,}")
    print(f"Illicit (1): {np.sum(y_train == 1):,}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------------------------------------------
# 4. Select classification threshold using validation
# --------------------------------------------------

def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to calculate validation thresholds.")

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_index])
    best_f1 = float(f1_scores[best_index])

    print("\nValidation threshold selection")
    print("------------------------------")
    print(f"Selected threshold: {best_threshold:.6f}")
    print(f"Validation F1:      {best_f1:.6f}")

    return best_threshold


# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

def evaluate_model(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    results = {
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "threshold": float(threshold),
    }

    return results


# --------------------------------------------------
# 6. Main training pipeline
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
    ) = create_splits(X, y, split)

    model_config = config["tabular_model"]["xgboost"]
    training_config = config["training"]

    # Calculate class imbalance using training labels only.
    negative_count = int(np.sum(y_train == 0))
    positive_count = int(np.sum(y_train == 1))

    scale_pos_weight = negative_count / positive_count

    print(f"\nTraining class weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        objective=model_config["objective"],
        eval_metric=model_config["eval_metric"],
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        learning_rate=model_config["learning_rate"],
        subsample=model_config["subsample"],
        colsample_bytree=model_config["colsample_bytree"],
        random_state=training_config["random_seed"],
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cpu",
        early_stopping_rounds=50,
        n_jobs=1,
    )

    print("\nStarting XGBoost training...")
    print("----------------------------")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=25,
    )

    print("\nTraining completed.")
    print(f"Best iteration: {model.best_iteration}")

    # Validation predictions: used to select threshold.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_probabilities)

    # Test predictions: final evaluation only.
    print("\nRunning final test evaluation...")
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_results = evaluate_model(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\nFinal test results")
    print("------------------")

    for metric, value in test_results.items():
        print(f"{metric}: {value}")

    # Save model.
    model_path = ARTIFACTS_DIR / "xgboost_baseline.json"
    model.save_model(model_path)

    # Save selected threshold and training metadata.
    metadata = {
        "model": "xgboost",
        "random_seed": training_config["random_seed"],
        "feature_count": int(X.shape[1]),
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration),
        "validation_selected_threshold": threshold,
    }

    metadata_path = ARTIFACTS_DIR / "xgboost_baseline_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Save final test metrics.
    results_path = ARTIFACTS_DIR / "xgboost_baseline_metrics.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(test_results, file, indent=4)

    # Save feature importance.
    importance = model.feature_importances_

    importance_data = {
        f"feature_{index + 1:03d}": float(value)
        for index, value in enumerate(importance)
    }

    importance_path = ARTIFACTS_DIR / "xgboost_feature_importance.json"
    with open(importance_path, "w", encoding="utf-8") as file:
        json.dump(importance_data, file, indent=4)

    print("\nSaved artifacts")
    print("---------------")
    print(model_path.relative_to(PROJECT_ROOT))
    print(metadata_path.relative_to(PROJECT_ROOT))
    print(results_path.relative_to(PROJECT_ROOT))
    print(importance_path.relative_to(PROJECT_ROOT))

    print("\nXGBoost baseline pipeline completed successfully.")


if __name__ == "__main__":
    main()
