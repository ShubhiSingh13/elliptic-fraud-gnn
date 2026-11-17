"""Classification and alert-capacity metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate binary classification metrics at a frozen threshold."""

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional.")

    if labels.shape[0] != scores.shape[0]:
        raise ValueError("Labels and scores must have the same length.")

    if labels.size == 0:
        raise ValueError("Labels and scores cannot be empty.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only binary values 0 and 1.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be probabilities between 0 and 1.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite probability between 0 and 1.")

    if np.unique(labels).size != 2:
        raise ValueError("Both classes must be present for these metrics.")

    predictions = (scores >= threshold).astype(np.int64)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "labels": ["licit_0", "illicit_1"],
            "matrix": matrix.tolist(),
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "sample_count": int(labels.size),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
    }


def calculate_alert_rate(
    all_scores: np.ndarray,
    threshold: float,
) -> dict:
    """Report how many transactions cross a frozen alert threshold."""

    all_scores = np.asarray(all_scores, dtype=np.float64)

    if all_scores.ndim != 1 or all_scores.size == 0:
        raise ValueError("Scores must be a non-empty one-dimensional array.")

    if not np.isfinite(all_scores).all():
        raise ValueError("Scores must contain only finite values.")

    if np.any((all_scores < 0) | (all_scores > 1)):
        raise ValueError("Scores must be probabilities between 0 and 1.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    alert_count = int(np.sum(all_scores >= threshold))

    return {
        "transaction_count": int(all_scores.size),
        "alert_count": alert_count,
        "alert_rate": float(alert_count / all_scores.size),
        "threshold": float(threshold),
    }