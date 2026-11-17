
"""Validation-only threshold and deterministic capacity-selection utilities."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import precision_recall_curve


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 1:
        raise ValueError("Scores must be a one-dimensional array.")

    if scores.size == 0:
        raise ValueError("Scores cannot be empty.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be probabilities between 0 and 1.")

    return scores


def select_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Select the F1-maximizing threshold using labelled validation data."""

    scores = _validate_scores(scores)
    labels = np.asarray(labels)

    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("Labels must be 1D and match the scores.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only binary values 0 and 1.")

    if np.unique(labels).size != 2:
        raise ValueError("Both classes must be present for F1 threshold selection.")

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    if thresholds.size == 0:
        raise ValueError("Could not calculate any classification thresholds.")

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall
    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_f1 = np.max(f1_scores)
    best_indices = np.flatnonzero(np.isclose(f1_scores, best_f1))
    best_index = best_indices[-1]

    return {
        "method": "validation_f1_max",
        "threshold": float(thresholds[best_index]),
        "validation_f1": float(f1_scores[best_index]),
        "validation_precision": float(precision[best_index]),
        "validation_recall": float(recall[best_index]),
        "selection_data": "labelled_validation_targets_only",
    }


def select_top_fraction(
    scores: np.ndarray,
    identifiers: np.ndarray,
    fraction: float = 0.01,
) -> dict:
    """
    Select exactly the requested top fraction.

    Transactions are ranked by descending score. Equal scores are resolved
    by ascending string representation of their stable transaction IDs.
    Labels are never used.
    """

    scores = _validate_scores(scores)
    identifiers = np.asarray(identifiers)

    if identifiers.ndim != 1 or identifiers.size != scores.size:
        raise ValueError("Identifiers must be 1D and match the scores.")

    if not 0 < fraction < 1:
        raise ValueError("fraction must be greater than 0 and less than 1.")

    stable_ids = np.asarray([str(value) for value in identifiers])

    if np.unique(stable_ids).size != stable_ids.size:
        raise ValueError("Transaction identifiers must be unique.")

    number_to_flag = max(1, math.ceil(scores.size * fraction))

    # Primary sort: descending score.
    # Tie-break: ascending stable transaction identifier.
    order = np.lexsort((stable_ids, -scores))
    selected_indices = order[:number_to_flag]

    alert_mask = np.zeros(scores.size, dtype=bool)
    alert_mask[selected_indices] = True

    threshold = float(np.min(scores[selected_indices]))

    return {
        "method": "deterministic_top_fraction",
        "requested_fraction": float(fraction),
        "threshold": threshold,
        "transaction_count": int(scores.size),
        "target_alert_count": int(number_to_flag),
        "actual_alert_count": int(alert_mask.sum()),
        "actual_alert_rate": float(alert_mask.mean()),
        "selection_data": "all_period_transactions_without_labels",
        "tie_break": "ascending_transaction_id",
        "alert_mask": alert_mask,
        "selected_indices": selected_indices,
    }


def select_top_fraction_threshold(
    scores: np.ndarray,
    fraction: float = 0.01,
) -> dict:
    """
    Backward-compatible threshold-only selector.

    New capacity-constrained evaluation should use select_top_fraction()
    with transaction identifiers for deterministic tie handling.
    """

    scores = _validate_scores(scores)

    if not 0 < fraction < 1:
        raise ValueError("fraction must be greater than 0 and less than 1.")

    number_to_flag = max(1, math.ceil(scores.size * fraction))
    threshold = float(
        np.partition(scores, scores.size - number_to_flag)[
            scores.size - number_to_flag
        ]
    )

    predicted_alerts = scores >= threshold

    return {
        "method": "validation_top_fraction_threshold_only",
        "requested_fraction": float(fraction),
        "threshold": threshold,
        "validation_transaction_count": int(scores.size),
        "target_alert_count": int(number_to_flag),
        "actual_alert_count": int(predicted_alerts.sum()),
        "actual_alert_rate": float(predicted_alerts.mean()),
        "selection_data": "all_validation_context_nodes_without_labels",
        "warning": "Tied scores can exceed the requested alert capacity.",
    }
