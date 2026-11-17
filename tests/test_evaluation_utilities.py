
import numpy as np
import pytest

from elliptic_fraud.evaluation.thresholds import (
    select_f1_threshold,
    select_top_fraction,
    select_top_fraction_threshold,
)
from elliptic_fraud.evaluation.metrics import (
    calculate_alert_rate,
    calculate_classification_metrics,
)


# ---------------------------------------------------------
# F1 threshold selection tests
# ---------------------------------------------------------

def test_f1_threshold_uses_binary_validation_labels():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    result = select_f1_threshold(scores, labels)

    assert 0 <= result["threshold"] <= 1
    assert result["method"] == "validation_f1_max"
    assert result["selection_data"] == "labelled_validation_targets_only"
    assert result["validation_f1"] == pytest.approx(1.0)


def test_f1_threshold_rejects_single_class():
    with pytest.raises(ValueError, match="Both classes"):
        select_f1_threshold(
            np.array([0.1, 0.2, 0.3]),
            np.array([0, 0, 0]),
        )


def test_f1_threshold_rejects_invalid_labels():
    with pytest.raises(ValueError, match="binary"):
        select_f1_threshold(
            np.array([0.1, 0.9]),
            np.array([0, -1]),
        )


def test_f1_threshold_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="match"):
        select_f1_threshold(
            np.array([0.1, 0.9]),
            np.array([0, 1, 0]),
        )


# ---------------------------------------------------------
# Deterministic top-fraction capacity selection tests
# ---------------------------------------------------------

def test_top_fraction_selects_exact_count():
    scores = np.linspace(0.01, 1.0, 100)
    identifiers = np.array([f"tx{i:03d}" for i in range(100)])

    result = select_top_fraction(
        scores,
        identifiers,
        fraction=0.01,
    )

    assert result["target_alert_count"] == 1
    assert result["actual_alert_count"] == 1
    assert result["actual_alert_rate"] == pytest.approx(0.01)
    assert result["selection_data"] == (
        "all_period_transactions_without_labels"
    )


def test_top_fraction_selects_exact_count_when_scores_tie():
    scores = np.array([0.1, 0.2, 0.9, 0.9])
    identifiers = np.array(["tx4", "tx3", "tx2", "tx1"])

    result = select_top_fraction(
        scores,
        identifiers,
        fraction=0.25,
    )

    assert result["target_alert_count"] == 1
    assert result["actual_alert_count"] == 1
    assert result["actual_alert_rate"] == pytest.approx(0.25)
    assert result["alert_mask"].sum() == 1

    # Equal scores are resolved by ascending transaction ID.
    assert identifiers[result["alert_mask"]].tolist() == ["tx1"]


def test_top_fraction_is_deterministic():
    scores = np.array([0.8, 0.8, 0.8, 0.2])
    identifiers = np.array(["tx3", "tx1", "tx2", "tx4"])

    first = select_top_fraction(
        scores,
        identifiers,
        fraction=0.5,
    )

    second = select_top_fraction(
        scores,
        identifiers,
        fraction=0.5,
    )

    assert np.array_equal(
        first["alert_mask"],
        second["alert_mask"],
    )

    assert identifiers[first["alert_mask"]].tolist() == [
        "tx1",
        "tx2",
    ]


def test_top_fraction_rejects_duplicate_identifiers():
    scores = np.array([0.8, 0.7, 0.6])
    identifiers = np.array(["tx1", "tx1", "tx2"])

    with pytest.raises(ValueError, match="unique"):
        select_top_fraction(
            scores,
            identifiers,
            fraction=0.5,
        )


def test_top_fraction_rejects_mismatched_identifiers():
    scores = np.array([0.8, 0.7, 0.6])
    identifiers = np.array(["tx1", "tx2"])

    with pytest.raises(ValueError, match="match"):
        select_top_fraction(
            scores,
            identifiers,
            fraction=0.5,
        )


def test_top_fraction_rejects_invalid_fraction():
    scores = np.array([0.1, 0.2, 0.3])
    identifiers = np.array(["tx1", "tx2", "tx3"])

    with pytest.raises(ValueError, match="fraction"):
        select_top_fraction(
            scores,
            identifiers,
            fraction=1.0,
        )


# ---------------------------------------------------------
# Backward-compatible threshold-only selector tests
# ---------------------------------------------------------

def test_top_fraction_threshold_uses_all_scores():
    scores = np.linspace(0.01, 1.0, 100)

    result = select_top_fraction_threshold(
        scores,
        fraction=0.01,
    )

    assert result["target_alert_count"] == 1
    assert result["actual_alert_count"] == 1
    assert result["actual_alert_rate"] == pytest.approx(0.01)
    assert result["selection_data"] == (
        "all_validation_context_nodes_without_labels"
    )


def test_top_fraction_threshold_reports_ties():
    scores = np.array([0.1, 0.2, 0.9, 0.9])

    result = select_top_fraction_threshold(
        scores,
        fraction=0.25,
    )

    assert result["target_alert_count"] == 1
    assert result["actual_alert_count"] == 2
    assert result["actual_alert_rate"] == pytest.approx(0.5)
    assert "Tied scores" in result["warning"]


# ---------------------------------------------------------
# Classification metric tests
# ---------------------------------------------------------

def test_classification_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = calculate_classification_metrics(
        labels,
        scores,
        threshold=0.5,
    )

    assert result["average_precision"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)

    assert result["confusion_matrix"]["true_negative"] == 2
    assert result["confusion_matrix"]["true_positive"] == 2


def test_classification_metrics_reject_invalid_threshold():
    with pytest.raises(ValueError, match="Threshold"):
        calculate_classification_metrics(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            threshold=1.5,
        )


# ---------------------------------------------------------
# Alert-rate tests
# ---------------------------------------------------------

def test_alert_rate_uses_all_transactions():
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = calculate_alert_rate(
        scores,
        threshold=0.8,
    )

    assert result["transaction_count"] == 4
    assert result["alert_count"] == 2
    assert result["alert_rate"] == pytest.approx(0.5)


def test_alert_rate_with_no_alerts():
    scores = np.array([0.1, 0.2, 0.3])

    result = calculate_alert_rate(
        scores,
        threshold=0.9,
    )

    assert result["transaction_count"] == 3
    assert result["alert_count"] == 0
    assert result["alert_rate"] == pytest.approx(0.0)
