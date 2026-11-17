
"""Five-seed evaluation for Elliptic Sentinel Directed GraphSAGE.

Methodology:
- Five final checkpoints are evaluated independently.
- F1 thresholds are selected using labelled validation targets only.
- Capacity alerts use deterministic top-k ranking with stable transaction-ID
  tie-breaking. Labels are never used for capacity selection.
- Test graph view is constructed only after validation calibration is frozen.
- Test metrics are exploratory, not untouched holdout results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from elliptic_fraud.evaluation.thresholds import (
    select_f1_threshold,
    select_top_fraction,
)
from elliptic_fraud.models.gnn import DirectedGraphSAGE
from elliptic_fraud.training.graph_views import build_asof_graph_view
from elliptic_fraud.training.train_gnn import (
    load_graph_data,
    resolve_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = PROJECT_ROOT / "params.yaml"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "gnn"
FINAL_SEED_DIR = OUTPUT_DIR / "final_five_seed"
TRAINING_SUMMARY_FILE = (
    FINAL_SEED_DIR / "five_seed_training_summary.json"
)


# =========================================================
# 1. CONFIGURATION AND COMPATIBILITY HELPERS
# =========================================================

def load_config(config_path: Path = CONFIG_FILE) -> dict[str, Any]:
    """Load project configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def load_selected_checkpoint(
    summary_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Backward-compatible loader for the earlier LR-pilot summary.

    The final five-seed evaluator does not use this function for model
    selection. It remains available for older tests and utilities.
    """
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Learning-rate pilot summary not found: {summary_path}"
        )

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("status") != "completed":
        raise ValueError("Learning-rate pilot is not completed.")

    if summary.get("test_set_used") is not False:
        raise ValueError(
    "The test set was excluded from pilot selection: confirmation failed."
    )

    selected_lr = summary.get("selected_learning_rate")
    results = summary.get("results", [])

    if selected_lr is None:
        raise ValueError("Pilot summary has no selected learning rate.")

    selected_results = [
        item
        for item in results
        if np.isclose(
            float(item["learning_rate"]),
            float(selected_lr),
        )
    ]

    if not selected_results:
        raise ValueError(
            "Selected learning rate has no matching checkpoint result."
        )

    result = selected_results[0]
    checkpoint_path = Path(result["checkpoint"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Selected checkpoint not found: {checkpoint_path}"
        )

    return result, checkpoint_path


def load_final_training_summary(
    summary_path: Path = TRAINING_SUMMARY_FILE,
) -> dict[str, Any]:
    """Validate the final five-seed training summary."""
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Five-seed training summary not found: {summary_path}"
        )

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("status") != "completed":
        raise ValueError("Final five-seed training is not completed.")

    if summary.get("test_graph_view_constructed") is not False:
        raise ValueError(
            "Training summary does not confirm test graph-view exclusion."
        )

    if summary.get("test_set_used_for_training_or_selection") is not False:
        raise ValueError(
            "Training summary does not confirm test-set exclusion."
        )

    seeds = summary.get("seeds")
    results = summary.get("results")
    learning_rate = summary.get("learning_rate")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Training summary has no seed list.")

    if not isinstance(results, list) or not results:
        raise ValueError("Training summary has no seed results.")

    if learning_rate is None or float(learning_rate) <= 0:
        raise ValueError("Training summary has an invalid learning rate.")

    result_seeds = sorted(int(item["seed"]) for item in results)
    expected_seeds = sorted(int(seed) for seed in seeds)

    if result_seeds != expected_seeds:
        raise ValueError("Training summary seed list does not match results.")

    return summary


# =========================================================
# 2. MODEL LOADING AND PREDICTION
# =========================================================

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DirectedGraphSAGE, dict[str, Any]]:
    """Load a Directed GraphSAGE checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    required_keys = {
        "model_state_dict",
        "model_config",
        "seed",
        "learning_rate",
        "best_epoch",
        "best_validation_average_precision",
    }

    missing = required_keys.difference(checkpoint.keys())

    if missing:
        raise ValueError(
            f"Checkpoint is missing keys: {sorted(missing)}"
        )

    model_config = checkpoint["model_config"]

    model = DirectedGraphSAGE(
        in_channels=int(model_config["in_channels"]),
        hidden_channels=int(model_config["hidden_channels"]),
        dropout=float(model_config["dropout"]),
        use_degree_features=bool(
            model_config["use_degree_features"]
        ),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_probabilities(
    model: DirectedGraphSAGE,
    graph_view: dict[str, torch.Tensor],
) -> np.ndarray:
    """Generate one probability per graph node."""
    model.eval()

    with torch.no_grad():
        logits = model(
            graph_view["x"],
            graph_view["edge_index"],
        )
        probabilities = torch.sigmoid(logits)

    values = probabilities.detach().cpu().numpy().reshape(-1)

    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    if np.any((values < 0) | (values > 1)):
        raise ValueError("Predictions must be probabilities in [0, 1].")

    return values.astype(np.float64)


# =========================================================
# 3. DATA AND MASK VALIDATION
# =========================================================

def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array."""
    return value.detach().cpu().numpy()


def move_graph_view_to_device(
    graph_view: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move tensor fields in a graph view to the requested device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in graph_view.items()
    }


def build_period_mask(
    graph_view: dict[str, torch.Tensor],
    start_time_step: int,
    end_time_step: int,
) -> np.ndarray:
    """Select new transactions in a specified time-step range."""
    if start_time_step < 1 or end_time_step < start_time_step:
        raise ValueError("Invalid time-step range.")

    required = {"time_step", "context_mask"}

    if not required.issubset(graph_view):
        raise KeyError(
            f"Graph view is missing keys: {sorted(required.difference(graph_view))}"
        )

    time_steps = tensor_to_numpy(
        graph_view["time_step"]
    ).reshape(-1)

    context_mask = tensor_to_numpy(
        graph_view["context_mask"]
    ).astype(bool).reshape(-1)

    if len(time_steps) != len(context_mask):
        raise ValueError("Time-step and context-mask lengths differ.")

    mask = (
        (time_steps >= start_time_step)
        & (time_steps <= end_time_step)
        & context_mask
    )

    if not mask.any():
        raise ValueError(
            f"No transactions found in steps "
            f"{start_time_step}-{end_time_step}."
        )

    return mask


def validate_target_period(
    target_mask: np.ndarray,
    period_mask: np.ndarray,
    period_name: str,
) -> None:
    """Ensure labelled targets belong to the requested period."""
    if target_mask.shape != period_mask.shape:
        raise ValueError(
            f"{period_name} target and period masks have different shapes."
        )

    if not target_mask.any():
        raise ValueError(
            f"{period_name} contains no labelled target transactions."
        )

    if np.any(target_mask & ~period_mask):
        raise ValueError(
            f"{period_name} targets include transactions outside the period."
        )


def validate_binary_labels(
    labels: np.ndarray,
    target_mask: np.ndarray,
    period_name: str,
) -> None:
    """Validate labels for supervised metrics."""
    selected = labels[target_mask]

    if not np.isin(selected, [0, 1]).all():
        raise ValueError(
            f"{period_name} targets contain non-binary labels."
        )

    if np.unique(selected).size != 2:
        raise ValueError(
            f"{period_name} targets must contain both classes."
        )


def validate_transaction_ids(
    transaction_ids: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    """Validate transaction identifiers used for deterministic tie-breaking."""
    transaction_ids = np.asarray(transaction_ids).reshape(-1)

    if len(transaction_ids) != expected_length:
        raise ValueError(
            "Transaction ID count does not match prediction count."
        )

    if not np.isfinite(
        np.asarray(transaction_ids, dtype=np.float64)
    ).all():
        raise ValueError("Transaction IDs contain invalid values.")

    return transaction_ids


# =========================================================
# 4. METRICS AND CAPACITY ALERTS
# =========================================================

def calculate_ranked_alert_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    alert_mask: np.ndarray,
) -> dict[str, Any]:
    """Calculate ranking and classification metrics for selected alerts."""
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    alert_mask = np.asarray(alert_mask, dtype=bool).reshape(-1)

    if not (
        len(labels) == len(scores) == len(alert_mask)
    ):
        raise ValueError("Labels, scores and alert mask lengths differ.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Metric labels must be binary.")

    if np.unique(labels).size != 2:
        raise ValueError("Both classes must be present for ranking metrics.")

    true_positive = int(np.sum(alert_mask & (labels == 1)))
    false_positive = int(np.sum(alert_mask & (labels == 0)))
    false_negative = int(np.sum(~alert_mask & (labels == 1)))
    true_negative = int(np.sum(~alert_mask & (labels == 0)))

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(
            (true_positive + true_negative) / len(labels)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "sample_count": int(len(labels)),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "alert_count": int(alert_mask.sum()),
        "alert_rate": float(alert_mask.mean()),
    }


def calculate_threshold_alert_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate metrics using a frozen probability threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    alert_mask = np.asarray(scores) >= threshold

    return calculate_ranked_alert_metrics(
        labels,
        scores,
        alert_mask,
    )


def calculate_alert_rate_from_mask(
    alert_mask: np.ndarray,
) -> dict[str, Any]:
    """Calculate alert rate from a deterministic Boolean selection."""
    alert_mask = np.asarray(alert_mask, dtype=bool).reshape(-1)

    if alert_mask.size == 0:
        raise ValueError("Alert mask cannot be empty.")

    return {
        "transaction_count": int(alert_mask.size),
        "alert_count": int(alert_mask.sum()),
        "alert_rate": float(alert_mask.mean()),
    }


def aggregate_metric_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate numeric metrics across random seeds."""
    if not results:
        raise ValueError("No metric results to aggregate.")

    keys = [
        key
        for key, value in results[0].items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]

    aggregate: dict[str, Any] = {}

    for key in keys:
        values = np.asarray(
            [float(item[key]) for item in results],
            dtype=np.float64,
        )

        aggregate[key] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "per_seed": [float(value) for value in values],
        }

    return aggregate


# =========================================================
# 5. FIVE-SEED EVALUATION
# =========================================================

def run_evaluation(
    config_path: Path = CONFIG_FILE,
    output_dir: Path = OUTPUT_DIR,
    training_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Calibrate on validation and evaluate all five final checkpoints."""

    config = load_config(config_path)

    if training_summary_path is None:
        training_summary_path = (
            output_dir
            / "final_five_seed"
            / "five_seed_training_summary.json"
        )

    training_summary = load_final_training_summary(
        training_summary_path
    )

    requested_device = str(
        config.get("gnn", {}).get("device", "cpu")
    )
    device = resolve_device(requested_device)

    seeds = [int(seed) for seed in training_summary["seeds"]]
    learning_rate = float(training_summary["learning_rate"])

    validation_start = int(
        config["split"]["validation"]["start_time_step"]
    )
    validation_end = int(
        config["split"]["validation"]["end_time_step"]
    )
    test_start = int(
        config["split"]["test"]["start_time_step"]
    )
    test_end = int(
        config["split"]["test"]["end_time_step"]
    )

    alert_fraction = float(
        config.get("evaluation", {}).get("alert_fraction", 0.01)
    )

    if not 0 < alert_fraction < 1:
        raise ValueError("Alert fraction must be greater than 0 and less than 1.")

    # -----------------------------------------------------
    # Load graph data, but do not construct test graph view.
    # -----------------------------------------------------

    graph = load_graph_data(device)

    transaction_ids = validate_transaction_ids(
        tensor_to_numpy(graph["tx_id"]),
        len(graph["y"]),
    )

    validation_view = build_asof_graph_view(
        graph,
        "validation",
    )
    validation_view = move_graph_view_to_device(
        validation_view,
        device,
    )

    validation_period_mask = build_period_mask(
        validation_view,
        validation_start,
        validation_end,
    )

    validation_target_mask = tensor_to_numpy(
        validation_view["target_mask"]
    ).astype(bool).reshape(-1)

    validation_labels = tensor_to_numpy(
        validation_view["y"]
    ).reshape(-1)

    validate_target_period(
        validation_target_mask,
        validation_period_mask,
        "Validation",
    )
    validate_binary_labels(
        validation_labels,
        validation_target_mask,
        "Validation",
    )

    validation_ids = transaction_ids[validation_period_mask]

    # -----------------------------------------------------
    # Validation predictions and threshold calibration.
    # -----------------------------------------------------

    validation_results: list[dict[str, Any]] = []
    validation_models: list[dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("ELLIPTIC SENTINEL — FIVE-SEED VALIDATION CALIBRATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print("Test graph view constructed: NO")

    for seed in seeds:
        checkpoint_path = (
            output_dir
            / "final_five_seed"
            / f"gnn_seed_{seed}_lr_{learning_rate:g}.pt"
        )

        model, checkpoint = load_model_from_checkpoint(
            checkpoint_path,
            device,
        )

        if int(checkpoint["seed"]) != seed:
            raise ValueError(
                f"Checkpoint seed mismatch: expected {seed}."
            )

        if not np.isclose(
            float(checkpoint["learning_rate"]),
            learning_rate,
        ):
            raise ValueError(
                f"Checkpoint learning-rate mismatch for seed {seed}."
            )

        probabilities = predict_probabilities(
            model,
            validation_view,
        )

        if len(probabilities) != len(validation_period_mask):
            raise ValueError(
                f"Validation prediction length mismatch for seed {seed}."
            )

        # F1 threshold: labelled validation targets only.
        f1_result = select_f1_threshold(
            probabilities[validation_target_mask],
            validation_labels[validation_target_mask],
        )

        # Capacity ranking: every new validation-period transaction,
        # including transactions with unknown labels.
        capacity_result = select_top_fraction(
            probabilities[validation_period_mask],
            validation_ids,
            fraction=alert_fraction,
        )

        validation_results.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_validation_average_precision": float(
                    checkpoint["best_validation_average_precision"]
                ),
                "f1_threshold": float(f1_result["threshold"]),
                "f1_calibration": f1_result,
                "capacity_calibration": {
                    key: value
                    for key, value in capacity_result.items()
                    if key not in {"alert_mask", "selected_indices"}
                },
            }
        )

        validation_models.append(
            {
                "seed": seed,
                "model": model,
                "checkpoint": checkpoint,
                "f1_threshold": float(f1_result["threshold"]),
            }
        )

        print(
            f"Seed {seed} | "
            f"Validation AP "
            f"{checkpoint['best_validation_average_precision']:.5f} | "
            f"F1 threshold {f1_result['threshold']:.6f} | "
            f"Capacity alerts "
            f"{capacity_result['actual_alert_count']}/"
            f"{capacity_result['transaction_count']}"
        )

    # Freeze validation-derived F1 thresholds.
    frozen_thresholds = {
        str(item["seed"]): {
            "f1_threshold": item["f1_threshold"],
            "capacity_fraction": alert_fraction,
            "capacity_tie_break": "ascending_transaction_id",
        }
        for item in validation_results
    }

    print("\nValidation calibration complete.")
    print("F1 thresholds frozen. Capacity policy fixed at the configured fraction.")
    print("Constructing test graph view now.")

    # -----------------------------------------------------
    # Construct test graph view only after validation.
    # -----------------------------------------------------

    test_view = build_asof_graph_view(
        graph,
        "test",
    )
    test_view = move_graph_view_to_device(
        test_view,
        device,
    )

    test_period_mask = build_period_mask(
        test_view,
        test_start,
        test_end,
    )

    test_target_mask = tensor_to_numpy(
        test_view["target_mask"]
    ).astype(bool).reshape(-1)

    test_labels = tensor_to_numpy(
        test_view["y"]
    ).reshape(-1)

    validate_target_period(
        test_target_mask,
        test_period_mask,
        "Test",
    )
    validate_binary_labels(
        test_labels,
        test_target_mask,
        "Test",
    )

    test_ids = transaction_ids[test_period_mask]

    # -----------------------------------------------------
    # Test predictions and frozen-policy evaluation.
    # -----------------------------------------------------

    per_seed_test_results: list[dict[str, Any]] = []
    test_probabilities_by_seed: list[np.ndarray] = []

    for item in validation_models:
        seed = int(item["seed"])
        model = item["model"]
        checkpoint = item["checkpoint"]
        f1_threshold = float(item["f1_threshold"])

        test_probabilities = predict_probabilities(
            model,
            test_view,
        )

        if len(test_probabilities) != len(test_period_mask):
            raise ValueError(
                f"Test prediction length mismatch for seed {seed}."
            )

        test_probabilities_by_seed.append(test_probabilities)

        # Supervised classification metrics use labelled test targets only.
        test_target_scores = test_probabilities[test_target_mask]
        test_target_labels = test_labels[test_target_mask]

        f1_metrics = calculate_threshold_alert_metrics(
            test_target_labels,
            test_target_scores,
            f1_threshold,
        )

        # Capacity ranking uses all new test-period transactions.
        # It does not use test labels or a test-selected probability cutoff.
        capacity_result = select_top_fraction(
            test_probabilities[test_period_mask],
            test_ids,
            fraction=alert_fraction,
        )

        capacity_metrics = calculate_ranked_alert_metrics(
            test_target_labels,
            test_target_scores,
            capacity_result["alert_mask"][
                np.isin(
                    np.arange(len(test_period_mask))[test_period_mask],
                    np.flatnonzero(test_target_mask),
                )
            ] if False else _target_capacity_mask(
                test_period_mask,
                test_target_mask,
                capacity_result["alert_mask"],
            ),
        )

        # The helper maps period-local top-k selections to labelled targets.
        capacity_alert_rate = calculate_alert_rate_from_mask(
            capacity_result["alert_mask"]
        )

        per_seed_test_results.append(
            {
                "seed": seed,
                "checkpoint": str(
                    validation_results[seeds.index(seed)]["checkpoint"]
                ),
                "best_epoch": int(checkpoint["best_epoch"]),
                "f1_threshold": f1_threshold,
                "f1_threshold_metrics": f1_metrics,
                "f1_threshold_alert_rate": {
                    "transaction_count": int(test_period_mask.sum()),
                    "alert_count": int(
                        np.sum(
                            test_probabilities[test_period_mask]
                            >= f1_threshold
                        )
                    ),
                    "alert_rate": float(
                        np.mean(
                            test_probabilities[test_period_mask]
                            >= f1_threshold
                        )
                    ),
                },
                "capacity_metrics": capacity_metrics,
                "capacity_alert_rate": capacity_alert_rate,
                "capacity_selection": {
                    key: value
                    for key, value in capacity_result.items()
                    if key not in {"alert_mask", "selected_indices"}
                },
            }
        )

        print(
            f"Seed {seed} | "
            f"Test AP {f1_metrics['average_precision']:.5f} | "
            f"Capacity alerts "
            f"{capacity_result['actual_alert_count']}/"
            f"{capacity_result['transaction_count']}"
        )

        model.cpu()

    # -----------------------------------------------------
    # Aggregate results.
    # -----------------------------------------------------

    f1_metrics = [
        item["f1_threshold_metrics"]
        for item in per_seed_test_results
    ]
    capacity_metrics = [
        item["capacity_metrics"]
        for item in per_seed_test_results
    ]

    f1_alert_rates = [
        item["f1_threshold_alert_rate"]
        for item in per_seed_test_results
    ]
    capacity_alert_rates = [
        item["capacity_alert_rate"]
        for item in per_seed_test_results
    ]

    # Descriptive mean probability across seeds.
    ensemble_probabilities = np.mean(
        np.stack(test_probabilities_by_seed, axis=0),
        axis=0,
    )

    if not np.isfinite(ensemble_probabilities).all():
        raise ValueError("Ensemble predictions contain invalid values.")

    # -----------------------------------------------------
    # Save summary and transaction-level predictions.
    # -----------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "project": "Elliptic Sentinel",
        "evaluation_status": "completed",
        "test_status": "exploratory",
        "test_holdout_warning": (
            "Test-period results are exploratory because test data "
            "was inspected during development. They are not results "
            "from an untouched holdout."
        ),
        "device": str(device),
        "experiment": "final_five_seed_directed_graphsage",
        "seeds": seeds,
        "learning_rate": learning_rate,
        "validation_calibration": {
            "period": {
                "start_time_step": validation_start,
                "end_time_step": validation_end,
                "transaction_count": int(validation_period_mask.sum()),
                "labelled_target_count": int(validation_target_mask.sum()),
            },
            "per_seed": validation_results,
            "frozen_thresholds": frozen_thresholds,
        },
        "test_evaluation": {
            "period": {
                "start_time_step": test_start,
                "end_time_step": test_end,
                "transaction_count": int(test_period_mask.sum()),
                "labelled_target_count": int(test_target_mask.sum()),
            },
            "per_seed": per_seed_test_results,
            "f1_threshold_metrics_mean_and_std": aggregate_metric_results(
                f1_metrics
            ),
            "capacity_metrics_mean_and_std": aggregate_metric_results(
                capacity_metrics
            ),
            "f1_alert_rate_mean_and_std": aggregate_metric_results(
                f1_alert_rates
            ),
            "capacity_alert_rate_mean_and_std": aggregate_metric_results(
                capacity_alert_rates
            ),
        },
        "capacity_policy": {
            "fraction": alert_fraction,
            "selection": "deterministic_top_k",
            "ranking": "descending_probability",
            "tie_break": "ascending_transaction_id",
            "uses_labels": False,
            "validation_and_test": (
                "The same predeclared capacity fraction is applied "
                "independently to each period."
            ),
        },
        "ensemble": {
            "description": (
                "Mean probability across five seeds. Saved for "
                "descriptive analysis only; no ensemble threshold "
                "was selected or applied."
            )
        },
        "methodology_notes": [
            "Each seed uses its own validation-calibrated F1 threshold.",
            "F1 thresholds use labelled validation targets only.",
            "Capacity ranking uses all new transactions in each period.",
            "Capacity ranking does not use labels.",
            "Capacity selection breaks score ties by ascending transaction ID.",
            "The test graph view is constructed after validation calibration.",
            "Test metrics are exploratory, not untouched holdout results.",
            "Seed variability is not temporal generalisation uncertainty.",
        ],
    }

    summary_path = output_dir / "five_seed_evaluation_summary.json"
    prediction_path = output_dir / "five_seed_test_transaction_scores.npz"

    np.savez_compressed(
        prediction_path,
        tx_id=test_ids,
        time_step=tensor_to_numpy(
            test_view["time_step"]
        ).reshape(-1)[test_period_mask],
        probability_by_seed=np.stack(
            test_probabilities_by_seed,
            axis=0,
        )[:, test_period_mask],
        ensemble_probability=ensemble_probabilities[test_period_mask],
        seeds=np.asarray(seeds, dtype=np.int64),
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 70)
    print("FIVE-SEED EXPLORATORY TEST SUMMARY")
    print("=" * 70)

    ap = summary["test_evaluation"][
        "f1_threshold_metrics_mean_and_std"
    ]["average_precision"]

    print(
        "Test AP mean ± std: "
        f"{ap['mean']:.5f} ± {ap['standard_deviation']:.5f}"
    )

    print(f"Test-period transactions: {int(test_period_mask.sum()):,}")
    print(f"Labelled test targets: {int(test_target_mask.sum()):,}")
    print(f"Capacity fraction: {alert_fraction:.2%}")
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved predictions: {prediction_path}")
    print("WARNING: Test metrics are exploratory, not an untouched holdout.")

    return summary


def _target_capacity_mask(
    period_mask: np.ndarray,
    target_mask: np.ndarray,
    period_local_alert_mask: np.ndarray,
) -> np.ndarray:
    """Map period-local capacity alerts to labelled period targets."""
    period_indices = np.flatnonzero(period_mask)
    target_indices = np.flatnonzero(target_mask)

    if len(period_local_alert_mask) != len(period_indices):
        raise ValueError("Capacity alert mask does not match period size.")

    alert_by_global_index = np.zeros(len(period_mask), dtype=bool)
    alert_by_global_index[period_indices] = period_local_alert_mask

    return alert_by_global_index[target_indices]


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
