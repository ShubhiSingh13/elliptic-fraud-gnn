"""Five-seed validation calibration and exploratory test evaluation for GCN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from elliptic_fraud.evaluation.thresholds import (
    select_f1_threshold,
    select_top_fraction,
)
from elliptic_fraud.models.gcn import EllipticGCN
from elliptic_fraud.training.graph_views import build_asof_graph_view
from elliptic_fraud.training.train_gnn import load_graph_data, resolve_device
from elliptic_fraud.evaluation.evaluate_gnn import (
    load_config,
    load_final_training_summary,
    tensor_to_numpy,
    move_graph_view_to_device,
    build_period_mask,
    validate_target_period,
    validate_binary_labels,
    validate_transaction_ids,
    calculate_threshold_alert_metrics,
    calculate_ranked_alert_metrics,
    calculate_alert_rate_from_mask,
    aggregate_metric_results,
    _target_capacity_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = PROJECT_ROOT / "params.yaml"
TRAINING_SUMMARY = (
    PROJECT_ROOT / "artifacts/gcn/final_five_seed/"
    "five_seed_training_summary.json"
)
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts/gcn/final_five_seed"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/gcn/evaluation"


def load_gcn_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[EllipticGCN, dict[str, Any]]:
    """Load and validate one GCN checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    required = {
        "model_state_dict",
        "model_config",
        "seed",
        "learning_rate",
        "best_epoch",
        "best_validation_average_precision",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")

    config = checkpoint["model_config"]
    if config.get("graph_direction") != "undirected":
        raise ValueError("Expected an undirected GCN checkpoint.")

    model = EllipticGCN(
        in_channels=int(config["in_channels"]),
        hidden_channels=int(config["hidden_channels"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        use_degree_features=bool(config["use_degree_features"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_probabilities(
    model: EllipticGCN,
    graph_view: dict[str, torch.Tensor],
) -> np.ndarray:
    """Generate one probability per node."""
    model.eval()
    with torch.no_grad():
        logits = model(graph_view["x"], graph_view["edge_index"])
        probabilities = torch.sigmoid(logits)

    values = probabilities.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Predictions must be in [0, 1].")
    return values.astype(np.float64)


def run_evaluation() -> dict[str, Any]:
    """Calibrate on validation, then evaluate the exploratory test period."""
    config = load_config(CONFIG_FILE)
    training_summary = load_final_training_summary(TRAINING_SUMMARY)

    device = resolve_device(str(config.get("gnn", {}).get("device", "cpu")))
    seeds = [int(seed) for seed in training_summary["seeds"]]
    learning_rate = float(training_summary["learning_rate"])

    validation_start = int(
        config["split"]["validation"]["start_time_step"]
    )
    validation_end = int(
        config["split"]["validation"]["end_time_step"]
    )
    test_start = int(config["split"]["test"]["start_time_step"])
    test_end = int(config["split"]["test"]["end_time_step"])
    alert_fraction = float(
        config.get("evaluation", {}).get("alert_fraction", 0.01)
    )
    if not 0 < alert_fraction < 1:
        raise ValueError("Alert fraction must be between 0 and 1.")

    graph = load_graph_data(device)
    transaction_ids = validate_transaction_ids(
        tensor_to_numpy(graph["tx_id"]),
        len(graph["y"]),
    )

    # Validation view only: no test view is constructed yet.
    validation_view = move_graph_view_to_device(
        build_asof_graph_view(graph, "validation"),
        device,
    )
    validation_period_mask = build_period_mask(
        validation_view, validation_start, validation_end
    )
    validation_target_mask = (
        tensor_to_numpy(validation_view["target_mask"])
        .astype(bool).reshape(-1)
    )
    validation_labels = tensor_to_numpy(
        validation_view["y"]
    ).reshape(-1)

    validate_target_period(
        validation_target_mask, validation_period_mask, "Validation"
    )
    validate_binary_labels(
        validation_labels, validation_target_mask, "Validation"
    )
    validation_ids = transaction_ids[validation_period_mask]

    validation_results = []
    validation_models = []

    print("\n" + "=" * 70)
    print("ELLIPTIC SENTINEL — GCN FIVE-SEED VALIDATION CALIBRATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print("Test graph view constructed: NO")

    # Calibrate every seed using labelled validation targets only.
    for seed in seeds:
        checkpoint_path = (
            CHECKPOINT_DIR / f"gcn_seed_{seed}_lr_{learning_rate:g}.pt"
        )
        model, checkpoint = load_gcn_checkpoint(checkpoint_path, device)

        if int(checkpoint["seed"]) != seed:
            raise ValueError(f"Checkpoint seed mismatch for seed {seed}.")
        if not np.isclose(
            float(checkpoint["learning_rate"]), learning_rate
        ):
            raise ValueError(
                f"Checkpoint learning-rate mismatch for seed {seed}."
            )

        probabilities = predict_probabilities(model, validation_view)
        if len(probabilities) != len(validation_period_mask):
            raise ValueError("Validation prediction length mismatch.")

        f1_result = select_f1_threshold(
            probabilities[validation_target_mask],
            validation_labels[validation_target_mask],
        )
        capacity_result = select_top_fraction(
            probabilities[validation_period_mask],
            validation_ids,
            fraction=alert_fraction,
        )

        validation_results.append({
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_validation_average_precision": float(
                checkpoint["best_validation_average_precision"]
            ),
            "f1_threshold": float(f1_result["threshold"]),
            "f1_calibration": f1_result,
            "capacity_calibration": {
                key: value for key, value in capacity_result.items()
                if key not in {"alert_mask", "selected_indices"}
            },
        })
        validation_models.append({
            "seed": seed,
            "model": model,
            "checkpoint": checkpoint,
            "f1_threshold": float(f1_result["threshold"]),
        })

        print(
            f"Seed {seed} | "
            f"Validation AP "
            f"{float(checkpoint['best_validation_average_precision']):.5f} | "
            f"F1 threshold {float(f1_result['threshold']):.6f} | "
            f"Capacity alerts {capacity_result['actual_alert_count']}/"
            f"{capacity_result['transaction_count']}"
        )

    frozen_thresholds = {
        str(item["seed"]): {
            "f1_threshold": item["f1_threshold"],
            "capacity_fraction": alert_fraction,
            "capacity_tie_break": "ascending_transaction_id",
        }
        for item in validation_results
    }

    print("\nValidation calibration complete.")
    print("All thresholds frozen. Constructing test graph view now.")

    # Test view is constructed only after all validation calibration.
    test_view = move_graph_view_to_device(
        build_asof_graph_view(graph, "test"),
        device,
    )
    test_period_mask = build_period_mask(
        test_view, test_start, test_end
    )
    test_target_mask = (
        tensor_to_numpy(test_view["target_mask"])
        .astype(bool).reshape(-1)
    )
    test_labels = tensor_to_numpy(test_view["y"]).reshape(-1)

    validate_target_period(test_target_mask, test_period_mask, "Test")
    validate_binary_labels(test_labels, test_target_mask, "Test")
    test_ids = transaction_ids[test_period_mask]

    per_seed_test_results = []
    test_probabilities_by_seed = []

    for item in validation_models:
        seed = int(item["seed"])
        model = item["model"]
        f1_threshold = float(item["f1_threshold"])

        probabilities = predict_probabilities(model, test_view)
        if len(probabilities) != len(test_period_mask):
            raise ValueError(f"Test prediction length mismatch for seed {seed}.")

        test_probabilities_by_seed.append(probabilities)

        target_scores = probabilities[test_target_mask]
        target_labels = test_labels[test_target_mask]

        f1_metrics = calculate_threshold_alert_metrics(
            target_labels, target_scores, f1_threshold
        )

        # Capacity selection ranks every new test-period transaction,
        # including unknown-label transactions. It does not use labels.
        capacity_result = select_top_fraction(
            probabilities[test_period_mask],
            test_ids,
            fraction=alert_fraction,
        )
        target_capacity_mask = _target_capacity_mask(
            test_period_mask,
            test_target_mask,
            capacity_result["alert_mask"],
        )
        capacity_metrics = calculate_ranked_alert_metrics(
            target_labels,
            target_scores,
            target_capacity_mask,
        )
        capacity_alert_rate = calculate_alert_rate_from_mask(
            capacity_result["alert_mask"]
        )

        per_seed_test_results.append({
            "seed": seed,
            "checkpoint": str(
                CHECKPOINT_DIR
                / f"gcn_seed_{seed}_lr_{learning_rate:g}.pt"
            ),
            "best_epoch": int(item["checkpoint"]["best_epoch"]),
            "f1_threshold": f1_threshold,
            "f1_threshold_metrics": f1_metrics,
            "f1_threshold_alert_rate": {
                "transaction_count": int(test_period_mask.sum()),
                "alert_count": int(
                    np.sum(probabilities[test_period_mask] >= f1_threshold)
                ),
                "alert_rate": float(
                    np.mean(probabilities[test_period_mask] >= f1_threshold)
                ),
            },
            "capacity_metrics": capacity_metrics,
            "capacity_alert_rate": capacity_alert_rate,
            "capacity_selection": {
                key: value for key, value in capacity_result.items()
                if key not in {"alert_mask", "selected_indices"}
            },
        })

        print(
            f"Seed {seed} | Test AP {f1_metrics['average_precision']:.5f} | "
            f"Capacity alerts {capacity_result['actual_alert_count']}/"
            f"{capacity_result['transaction_count']}"
        )
        model.cpu()

    f1_metrics_all = [
        item["f1_threshold_metrics"] for item in per_seed_test_results
    ]
    capacity_metrics_all = [
        item["capacity_metrics"] for item in per_seed_test_results
    ]
    f1_alert_rates = [
        item["f1_threshold_alert_rate"] for item in per_seed_test_results
    ]
    capacity_alert_rates = [
        item["capacity_alert_rate"] for item in per_seed_test_results
    ]

    ensemble_probabilities = np.mean(
        np.stack(test_probabilities_by_seed, axis=0), axis=0
    )
    if not np.isfinite(ensemble_probabilities).all():
        raise ValueError("Ensemble predictions contain invalid values.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "project": "Elliptic Sentinel",
        "evaluation_status": "completed",
        "test_status": "exploratory",
        "test_holdout_warning": (
            "Test-period results are exploratory because test data was "
            "inspected during development. They are not results from an "
            "untouched holdout."
        ),
        "device": str(device),
        "experiment": "final_five_seed_undirected_gcn",
        "graph_direction": "undirected",
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
                f1_metrics_all
            ),
            "capacity_metrics_mean_and_std": aggregate_metric_results(
                capacity_metrics_all
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
                "Mean probability across five seeds, saved for descriptive "
                "analysis only. No ensemble threshold was selected or applied."
            )
        },
        "methodology_notes": [
            "Each seed uses its own validation-calibrated F1 threshold.",
            "F1 thresholds use labelled validation targets only.",
            "Capacity ranking uses all new transactions in each period.",
            "Capacity ranking does not use labels.",
            "Capacity selection breaks score ties by ascending transaction ID.",
            "The GCN uses an undirected graph created by symmetrizing edges.",
            "The test graph view is constructed after validation calibration.",
            "Test metrics are exploratory, not untouched holdout results.",
            "Seed variability is not temporal generalisation uncertainty.",
            "GCN and directed GraphSAGE differ in graph directionality as well "
            "as aggregation architecture.",
        ],
    }

    summary_path = OUTPUT_DIR / "five_seed_evaluation_summary.json"
    prediction_path = OUTPUT_DIR / "five_seed_test_transaction_scores.npz"

    np.savez_compressed(
        prediction_path,
        tx_id=test_ids,
        time_step=tensor_to_numpy(test_view["time_step"])
        .reshape(-1)[test_period_mask],
        probability_by_seed=np.stack(
            test_probabilities_by_seed, axis=0
        )[:, test_period_mask],
        ensemble_probability=ensemble_probabilities[test_period_mask],
        seeds=np.asarray(seeds, dtype=np.int64),
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    ap = summary["test_evaluation"][
        "f1_threshold_metrics_mean_and_std"
    ]["average_precision"]

    print("\n" + "=" * 70)
    print("GCN FIVE-SEED EXPLORATORY TEST SUMMARY")
    print("=" * 70)
    print(
        f"Test AP mean ± std: "
        f"{ap['mean']:.5f} ± {ap['standard_deviation']:.5f}"
    )
    print(f"Test-period transactions: {int(test_period_mask.sum()):,}")
    print(f"Summary saved: {summary_path}")
    print(f"Predictions saved: {prediction_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
