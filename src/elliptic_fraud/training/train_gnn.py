"""Leakage-aware Directed GraphSAGE training pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from elliptic_fraud.models.gnn import DirectedGraphSAGE
from elliptic_fraud.training.graph_views import build_asof_graph_view
from elliptic_fraud.training.losses import (
    compute_pos_weight,
    masked_weighted_bce_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPH_FILE = PROJECT_ROOT / "data/processed/elliptic/transaction_graph.pt"
SCALED_FEATURES_FILE = (
    PROJECT_ROOT / "data/processed/elliptic/scaled_features.pt"
)
CONFIG_FILE = PROJECT_ROOT / "params.yaml"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/gnn"


# ============================================================
# 1. REPRODUCIBILITY AND DEVICE
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested device."""
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    raise ValueError(
        f"Unsupported device {requested!r}. Use cpu, mps, or cuda."
    )


# ============================================================
# 2. GRAPH AND FEATURE LOADING
# ============================================================

def load_graph_data(device: torch.device) -> dict[str, Any]:
    """
    Load the graph and substitute the scaled feature matrix.

    No graph views are constructed here.
    """

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_FILE}")

    if not SCALED_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Scaled feature file not found: {SCALED_FEATURES_FILE}"
        )

    graph = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(graph, dict):
        raise TypeError("The saved transaction graph must be a dictionary.")

    if "x" not in graph:
        raise KeyError("The transaction graph does not contain node features.")

    scaled_features = torch.load(
        SCALED_FEATURES_FILE,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(scaled_features, torch.Tensor):
        raise TypeError("scaled_features.pt must contain a tensor.")

    if scaled_features.shape != graph["x"].shape:
        raise ValueError(
            "Scaled feature dimensions do not match the graph feature "
            f"dimensions: {tuple(scaled_features.shape)} vs "
            f"{tuple(graph['x'].shape)}."
        )

    if not torch.isfinite(scaled_features).all().item():
        raise ValueError("Scaled features contain NaN or infinite values.")

    # Replace features in memory only.
    graph = dict(graph)
    graph["x"] = scaled_features.to(dtype=torch.float32)

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }


def _move_views_to_device(
    views: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Move every tensor in every graph view to the selected device."""

    return {
        split: {
            key: value.to(device)
            for key, value in view.items()
        }
        for split, view in views.items()
    }


def load_experiment_data(
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    """
    Load the graph and construct all three views.

    Retained for backward compatibility and evaluation utilities.
    """

    graph = load_graph_data(device)

    views = {
        split: build_asof_graph_view(graph, split)
        for split in ("train", "validation", "test")
    }

    views = _move_views_to_device(views, device)

    return graph, views


def load_training_data(
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    """
    Load the graph and construct only train and validation views.

    The test graph view is deliberately not constructed during training.
    """

    graph = load_graph_data(device)

    views = {
        split: build_asof_graph_view(graph, split)
        for split in ("train", "validation")
    }

    views = _move_views_to_device(views, device)

    return graph, views


# ============================================================
# 3. VALIDATION METRIC
# ============================================================

def get_validation_average_precision(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    """Calculate AP on labelled validation targets only."""

    if not target_mask.any().item():
        raise ValueError("Validation target mask is empty.")

    selected_labels = labels[target_mask]
    selected_logits = logits[target_mask]

    if not torch.all(
        (selected_labels == 0) | (selected_labels == 1)
    ).item():
        raise ValueError("Validation targets must have binary labels.")

    if torch.unique(selected_labels).numel() != 2:
        raise ValueError(
            "Validation targets must contain both classes for AP."
        )

    probabilities = (
        torch.sigmoid(selected_logits).detach().cpu().numpy()
    )

    true_labels = selected_labels.detach().cpu().numpy()

    return float(
        average_precision_score(true_labels, probabilities)
    )


# ============================================================
# 4. TRAIN ONE MODEL
# ============================================================

def train_one_run(
    graph_views: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
    learning_rate: float,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Train one Directed GraphSAGE model.

    The best checkpoint is selected using validation AP.
    """

    set_seed(seed)

    gnn_config = config["gnn"]

    device = graph_views["train"]["x"].device

    train_view = graph_views["train"]
    validation_view = graph_views["validation"]

    train_target_mask = train_view["target_mask"]
    validation_target_mask = validation_view["target_mask"]

    if not train_target_mask.any().item():
        raise ValueError("No labelled training targets were found.")

    if not validation_target_mask.any().item():
        raise ValueError("No labelled validation targets were found.")

    # Positive-class weight is calculated using training labels only.
    pos_weight = compute_pos_weight(
        train_view["y"],
        train_target_mask,
    ).to(device)

    model = DirectedGraphSAGE(
        in_channels=train_view["x"].size(1),
        hidden_channels=int(gnn_config["hidden_channels"]),
        dropout=float(gnn_config["dropout"]),
        use_degree_features=bool(
            gnn_config["use_degree_features"]
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(gnn_config["weight_decay"]),
    )

    max_epochs = int(gnn_config["max_epochs"])
    patience = int(gnn_config["early_stopping_patience"])

    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")

    if patience < 1:
        raise ValueError(
            "early_stopping_patience must be at least 1."
        )

    best_validation_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):

        # -------------------------
        # Training
        # -------------------------

        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_logits = model(
            train_view["x"],
            train_view["edge_index"],
        )

        train_loss = masked_weighted_bce_loss(
            train_logits,
            train_view["y"],
            train_target_mask,
            pos_weight,
        )

        train_loss.backward()
        optimizer.step()

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        with torch.no_grad():

            validation_logits = model(
                validation_view["x"],
                validation_view["edge_index"],
            )

            validation_ap = get_validation_average_precision(
                validation_logits,
                validation_view["y"],
                validation_target_mask,
            )

        current_loss = float(train_loss.detach().cpu().item())

        history.append(
            {
                "epoch": epoch,
                "train_loss": current_loss,
                "validation_average_precision": validation_ap,
            }
        )

        # -------------------------
        # Best checkpoint tracking
        # -------------------------

        if validation_ap > best_validation_ap:

            best_validation_ap = validation_ap
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:
            epochs_without_improvement += 1

        # -------------------------
        # Training logs
        # -------------------------

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Seed {seed} | LR {learning_rate:g} | "
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"Loss {current_loss:.5f} | "
                f"Validation AP {validation_ap:.5f}"
            )

        # -------------------------
        # Early stopping
        # -------------------------

        if epochs_without_improvement >= patience:

            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )

            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint."
        )

    # Restore best validation checkpoint.
    model.load_state_dict(best_state)
    model.eval()

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / (
        f"gnn_seed_{seed}_lr_{learning_rate:g}.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "seed": seed,
            "learning_rate": float(learning_rate),
            "best_epoch": best_epoch,
            "best_validation_average_precision": best_validation_ap,
            "model_config": {
                "in_channels": int(train_view["x"].size(1)),
                "hidden_channels": int(
                    gnn_config["hidden_channels"]
                ),
                "dropout": float(gnn_config["dropout"]),
                "use_degree_features": bool(
                    gnn_config["use_degree_features"]
                ),
            },
        },
        checkpoint_path,
    )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_path = output_dir / (
        f"history_seed_{seed}_lr_{learning_rate:g}.json"
    )

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    return {
        "seed": seed,
        "learning_rate": float(learning_rate),
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_validation_ap,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }


# ============================================================
# 5. LEARNING-RATE PILOT
# ============================================================

def run_learning_rate_pilot(
    config: dict[str, Any],
    graph_views: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Compare configured learning rates using seed 42 and validation AP."""

    candidates = [
        float(value)
        for value in config["gnn"]["learning_rate_candidates"]
    ]

    if not candidates:
        raise ValueError(
            "No learning-rate candidates were configured."
        )

    if any(value <= 0 for value in candidates):
        raise ValueError(
            "All learning-rate candidates must be positive."
        )

    seed = int(config["gnn"]["random_state"])

    results = []

    for learning_rate in candidates:

        print("\n" + "=" * 65)
        print(f"LEARNING-RATE PILOT: {learning_rate:g}")
        print("=" * 65)

        result = train_one_run(
            graph_views=graph_views,
            config=config,
            learning_rate=learning_rate,
            seed=seed,
            output_dir=OUTPUT_DIR / "learning_rate_pilot",
        )

        results.append(result)

    # Highest validation AP wins.
    # Lower learning rate breaks an exact tie.

    results.sort(
        key=lambda item: (
            -item["best_validation_average_precision"],
            item["learning_rate"],
        )
    )

    selected_learning_rate = results[0]["learning_rate"]

    summary = {
        "status": "completed",
        "seed": seed,
        "selection_metric": "validation_average_precision",
        "tie_break_rule": "lower_learning_rate",
        "selected_learning_rate": selected_learning_rate,
        "test_set_used": False,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "learning_rate_pilot_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 65)
    print("LEARNING-RATE PILOT SUMMARY")
    print("=" * 65)

    for result in results:

        print(
            f"LR {result['learning_rate']:g} | "
            f"Best epoch {result['best_epoch']} | "
            f"Validation AP "
            f"{result['best_validation_average_precision']:.5f}"
        )

    print(
        f"\nSelected learning rate: {selected_learning_rate:g}"
    )

    print(f"Summary saved to: {summary_path}")

    return summary


# ============================================================
# 6. FINAL FIVE-SEED TRAINING
# ============================================================

def run_final_seed_training(
    config: dict[str, Any],
    graph_views: dict[str, dict[str, torch.Tensor]],
    pilot_summary_path: Path,
) -> dict[str, Any]:
    """
    Train five final models using the learning rate selected by the pilot.

    Only training and validation data are used.
    No test graph view is constructed.
    """

    if not pilot_summary_path.exists():

        raise FileNotFoundError(
            "Completed learning-rate pilot summary not found: "
            f"{pilot_summary_path}"
        )

    with pilot_summary_path.open("r", encoding="utf-8") as file:
        pilot_summary = json.load(file)

    if pilot_summary.get("status") != "completed":

        raise ValueError(
            "The learning-rate pilot is not marked completed."
        )

    if pilot_summary.get("test_set_used") is not False:

        raise ValueError(
            "Pilot summary does not confirm validation-only selection."
        )

    learning_rate = float(
        pilot_summary["selected_learning_rate"]
    )

    if not np.isfinite(learning_rate) or learning_rate <= 0:

        raise ValueError(
            "Selected learning rate must be finite and positive."
        )

    # Five final random seeds.
    seeds = [42, 43, 44, 45, 46]

    output_dir = OUTPUT_DIR / "final_five_seed"

    results: list[dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("FINAL FIVE-SEED DIRECTED GRAPHSAGE TRAINING")
    print("=" * 70)

    print(f"Selected learning rate: {learning_rate:g}")
    print(f"Seeds: {seeds}")
    print("Test graph view: not constructed")

    # Train five independent models.
    for seed in seeds:

        print("\n" + "-" * 70)
        print(f"FINAL MODEL — SEED {seed}")
        print("-" * 70)

        result = train_one_run(
            graph_views=graph_views,
            config=config,
            learning_rate=learning_rate,
            seed=seed,
            output_dir=output_dir,
        )

        results.append(result)

    # ========================================================
    # VALIDATION SCORE SUMMARY
    # ========================================================

    scores = np.asarray(
        [
            item["best_validation_average_precision"]
            for item in results
        ],
        dtype=np.float64,
    )

    summary = {
        "status": "completed",
        "experiment": "final_five_seed_directed_graphsage",
        "seeds": seeds,
        "learning_rate": learning_rate,
        "learning_rate_source": str(pilot_summary_path),
        "selection_metric": "validation_average_precision",

        "validation_average_precision": {
            "per_seed": {
                str(item["seed"]): (
                    item["best_validation_average_precision"]
                )
                for item in results
            },

            "mean": float(scores.mean()),

            "standard_deviation": float(
                scores.std(ddof=1)
            ),

            "minimum": float(scores.min()),

            "maximum": float(scores.max()),
        },

        "test_graph_view_constructed": False,

        "test_set_used_for_training_or_selection": False,

        "results": results,
    }

    # ========================================================
    # SAVE FINAL SUMMARY
    # ========================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        output_dir / "five_seed_training_summary.json"
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FIVE-SEED TRAINING SUMMARY")
    print("=" * 70)

    for item in results:

        print(
            f"Seed {item['seed']} | "
            f"Best epoch {item['best_epoch']} | "
            f"Validation AP "
            f"{item['best_validation_average_precision']:.5f}"
        )

    print(
        "\nValidation AP mean ± std: "
        f"{scores.mean():.5f} ± "
        f"{scores.std(ddof=1):.5f}"
    )

    print(f"Summary saved to: {summary_path}")

    return summary


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main() -> None:
    """Run final five-seed training using the completed LR pilot."""

    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    requested_device = str(
        config["gnn"].get("device", "cpu")
    )

    device = resolve_device(requested_device)

    print("=" * 70)
    print("ELLIPTIC SENTINEL — FINAL DIRECTED GRAPHSAGE TRAINING")
    print("=" * 70)

    print(f"Device: {device}")

    print(
        f"Maximum epochs: {config['gnn']['max_epochs']}"
    )

    print(
        "Early-stopping patience: "
        f"{config['gnn']['early_stopping_patience']}"
    )

    print("Seeds: [42, 43, 44, 45, 46]")

    print("Feature source: scaled_features.pt")

    print("Test graph view: not constructed")

    # Only training and validation views are constructed.
    _, graph_views = load_training_data(device)

    print("\nGRAPH VIEW SUMMARY")

    for split, view in graph_views.items():

        print(
            f"{split.title():10s} | "
            f"Context nodes: "
            f"{int(view['context_mask'].sum()):,} | "
            f"Supervised targets: "
            f"{int(view['target_mask'].sum()):,} | "
            f"Edges: {view['edge_index'].size(1):,}"
        )

    # Use the existing pilot result.
    pilot_summary_path = (
        OUTPUT_DIR / "learning_rate_pilot_summary.json"
    )

    run_final_seed_training(
        config=config,
        graph_views=graph_views,
        pilot_summary_path=pilot_summary_path,
    )


if __name__ == "__main__":
    main()