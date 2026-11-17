
"""Leakage-aware Directed GraphSAGE training and learning-rate pilot."""

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
SCALED_FEATURES_FILE = PROJECT_ROOT / "data/processed/elliptic/scaled_features.pt"
CONFIG_FILE = PROJECT_ROOT / "params.yaml"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/gnn"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested device, with a clear fallback policy."""
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


def load_experiment_data(
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    """Load the graph and substitute the verified scaled feature matrix."""
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

    # Do not modify the original graph or its saved file.
    graph = dict(graph)
    graph["x"] = scaled_features.to(dtype=torch.float32)

    views = {
        split: build_asof_graph_view(graph, split)
        for split in ("train", "validation", "test")
    }

    # Move tensors to the selected device.
    graph = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }

    views = {
        split: {
            key: value.to(device)
            for key, value in view.items()
        }
        for split, view in views.items()
    }

    return graph, views


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

    probabilities = torch.sigmoid(selected_logits).detach().cpu().numpy()
    true_labels = selected_labels.detach().cpu().numpy()

    return float(average_precision_score(true_labels, probabilities))


def train_one_run(
    graph_views: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
    learning_rate: float,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one model and restore its best validation-AP checkpoint."""
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

    # The positive-class weight is derived from training labels only.
    pos_weight = compute_pos_weight(
        train_view["y"],
        train_target_mask,
    ).to(device)

    model = DirectedGraphSAGE(
        in_channels=train_view["x"].size(1),
        hidden_channels=int(gnn_config["hidden_channels"]),
        dropout=float(gnn_config["dropout"]),
        use_degree_features=bool(gnn_config["use_degree_features"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(gnn_config["weight_decay"]),
    )

    max_epochs = int(gnn_config["max_epochs"])
    patience = int(gnn_config["early_stopping_patience"])

    best_validation_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
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

        # Validation is used for model selection and early stopping.
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

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Seed {seed} | LR {learning_rate:g} | "
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"Loss {current_loss:.5f} | "
                f"Validation AP {validation_ap:.5f}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    # Restore the checkpoint selected using validation AP.
    model.load_state_dict(best_state)
    model.eval()

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
                "hidden_channels": int(gnn_config["hidden_channels"]),
                "dropout": float(gnn_config["dropout"]),
                "use_degree_features": bool(
                    gnn_config["use_degree_features"]
                ),
            },
        },
        checkpoint_path,
    )

    history_path = output_dir / (
        f"history_seed_{seed}_lr_{learning_rate:g}.json"
    )

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # Return full paths so this also works with pytest temporary directories.
    return {
        "seed": seed,
        "learning_rate": float(learning_rate),
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_validation_ap,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }


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
        raise ValueError("No learning-rate candidates were configured.")

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

    # Highest validation AP wins. Lower learning rate breaks an exact tie.
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

    print(f"\nSelected learning rate: {selected_learning_rate:g}")
    print(f"Summary saved to: {summary_path}")

    return summary


def main() -> None:
    """Load configuration and run the validation-only LR pilot."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    requested_device = str(config["gnn"].get("device", "cpu"))
    device = resolve_device(requested_device)

    print("=" * 65)
    print("ELLIPTIC SENTINEL — DIRECTED GRAPHSAGE TRAINING")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Maximum epochs: {config['gnn']['max_epochs']}")
    print(
        f"Early-stopping patience: "
        f"{config['gnn']['early_stopping_patience']}"
    )
    print(
        f"Learning-rate candidates: "
        f"{config['gnn']['learning_rate_candidates']}"
    )
    print("Feature source: scaled_features.pt")
    print("Test evaluation: disabled during learning-rate selection")

    _, graph_views = load_experiment_data(device)

    print("\nGRAPH VIEW SUMMARY")
    for split, view in graph_views.items():
        print(
            f"{split.title():10s} | "
            f"Context nodes: {int(view['context_mask'].sum()):,} | "
            f"Supervised targets: {int(view['target_mask'].sum()):,} | "
            f"Edges: {view['edge_index'].size(1):,}"
        )

    run_learning_rate_pilot(config, graph_views)


if __name__ == "__main__":
    main()
