"""Leakage-aware training pipeline for the undirected Elliptic GCN baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from elliptic_fraud.models.gcn import EllipticGCN
from elliptic_fraud.training.losses import (
    compute_pos_weight,
    masked_weighted_bce_loss,
)
from elliptic_fraud.training.train_gnn import (
    get_validation_average_precision,
    load_training_data,
    resolve_device,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = PROJECT_ROOT / "params.yaml"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/gcn"
FINAL_SEEDS = [42, 43, 44, 45, 46]


def train_one_run(
    graph_views: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
    learning_rate: float,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one GCN and select its checkpoint using validation AP."""

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")

    set_seed(seed)

    model_config = config["gnn"]
    train_view = graph_views["train"]
    validation_view = graph_views["validation"]
    device = train_view["x"].device

    train_mask = train_view["target_mask"]
    validation_mask = validation_view["target_mask"]

    if not train_mask.any().item():
        raise ValueError("No labelled training targets were found.")

    if not validation_mask.any().item():
        raise ValueError("No labelled validation targets were found.")

    # Class weighting uses training labels only.
    pos_weight = compute_pos_weight(
        train_view["y"],
        train_mask,
    ).to(device)

    model = EllipticGCN(
        in_channels=train_view["x"].size(1),
        hidden_channels=int(model_config["hidden_channels"]),
        num_layers=2,
        dropout=float(model_config["dropout"]),
        use_degree_features=bool(
            model_config["use_degree_features"]
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(model_config["weight_decay"]),
    )

    max_epochs = int(model_config["max_epochs"])
    patience = int(model_config["early_stopping_patience"])

    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")

    best_ap = -1.0
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
            train_mask,
            pos_weight,
        )

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(
                validation_view["x"],
                validation_view["edge_index"],
            )

            validation_ap = get_validation_average_precision(
                validation_logits,
                validation_view["y"],
                validation_mask,
            )

        current_loss = float(train_loss.detach().cpu().item())

        history.append({
            "epoch": epoch,
            "train_loss": current_loss,
            "validation_average_precision": validation_ap,
        })

        if validation_ap > best_ap:
            best_ap = validation_ap
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
                f"GCN | Seed {seed} | LR {learning_rate:g} | "
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
        raise RuntimeError("GCN training did not produce a checkpoint.")

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / (
        f"gcn_seed_{seed}_lr_{learning_rate:g}.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "seed": seed,
            "learning_rate": float(learning_rate),
            "best_epoch": best_epoch,
            "best_validation_average_precision": best_ap,
            "model_config": {
                "in_channels": int(train_view["x"].size(1)),
                "hidden_channels": int(model_config["hidden_channels"]),
                "num_layers": 2,
                "dropout": float(model_config["dropout"]),
                "use_degree_features": bool(
                    model_config["use_degree_features"]
                ),
                "graph_direction": "undirected",
            },
        },
        checkpoint_path,
    )

    history_path = output_dir / (
        f"history_seed_{seed}_lr_{learning_rate:g}.json"
    )

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    return {
        "seed": seed,
        "learning_rate": float(learning_rate),
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_ap,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }


def run_learning_rate_pilot(
    config: dict[str, Any],
    graph_views: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Select a learning rate using seed 42 and validation AP."""

    candidates = [
        float(value)
        for value in config["gnn"]["learning_rate_candidates"]
    ]

    if not candidates or any(
        not np.isfinite(value) or value <= 0
        for value in candidates
    ):
        raise ValueError("Learning-rate candidates must be positive.")

    seed = int(config["gnn"]["random_state"])
    results = []

    for learning_rate in candidates:
        print("\n" + "=" * 65)
        print(f"GCN LEARNING-RATE PILOT: {learning_rate:g}")
        print("=" * 65)

        results.append(
            train_one_run(
                graph_views=graph_views,
                config=config,
                learning_rate=learning_rate,
                seed=seed,
                output_dir=OUTPUT_DIR / "learning_rate_pilot",
            )
        )

    results.sort(
        key=lambda item: (
            -item["best_validation_average_precision"],
            item["learning_rate"],
        )
    )

    summary = {
        "status": "completed",
        "experiment": "undirected_gcn_learning_rate_pilot",
        "seed": seed,
        "selection_metric": "validation_average_precision",
        "tie_break_rule": "lower_learning_rate",
        "selected_learning_rate": results[0]["learning_rate"],
        "test_set_used": False,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "learning_rate_pilot_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nGCN LEARNING-RATE PILOT SUMMARY")
    for item in results:
        print(
            f"LR {item['learning_rate']:g} | "
            f"Best epoch {item['best_epoch']} | "
            f"Validation AP "
            f"{item['best_validation_average_precision']:.5f}"
        )

    print(
        "\nSelected GCN learning rate: "
        f"{summary['selected_learning_rate']:g}"
    )
    print(f"Summary saved to: {summary_path}")

    return summary


def run_final_seed_training(
    config: dict[str, Any],
    graph_views: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Train five final GCN models using the selected learning rate."""

    pilot_path = OUTPUT_DIR / "learning_rate_pilot_summary.json"

    if not pilot_path.exists():
        raise FileNotFoundError(
            "Run the GCN learning-rate pilot first: "
            f"{pilot_path}"
        )

    with pilot_path.open("r", encoding="utf-8") as file:
        pilot = json.load(file)

    if pilot.get("status") != "completed":
        raise ValueError("GCN learning-rate pilot is not completed.")

    if pilot.get("test_set_used") is not False:
        raise ValueError("GCN pilot must use validation data only.")

    learning_rate = float(pilot["selected_learning_rate"])

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Selected GCN learning rate is invalid.")

    output_dir = OUTPUT_DIR / "final_five_seed"
    results = []

    print("\n" + "=" * 70)
    print("ELLIPTIC SENTINEL — FINAL FIVE-SEED GCN TRAINING")
    print("=" * 70)
    print(f"Selected learning rate: {learning_rate:g}")
    print(f"Seeds: {FINAL_SEEDS}")
    print("Test graph view: not constructed")

    for seed in FINAL_SEEDS:
        print("\n" + "-" * 70)
        print(f"FINAL GCN MODEL — SEED {seed}")
        print("-" * 70)

        results.append(
            train_one_run(
                graph_views=graph_views,
                config=config,
                learning_rate=learning_rate,
                seed=seed,
                output_dir=output_dir,
            )
        )

    scores = np.asarray(
        [
            item["best_validation_average_precision"]
            for item in results
        ],
        dtype=np.float64,
    )

    summary = {
        "status": "completed",
        "experiment": "final_five_seed_undirected_gcn",
        "graph_direction": "undirected",
        "seeds": FINAL_SEEDS,
        "learning_rate": learning_rate,
        "learning_rate_source": str(pilot_path),
        "selection_metric": "validation_average_precision",
        "validation_average_precision": {
            "per_seed": {
                str(item["seed"]): (
                    item["best_validation_average_precision"]
                )
                for item in results
            },
            "mean": float(scores.mean()),
            "standard_deviation": float(scores.std(ddof=1)),
            "minimum": float(scores.min()),
            "maximum": float(scores.max()),
        },
        "test_graph_view_constructed": False,
        "test_set_used_for_training_or_selection": False,
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "five_seed_training_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nGCN FIVE-SEED TRAINING SUMMARY")
    for item in results:
        print(
            f"Seed {item['seed']} | "
            f"Best epoch {item['best_epoch']} | "
            f"Validation AP "
            f"{item['best_validation_average_precision']:.5f}"
        )

    print(
        "\nValidation AP mean ± std: "
        f"{scores.mean():.5f} ± {scores.std(ddof=1):.5f}"
    )
    print(f"Summary saved to: {summary_path}")

    return summary


def main() -> None:
    """Run the GCN learning-rate pilot and final five-seed training."""

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = resolve_device(
        str(config["gnn"].get("device", "cpu"))
    )

    print("=" * 70)
    print("ELLIPTIC SENTINEL — UNDIRECTED GCN TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Maximum epochs: {config['gnn']['max_epochs']}")
    print(
        "Early-stopping patience: "
        f"{config['gnn']['early_stopping_patience']}"
    )
    print(f"Seeds: {FINAL_SEEDS}")
    print("Feature source: scaled_features.pt")
    print("Graph direction: undirected (reverse edges added)")
    print("Test graph view: not constructed")

    # Reuse the leakage-aware loader: only train and validation views.
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

    # GCN gets its own pilot and its own selected learning rate.
    run_learning_rate_pilot(config, graph_views)
    run_final_seed_training(config, graph_views)


if __name__ == "__main__":
    main()
