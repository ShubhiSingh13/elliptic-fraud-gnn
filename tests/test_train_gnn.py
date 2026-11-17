
"""Tests for the leakage-aware GNN training engine."""

import json

import pytest
import torch

from elliptic_fraud.training import train_gnn


def make_synthetic_graph():
    """Create a tiny chronological graph for integration tests."""
    num_nodes = 6

    return {
        "x": torch.randn(num_nodes, 165),
        "edge_index": torch.tensor(
            [[0, 2, 4], [1, 3, 5]],
            dtype=torch.long,
        ),
        "y": torch.tensor([0, 1, 0, 1, 0, 1]),
        "time_step": torch.tensor([1, 1, 30, 30, 35, 35]),
        "split": torch.tensor([0, 0, 1, 1, 2, 2]),
        "train_mask": torch.tensor(
            [True, True, False, False, False, False]
        ),
        "validation_mask": torch.tensor(
            [True, True, True, True, False, False]
        ),
        "test_mask": torch.tensor(
            [True, True, True, True, True, True]
        ),
        "train_edge_index": torch.tensor(
            [[0], [1]], dtype=torch.long
        ),
        "validation_edge_index": torch.tensor(
            [[0, 2], [1, 3]], dtype=torch.long
        ),
        "test_edge_index": torch.tensor(
            [[0, 2, 4], [1, 3, 5]],
            dtype=torch.long,
        ),
    }


def make_synthetic_views():
    graph = make_synthetic_graph()

    return {
        split: train_gnn.build_asof_graph_view(graph, split)
        for split in ("train", "validation", "test")
    }


def make_small_config():
    return {
        "training": {
            "device": "cpu",
        },
        "gnn": {
            "hidden_channels": 8,
            "dropout": 0.1,
            "use_degree_features": True,
            "weight_decay": 0.0001,
            "max_epochs": 3,
            "early_stopping_patience": 2,
            "random_state": 42,
            "learning_rate_candidates": [0.001, 0.005, 0.01],
        },
    }


def test_load_experiment_data_uses_scaled_features(tmp_path, monkeypatch):
    graph = make_synthetic_graph()
    scaled = torch.randn(6, 165)

    graph_path = tmp_path / "graph.pt"
    scaled_path = tmp_path / "scaled.pt"

    torch.save(graph, graph_path)
    torch.save(scaled, scaled_path)

    monkeypatch.setattr(train_gnn, "GRAPH_FILE", graph_path)
    monkeypatch.setattr(
        train_gnn, "SCALED_FEATURES_FILE", scaled_path
    )

    loaded_graph, views = train_gnn.load_experiment_data(
        torch.device("cpu")
    )

    assert torch.equal(loaded_graph["x"], scaled)
    assert torch.equal(views["train"]["x"], scaled)
    assert torch.equal(views["validation"]["x"], scaled)
    assert torch.equal(views["test"]["x"], scaled)

    # The original graph file must remain unchanged.
    original = torch.load(graph_path, weights_only=True)
    assert torch.equal(original["x"], graph["x"])
    assert not torch.equal(original["x"], scaled)


def test_load_experiment_data_rejects_wrong_scaled_shape(
    tmp_path, monkeypatch
):
    graph = make_synthetic_graph()
    scaled = torch.randn(6, 93)

    graph_path = tmp_path / "graph.pt"
    scaled_path = tmp_path / "scaled.pt"

    torch.save(graph, graph_path)
    torch.save(scaled, scaled_path)

    monkeypatch.setattr(train_gnn, "GRAPH_FILE", graph_path)
    monkeypatch.setattr(
        train_gnn, "SCALED_FEATURES_FILE", scaled_path
    )

    with pytest.raises(ValueError, match="dimensions"):
        train_gnn.load_experiment_data(torch.device("cpu"))


def test_validation_average_precision_uses_target_mask():
    logits = torch.tensor([0.0, 2.0, 3.0, -2.0])
    labels = torch.tensor([0, 1, 999, 999])
    target_mask = torch.tensor([True, True, False, False])

    result = train_gnn.get_validation_average_precision(
        logits, labels, target_mask
    )

    assert result == pytest.approx(1.0)


def test_validation_average_precision_rejects_one_class():
    logits = torch.tensor([0.0, 1.0])
    labels = torch.tensor([0, 0])
    target_mask = torch.tensor([True, True])

    with pytest.raises(ValueError, match="both classes"):
        train_gnn.get_validation_average_precision(
            logits, labels, target_mask
        )


def test_train_one_run_saves_best_checkpoint_and_history(tmp_path):
    views = make_synthetic_views()
    config = make_small_config()

    result = train_gnn.train_one_run(
        graph_views=views,
        config=config,
        learning_rate=0.001,
        seed=42,
        output_dir=tmp_path,
    )

    checkpoint_path = tmp_path / (
        "gnn_seed_42_lr_0.001.pt"
    )
    history_path = tmp_path / (
        "history_seed_42_lr_0.001.json"
    )

    assert checkpoint_path.exists()
    assert history_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["seed"] == 42
    assert checkpoint["best_epoch"] == result["best_epoch"]
    assert (
        checkpoint["best_validation_average_precision"]
        == result["best_validation_average_precision"]
    )

    history = json.loads(history_path.read_text())

    assert len(history) <= config["gnn"]["max_epochs"]
    assert len(history) >= 1
    assert all(
        "validation_average_precision" in row
        for row in history
    )


def test_learning_rate_pilot_uses_validation_ap_and_lower_lr_tie(
    tmp_path, monkeypatch
):
    calls = []

    validation_scores = {
        0.001: 0.80,
        0.005: 0.80,
        0.01: 0.75,
    }

    def fake_train_one_run(
        graph_views,
        config,
        learning_rate,
        seed,
        output_dir,
    ):
        calls.append(learning_rate)

        return {
            "seed": seed,
            "learning_rate": learning_rate,
            "best_epoch": 2,
            "best_validation_average_precision":
                validation_scores[learning_rate],
            "checkpoint": "synthetic.pt",
            "history": "synthetic.json",
        }

    monkeypatch.setattr(
        train_gnn, "train_one_run", fake_train_one_run
    )
    monkeypatch.setattr(train_gnn, "OUTPUT_DIR", tmp_path)

    config = make_small_config()
    views = make_synthetic_views()

    summary = train_gnn.run_learning_rate_pilot(
        config, views
    )

    assert calls == [0.001, 0.005, 0.01]
    assert summary["selected_learning_rate"] == 0.001
    assert summary["selection_metric"] == (
        "validation_average_precision"
    )
    assert summary["test_set_used"] is False

    saved_summary = json.loads(
        (tmp_path / "learning_rate_pilot_summary.json").read_text()
    )

    assert saved_summary["selected_learning_rate"] == 0.001
