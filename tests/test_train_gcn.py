"""Tests for the leakage-aware GCN training pipeline."""

import json

import pytest
import torch

from elliptic_fraud.training import train_gcn, train_gnn


def make_synthetic_graph():
    """Create a tiny chronological graph for GCN training tests."""
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
    """Build chronological train, validation and test graph views."""
    graph = make_synthetic_graph()

    return {
        split: train_gnn.build_asof_graph_view(graph, split)
        for split in ("train", "validation", "test")
    }


def make_small_config():
    """Return a small configuration suitable for fast tests."""
    return {
        "training": {"device": "cpu"},
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


def make_fake_training_result(learning_rate, seed, ap=0.8):
    """Return a valid synthetic result without training a model."""
    return {
        "seed": seed,
        "learning_rate": float(learning_rate),
        "best_epoch": 2,
        "best_validation_average_precision": float(ap),
        "checkpoint": "synthetic.pt",
        "history": "synthetic.json",
    }


def test_train_one_run_saves_best_checkpoint_and_history(tmp_path):
    views = make_synthetic_views()
    config = make_small_config()

    result = train_gcn.train_one_run(
        graph_views=views,
        config=config,
        learning_rate=0.001,
        seed=42,
        output_dir=tmp_path,
    )

    checkpoint_path = tmp_path / "gcn_seed_42_lr_0.001.pt"
    history_path = tmp_path / "history_seed_42_lr_0.001.json"

    assert checkpoint_path.exists()
    assert history_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["seed"] == 42
    assert checkpoint["learning_rate"] == 0.001
    assert checkpoint["best_epoch"] == result["best_epoch"]
    assert (
        checkpoint["best_validation_average_precision"]
        == result["best_validation_average_precision"]
    )
    assert checkpoint["model_config"]["graph_direction"] == "undirected"

    history = json.loads(history_path.read_text(encoding="utf-8"))

    assert 1 <= len(history) <= config["gnn"]["max_epochs"]
    assert all(
        "validation_average_precision" in row
        for row in history
    )
    assert all("train_loss" in row for row in history)


@pytest.mark.parametrize(
    "learning_rate",
    [0, -0.001, float("nan"), float("inf")],
)
def test_train_one_run_rejects_invalid_learning_rate(
    tmp_path,
    learning_rate,
):
    with pytest.raises(ValueError, match="finite and positive"):
        train_gcn.train_one_run(
            graph_views=make_synthetic_views(),
            config=make_small_config(),
            learning_rate=learning_rate,
            seed=42,
            output_dir=tmp_path,
        )


def test_learning_rate_pilot_uses_validation_ap_and_lower_lr_tie(
    tmp_path,
    monkeypatch,
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
        return make_fake_training_result(
            learning_rate,
            seed,
            validation_scores[learning_rate],
        )

    monkeypatch.setattr(
        train_gcn,
        "train_one_run",
        fake_train_one_run,
    )
    monkeypatch.setattr(train_gcn, "OUTPUT_DIR", tmp_path)

    config = make_small_config()
    views = make_synthetic_views()

    summary = train_gcn.run_learning_rate_pilot(config, views)

    assert calls == [0.001, 0.005, 0.01]
    assert summary["selected_learning_rate"] == 0.001
    assert summary["selection_metric"] == "validation_average_precision"
    assert summary["test_set_used"] is False

    saved_summary = json.loads(
        (tmp_path / "learning_rate_pilot_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert saved_summary["selected_learning_rate"] == 0.001
    assert saved_summary["test_set_used"] is False


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        [0],
        [-0.001],
        [float("nan")],
        [float("inf")],
    ],
)
def test_learning_rate_pilot_rejects_invalid_candidates(
    tmp_path,
    monkeypatch,
    candidates,
):
    monkeypatch.setattr(train_gcn, "OUTPUT_DIR", tmp_path)

    config = make_small_config()
    config["gnn"]["learning_rate_candidates"] = candidates

    with pytest.raises(ValueError, match="Learning-rate candidates"):
        train_gcn.run_learning_rate_pilot(
            config,
            make_synthetic_views(),
        )


def test_final_seed_training_runs_five_seeds_and_saves_summary(
    tmp_path,
    monkeypatch,
):
    pilot_summary = {
        "status": "completed",
        "selected_learning_rate": 0.005,
        "test_set_used": False,
        "results": [
            make_fake_training_result(0.005, 42, 0.80),
        ],
    }

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "learning_rate_pilot_summary.json").write_text(
        json.dumps(pilot_summary),
        encoding="utf-8",
    )

    calls = []

    def fake_train_one_run(
        graph_views,
        config,
        learning_rate,
        seed,
        output_dir,
    ):
        calls.append((seed, learning_rate, output_dir))
        return make_fake_training_result(
            learning_rate,
            seed,
            ap=0.80 + seed / 10000,
        )

    monkeypatch.setattr(
        train_gcn,
        "train_one_run",
        fake_train_one_run,
    )
    monkeypatch.setattr(train_gcn, "OUTPUT_DIR", tmp_path)

    summary = train_gcn.run_final_seed_training(
        make_small_config(),
        make_synthetic_views(),
    )

    assert [call[0] for call in calls] == [42, 43, 44, 45, 46]
    assert all(call[1] == 0.005 for call in calls)
    assert summary["seeds"] == [42, 43, 44, 45, 46]
    assert summary["learning_rate"] == 0.005
    assert summary["test_graph_view_constructed"] is False
    assert summary["test_set_used_for_training_or_selection"] is False
    assert len(summary["results"]) == 5

    saved_summary = json.loads(
        (
            tmp_path
            / "final_five_seed"
            / "five_seed_training_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert saved_summary["status"] == "completed"
    assert saved_summary["seeds"] == [42, 43, 44, 45, 46]
    assert saved_summary["validation_average_precision"]["mean"] == (
        pytest.approx(
            sum(
                item["best_validation_average_precision"]
                for item in summary["results"]
            ) / 5
        )
    )


def test_final_seed_training_rejects_missing_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(train_gcn, "OUTPUT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="pilot first"):
        train_gcn.run_final_seed_training(
            make_small_config(),
            make_synthetic_views(),
        )


@pytest.mark.parametrize(
    "pilot_updates",
    [
        {"status": "running"},
        {"test_set_used": True},
        {"selected_learning_rate": 0},
        {"selected_learning_rate": float("nan")},
    ],
)
def test_final_seed_training_rejects_invalid_pilot_summary(
    tmp_path,
    monkeypatch,
    pilot_updates,
):
    pilot_summary = {
        "status": "completed",
        "selected_learning_rate": 0.005,
        "test_set_used": False,
        "results": [
            make_fake_training_result(0.005, 42, 0.80),
        ],
    }
    pilot_summary.update(pilot_updates)

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "learning_rate_pilot_summary.json").write_text(
        json.dumps(pilot_summary),
        encoding="utf-8",
    )

    monkeypatch.setattr(train_gcn, "OUTPUT_DIR", tmp_path)

    with pytest.raises(ValueError):
        train_gcn.run_final_seed_training(
            make_small_config(),
            make_synthetic_views(),
        )
