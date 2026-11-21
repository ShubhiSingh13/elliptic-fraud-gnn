import json

import numpy as np
import pytest
import torch

from elliptic_fraud.evaluation import evaluate_gcn
from elliptic_fraud.models.gcn import EllipticGCN


def make_checkpoint(model):
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": 4,
            "hidden_channels": 8,
            "num_layers": 2,
            "dropout": 0.2,
            "use_degree_features": False,
            "graph_direction": "undirected",
        },
        "seed": 42,
        "learning_rate": 0.001,
        "best_epoch": 5,
        "best_validation_average_precision": 0.8,
    }


def test_load_gcn_checkpoint(tmp_path):
    original_model = EllipticGCN(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        dropout=0.2,
        use_degree_features=False,
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save(make_checkpoint(original_model), checkpoint_path)

    model, checkpoint = evaluate_gcn.load_gcn_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )

    assert isinstance(model, EllipticGCN)
    assert checkpoint["seed"] == 42
    assert model.training is False


def test_load_gcn_checkpoint_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_gcn.load_gcn_checkpoint(
            tmp_path / "missing.pt",
            torch.device("cpu"),
        )


def test_load_gcn_checkpoint_rejects_missing_keys(tmp_path):
    checkpoint_path = tmp_path / "incomplete.pt"
    torch.save({"seed": 42}, checkpoint_path)

    with pytest.raises(ValueError, match="missing keys"):
        evaluate_gcn.load_gcn_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
        )


def test_load_gcn_checkpoint_rejects_directed_checkpoint(tmp_path):
    model = EllipticGCN(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        dropout=0.2,
        use_degree_features=False,
    )
    checkpoint = make_checkpoint(model)
    checkpoint["model_config"]["graph_direction"] = "directed"

    checkpoint_path = tmp_path / "directed.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="undirected"):
        evaluate_gcn.load_gcn_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
        )


def test_predict_probabilities_returns_valid_values():
    model = EllipticGCN(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        dropout=0.2,
        use_degree_features=False,
    )
    graph_view = {
        "x": torch.randn(5, 4),
        "edge_index": torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            dtype=torch.long,
        ),
    }

    probabilities = evaluate_gcn.predict_probabilities(model, graph_view)

    assert probabilities.shape == (5,)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_predict_probabilities_rejects_non_finite_values():
    class InvalidModel(torch.nn.Module):
        def forward(self, x, edge_index):
            return torch.full(
                (len(x),),
                float("nan"),
                dtype=x.dtype,
                device=x.device,
            )

    graph_view = {
        "x": torch.randn(3, 4),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }

    with pytest.raises(ValueError, match="NaN or infinite"):
        evaluate_gcn.predict_probabilities(InvalidModel(), graph_view)


def make_graph_view(time_step):
    return {
        "x": torch.zeros((6, 4), dtype=torch.float32),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "time_step": torch.full((6,), time_step, dtype=torch.long),
        "context_mask": torch.ones(6, dtype=torch.bool),
        "target_mask": torch.tensor(
            [True, True, True, True, False, False]
        ),
        "y": torch.tensor([0, 1, 0, 1, -1, -1]),
    }


def test_run_evaluation_calibrates_before_test_and_saves_outputs(
    tmp_path,
    monkeypatch,
):
    seeds = [42, 43, 44, 45, 46]
    events = []

    config = {
        "gnn": {"device": "cpu"},
        "split": {
            "validation": {
                "start_time_step": 30,
                "end_time_step": 30,
            },
            "test": {
                "start_time_step": 35,
                "end_time_step": 35,
            },
        },
        "evaluation": {"alert_fraction": 0.01},
    }

    training_summary = {
        "seeds": seeds,
        "learning_rate": 0.001,
    }

    graph = {
        "tx_id": torch.tensor([100, 101, 102, 103, 104, 105]),
        "y": torch.tensor([0, 1, 0, 1, -1, -1]),
    }

    validation_view = make_graph_view(30)
    test_view = make_graph_view(35)

    class DummyModel:
        def cpu(self):
            return self

    dummy_model = DummyModel()

    checkpoint = {
        "seed": 42,
        "learning_rate": 0.001,
        "best_epoch": 5,
        "best_validation_average_precision": 0.8,
    }

    monkeypatch.setattr(
        evaluate_gcn,
        "load_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        evaluate_gcn,
        "load_final_training_summary",
        lambda path: training_summary,
    )
    monkeypatch.setattr(
        evaluate_gcn,
        "resolve_device",
        lambda device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        evaluate_gcn,
        "load_graph_data",
        lambda device: graph,
    )

    def build_view(graph_data, split):
        events.append(f"build_{split}")
        return validation_view if split == "validation" else test_view

    monkeypatch.setattr(
        evaluate_gcn,
        "build_asof_graph_view",
        build_view,
    )

    def fake_load_checkpoint(path, device):
        seed = int(path.stem.split("_")[2])
        seed_checkpoint = {**checkpoint, "seed": seed}
        return dummy_model, seed_checkpoint

    monkeypatch.setattr(
        evaluate_gcn,
        "load_gcn_checkpoint",
        fake_load_checkpoint,
    )

    probabilities = np.array(
        [0.1, 0.9, 0.2, 0.8, 0.5, 0.4],
        dtype=np.float64,
    )

    monkeypatch.setattr(
        evaluate_gcn,
        "predict_probabilities",
        lambda model, view: probabilities.copy(),
    )

    original_select_threshold = evaluate_gcn.select_f1_threshold

    def record_threshold_selection(scores, labels):
        events.append("calibrate_validation")
        return original_select_threshold(scores, labels)

    monkeypatch.setattr(
        evaluate_gcn,
        "select_f1_threshold",
        record_threshold_selection,
    )
    monkeypatch.setattr(evaluate_gcn, "OUTPUT_DIR", tmp_path)

    result = evaluate_gcn.run_evaluation()

    # All five validation calibrations must happen before test construction.
    assert events.count("calibrate_validation") == 5
    assert events.index("build_test") > max(
        index
        for index, event in enumerate(events)
        if event == "calibrate_validation"
    )

    assert result["evaluation_status"] == "completed"
    assert result["test_status"] == "exploratory"
    assert result["seeds"] == seeds
    assert len(result["validation_calibration"]["per_seed"]) == 5
    assert len(result["test_evaluation"]["per_seed"]) == 5

    assert result["capacity_policy"]["uses_labels"] is False
    assert result["validation_calibration"]["frozen_thresholds"]

    summary_path = tmp_path / "five_seed_evaluation_summary.json"
    prediction_path = tmp_path / "five_seed_test_transaction_scores.npz"

    assert summary_path.exists()
    assert prediction_path.exists()

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["test_status"] == "exploratory"

    with np.load(prediction_path) as saved_predictions:
        assert saved_predictions["tx_id"].shape == (6,)
        assert saved_predictions["probability_by_seed"].shape == (5, 6)
        assert saved_predictions["ensemble_probability"].shape == (6,)
        assert saved_predictions["seeds"].tolist() == seeds
