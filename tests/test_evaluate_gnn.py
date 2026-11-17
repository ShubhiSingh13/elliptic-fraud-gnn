import json

import pytest
import torch

from elliptic_fraud.evaluation.evaluate_gnn import (
    load_selected_checkpoint,
    load_model_from_checkpoint,
    predict_probabilities,
)
from elliptic_fraud.models.gnn import DirectedGraphSAGE


def test_load_selected_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "selected.pt"
    checkpoint_path.touch()

    summary_path = tmp_path / "pilot.json"
    summary = {
        "status": "completed",
        "test_set_used": False,
        "selected_learning_rate": 0.001,
        "results": [
            {
                "learning_rate": 0.001,
                "checkpoint": str(checkpoint_path),
            }
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result, loaded_path = load_selected_checkpoint(summary_path)

    assert result["learning_rate"] == 0.001
    assert loaded_path == checkpoint_path


def test_checkpoint_loader_rejects_test_usage(tmp_path):
    summary_path = tmp_path / "pilot.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "test_set_used": True,
                "selected_learning_rate": 0.001,
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test set was excluded"):
        load_selected_checkpoint(summary_path)


def test_checkpoint_loader_rejects_missing_file(tmp_path):
    summary_path = tmp_path / "pilot.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "test_set_used": False,
                "selected_learning_rate": 0.001,
                "results": [
                    {
                        "learning_rate": 0.001,
                        "checkpoint": str(tmp_path / "missing.pt"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        load_selected_checkpoint(summary_path)


def test_load_model_from_checkpoint(tmp_path):
    original_model = DirectedGraphSAGE(
        in_channels=4,
        hidden_channels=8,
        dropout=0.2,
        use_degree_features=False,
    )

    checkpoint_path = tmp_path / "model.pt"

    torch.save(
        {
            "model_state_dict": original_model.state_dict(),
            "seed": 42,
            "learning_rate": 0.001,
            "best_epoch": 5,
            "best_validation_average_precision": 0.8,
            "model_config": {
                "in_channels": 4,
                "hidden_channels": 8,
                "dropout": 0.2,
                "use_degree_features": False,
            },
        },
        checkpoint_path,
    )

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )

    assert isinstance(model, DirectedGraphSAGE)
    assert checkpoint["seed"] == 42
    assert model.training is False


def test_predict_probabilities_returns_valid_values():
    model = DirectedGraphSAGE(
        in_channels=4,
        hidden_channels=8,
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

    probabilities = predict_probabilities(model, graph_view)

    assert probabilities.shape == (5,)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()