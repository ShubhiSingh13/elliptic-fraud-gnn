import pytest
import torch

from elliptic_fraud.evaluation.evaluate_gnn import build_period_mask


def make_graph_view():
    """Create a small cumulative graph view for testing."""
    return {
        "time_step": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        "context_mask": torch.tensor(
            [True, True, True, True, True, True, True, True]
        ),
    }


def test_validation_period_mask_excludes_historical_transactions():
    view = make_graph_view()

    mask = build_period_mask(
        view,
        start_time_step=4,
        end_time_step=5,
    )

    assert mask.tolist() == [
        False, False, False, True, True, False, False, False
    ]
    assert int(mask.sum()) == 2


def test_test_period_mask_excludes_validation_and_training_history():
    view = make_graph_view()

    mask = build_period_mask(
        view,
        start_time_step=6,
        end_time_step=8,
    )

    assert mask.tolist() == [
        False, False, False, False, False, True, True, True
    ]
    assert int(mask.sum()) == 3


def test_period_mask_respects_context_mask():
    view = make_graph_view()
    view["context_mask"] = torch.tensor(
        [True, True, True, True, False, True, True, True]
    )

    mask = build_period_mask(
        view,
        start_time_step=4,
        end_time_step=6,
    )

    assert mask.tolist() == [
        False, False, False, True, False, True, False, False
    ]


def test_period_mask_rejects_invalid_range():
    view = make_graph_view()

    with pytest.raises(ValueError):
        build_period_mask(view, 5, 3)


def test_period_mask_rejects_period_without_transactions():
    view = make_graph_view()

    with pytest.raises(ValueError):
        build_period_mask(view, 10, 12)


def test_period_mask_rejects_missing_time_step():
    view = {"context_mask": torch.ones(8, dtype=torch.bool)}

    with pytest.raises(KeyError):
        build_period_mask(view, 4, 5)