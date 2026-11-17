
import pytest
import torch

from elliptic_fraud.training.graph_views import (
    build_asof_graph_view,
    get_supervised_targets,
)
from elliptic_fraud.training.losses import (
    compute_pos_weight,
    masked_weighted_bce_loss,
)


@pytest.fixture
def toy_graph():
    # Six nodes across the three chronological periods.
    # Node 3 is deliberately unlabelled.
    time_step = torch.tensor([1, 29, 30, 34, 35, 49])
    split_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    labels = torch.tensor([0, 1, 0, -1, 1, 0])

    return {
        "x": torch.randn(6, 3),
        "edge_index": torch.tensor(
            [[0, 2, 4], [1, 3, 5]],
            dtype=torch.long,
        ),
        "y": labels,
        "time_step": time_step,
        "split": split_ids,
        "train_mask": torch.tensor(
            [True, True, False, False, False, False]
        ),
        "validation_mask": torch.tensor(
            [True, True, True, True, False, False]
        ),
        "test_mask": torch.tensor(
            [True, True, True, True, True, True]
        ),
        "train_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "validation_edge_index": torch.tensor(
            [[0, 2], [1, 3]], dtype=torch.long
        ),
        "test_edge_index": torch.tensor(
            [[0, 2, 4], [1, 3, 5]], dtype=torch.long
        ),
    }


def test_train_view_contains_only_train_context(toy_graph):
    view = build_asof_graph_view(toy_graph, "train")

    assert view["context_mask"].sum().item() == 2
    assert view["target_mask"].tolist() == [
        True, True, False, False, False, False
    ]
    assert view["edge_index"].shape == (2, 1)


def test_validation_view_uses_past_context_but_validation_targets(
    toy_graph,
):
    view = build_asof_graph_view(toy_graph, "validation")

    assert view["context_mask"].tolist() == [
        True, True, True, True, False, False
    ]
    assert view["target_mask"].tolist() == [
        False, False, True, False, False, False
    ]


def test_test_view_targets_only_labelled_test_nodes(toy_graph):
    view = build_asof_graph_view(toy_graph, "test")

    assert view["context_mask"].all().item()
    assert view["target_mask"].tolist() == [
        False, False, False, False, True, True
    ]


def test_unknown_labels_are_never_targets(toy_graph):
    for split in ("train", "validation", "test"):
        view = build_asof_graph_view(toy_graph, split)
        assert not (view["target_mask"] & (view["y"] == -1)).any()


def test_get_supervised_targets(toy_graph):
    view = build_asof_graph_view(toy_graph, "validation")
    labels, mask = get_supervised_targets(view)

    assert torch.equal(labels, toy_graph["y"])
    assert mask.sum().item() == 1


def test_invalid_split_is_rejected(toy_graph):
    with pytest.raises(ValueError):
        build_asof_graph_view(toy_graph, "future")


def test_future_edge_in_context_is_rejected(toy_graph):
    toy_graph["validation_edge_index"] = torch.tensor(
        [[0, 4], [1, 5]], dtype=torch.long
    )

    with pytest.raises(ValueError, match="outside its chronological context"):
        build_asof_graph_view(toy_graph, "validation")


def test_training_class_weight_uses_only_selected_targets():
    labels = torch.tensor([0, 1, 0, 0, 1, 1])
    train_mask = torch.tensor(
        [True, True, False, False, False, False]
    )

    weight = compute_pos_weight(labels, train_mask)

    # One negative and one positive in the training targets.
    assert weight.item() == pytest.approx(1.0)


def test_class_weight_changes_when_training_class_counts_change():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    train_mask = torch.tensor(
        [True, True, True, True, False, False]
    )

    weight = compute_pos_weight(labels, train_mask)

    assert weight.item() == pytest.approx(3.0)

def test_class_weight_rejects_unknown_selected_labels():
    labels = torch.tensor([0, -1, 1])
    mask = torch.tensor([True, True, False])

    with pytest.raises(ValueError, match="binary labels"):
        compute_pos_weight(labels, mask)


def test_class_weight_rejects_no_positive_examples():
    labels = torch.tensor([0, 0, 0])
    mask = torch.tensor([True, True, True])

    with pytest.raises(ValueError, match="no positive"):
        compute_pos_weight(labels, mask)


def test_masked_loss_ignores_unselected_nodes():
    logits = torch.tensor([0.0, 0.0, 100.0], requires_grad=True)
    labels = torch.tensor([0, 1, 0])
    mask = torch.tensor([True, True, False])
    pos_weight = torch.tensor(1.0)

    loss = masked_weighted_bce_loss(
        logits, labels, mask, pos_weight
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad[2].item() == pytest.approx(0.0)


def test_masked_loss_rejects_unknown_selected_label():
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([0, -1])
    mask = torch.tensor([True, True])

    with pytest.raises(ValueError, match="labels 0 and 1"):
        masked_weighted_bce_loss(
            logits, labels, mask, torch.tensor(1.0)
        )


def test_masked_loss_rejects_empty_target_mask():
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([0, 1])
    mask = torch.tensor([False, False])

    with pytest.raises(ValueError, match="empty target mask"):
        masked_weighted_bce_loss(
            logits, labels, mask, torch.tensor(1.0)
        )
