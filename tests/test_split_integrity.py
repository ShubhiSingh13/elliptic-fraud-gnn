
from pathlib import Path

import torch


GRAPH_FILE = Path("data/processed/elliptic/transaction_graph.pt")


def load_graph():
    assert GRAPH_FILE.exists(), "Transaction graph file is missing."
    return torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=True,
    )


def test_temporal_splits_are_correct():
    graph = load_graph()
    time_step = graph["time_step"]
    split = graph["split"]

    assert torch.all(split[time_step <= 29] == 0)
    assert torch.all(
        split[(time_step >= 30) & (time_step <= 34)] == 1
    )
    assert torch.all(split[time_step >= 35] == 2)


def test_split_masks_do_not_overlap_as_target_periods():
    graph = load_graph()
    split = graph["split"]

    train = split == 0
    validation = split == 1
    test = split == 2

    assert not (train & validation).any()
    assert not (train & test).any()
    assert not (validation & test).any()

    assert (train | validation | test).all()


def test_unknown_labels_are_not_supervised_targets():
    graph = load_graph()
    y = graph["y"]
    split = graph["split"]

    for split_id in (0, 1, 2):
        targets = (split == split_id) & (y >= 0)

        assert not (targets & (y == -1)).any()


def test_cumulative_context_masks_are_correct():
    graph = load_graph()
    time_step = graph["time_step"]

    assert torch.equal(
        graph["train_mask"],
        time_step <= 29,
    )

    assert torch.equal(
        graph["validation_mask"],
        time_step <= 34,
    )

    assert torch.equal(
        graph["test_mask"],
        time_step <= 49,
    )


def test_training_graph_contains_no_future_nodes_or_edges():
    graph = load_graph()
    edge_index = graph["train_edge_index"]
    time_step = graph["time_step"]

    source_steps = time_step[edge_index[0]]
    target_steps = time_step[edge_index[1]]

    assert torch.all(source_steps <= 29)
    assert torch.all(target_steps <= 29)

    assert torch.all(source_steps >= 1)
    assert torch.all(target_steps >= 1)
