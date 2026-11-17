
"""Leakage-aware chronological graph views for Elliptic Sentinel."""

from __future__ import annotations

from typing import Any

import torch


_SPLIT_CONFIG = {
    "train": {
        "context_mask": "train_mask",
        "edge_index": "train_edge_index",
        "split_id": 0,
    },
    "validation": {
        "context_mask": "validation_mask",
        "edge_index": "validation_edge_index",
        "split_id": 1,
    },
    "test": {
        "context_mask": "test_mask",
        "edge_index": "test_edge_index",
        "split_id": 2,
    },
}


def build_asof_graph_view(
    graph: dict[str, Any],
    split: str,
) -> dict[str, torch.Tensor]:
    """Build the graph context available at a chronological split.

    Context nodes:
        Train: steps 1-29
        Validation: steps 1-34
        Test: steps 1-49

    Supervised targets:
        Only labelled nodes belonging to the requested split.

    Unknown-label nodes may remain in the graph context, but are never
    included in the supervised target mask.
    """
    if split not in _SPLIT_CONFIG:
        raise ValueError(
            f"Unknown split {split!r}. "
            f"Expected one of {tuple(_SPLIT_CONFIG)}."
        )

    required = {
        "x",
        "edge_index",
        "y",
        "time_step",
        "split",
        "train_mask",
        "validation_mask",
        "test_mask",
        "train_edge_index",
        "validation_edge_index",
        "test_edge_index",
    }
    missing = required.difference(graph)
    if missing:
        raise KeyError(f"Graph is missing required keys: {sorted(missing)}")

    config = _SPLIT_CONFIG[split]

    x = graph["x"]
    y = graph["y"]
    time_step = graph["time_step"]
    split_ids = graph["split"]
    context_mask = graph[config["context_mask"]]
    edge_index = graph[config["edge_index"]]

    if x.ndim != 2:
        raise ValueError("x must be a 2D node-feature tensor.")

    num_nodes = x.size(0)

    for name, tensor in (
        ("y", y),
        ("time_step", time_step),
        ("split", split_ids),
        ("context_mask", context_mask),
    ):
        if tensor.ndim != 1 or tensor.numel() != num_nodes:
            raise ValueError(
                f"{name} must be a 1D tensor with {num_nodes} entries."
            )

    if context_mask.dtype != torch.bool:
        raise TypeError("Context masks must have boolean dtype.")

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long.")

    if edge_index.numel() > 0:
        if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
            raise ValueError("edge_index contains an invalid node index.")

        # Both endpoints must be available in this chronological view.
        if not context_mask[edge_index].all().item():
            raise ValueError(
                f"{split} graph contains an edge connected to a node "
                "outside its chronological context."
            )

    # All context nodes must be from the current or an earlier time step.
    max_step = {
        "train": 29,
        "validation": 34,
        "test": 49,
    }[split]

    if context_mask.any():
        if (time_step[context_mask] > max_step).any().item():
            raise ValueError(
                f"{split} context contains a future time step."
            )

    # Supervise only labelled nodes belonging to this split.
    target_mask = (split_ids == config["split_id"]) & (y >= 0)

    if (target_mask & ~context_mask).any().item():
        raise ValueError("A supervised target is outside the graph context.")

    return {
        "x": x,
        "edge_index": edge_index,
        "y": y,
        "time_step": time_step,
        "context_mask": context_mask,
        "target_mask": target_mask,
    }


def get_supervised_targets(
    graph_view: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return labels and mask for supervised evaluation or training."""
    if "y" not in graph_view or "target_mask" not in graph_view:
        raise KeyError("Graph view must contain y and target_mask.")

    return graph_view["y"], graph_view["target_mask"]
