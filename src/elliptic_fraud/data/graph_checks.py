
from pathlib import Path

import torch


GRAPH_FILE = Path(
    "data/processed/elliptic/transaction_graph.pt"
)


def check_graph():
    print("\nELLIPTIC GRAPH INTEGRITY CHECKS")
    print("=" * 55)

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_FILE}"
        )

    graph = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=True,
    )

    x = graph["x"]
    edge_index = graph["edge_index"]
    y = graph["y"]
    tx_id = graph["tx_id"]
    time_step = graph["time_step"]
    split = graph["split"]

    # --------------------------------------------------
    # 1. Tensor dimensions
    # --------------------------------------------------

    assert x.shape == (203769, 165), (
        "Unexpected feature dimensions."
    )

    assert edge_index.ndim == 2
    assert edge_index.shape == (2, 234355), (
        "Unexpected edge dimensions."
    )

    assert y.shape[0] == x.shape[0]
    assert tx_id.shape[0] == x.shape[0]
    assert time_step.shape[0] == x.shape[0]
    assert split.shape[0] == x.shape[0]

    assert x.dtype == torch.float32
    assert edge_index.dtype == torch.long
    assert y.dtype == torch.long

    assert torch.isfinite(x).all()

    print("[PASS] Tensor dimensions and dtypes are correct.")

    # --------------------------------------------------
    # 2. Unique transaction IDs
    # --------------------------------------------------

    assert torch.unique(tx_id).numel() == tx_id.numel()

    print("[PASS] Transaction IDs are unique.")

    # --------------------------------------------------
    # 3. Edge index validation
    # --------------------------------------------------

    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0

    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < x.shape[0]

    print("[PASS] Every edge references a valid node.")

    # --------------------------------------------------
    # 4. Duplicate directed edges
    # --------------------------------------------------

    unique_edges = torch.unique(
        edge_index,
        dim=1,
    )

    assert unique_edges.shape[1] == edge_index.shape[1]

    print("[PASS] No duplicate directed edges.")

    # --------------------------------------------------
    # 5. No cross-time-step edges
    # --------------------------------------------------

    source_steps = time_step[edge_index[0]]
    target_steps = time_step[edge_index[1]]

    assert torch.equal(
        source_steps,
        target_steps,
    )

    print("[PASS] No edges connect different time steps.")

    # --------------------------------------------------
    # 6. Chronological split assignments
    # --------------------------------------------------

    expected_split = torch.where(
        time_step <= 29,
        0,
        torch.where(
            time_step <= 34,
            1,
            2,
        ),
    ).to(torch.int8)

    assert torch.equal(
        split,
        expected_split,
    )

    print("[PASS] Train, validation and test assignments are correct.")

    # --------------------------------------------------
    # 7. Cumulative node context masks
    # --------------------------------------------------

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

    print("[PASS] Chronological node context masks are correct.")

    # --------------------------------------------------
    # 8. Chronological edge views
    # --------------------------------------------------

    for name, cutoff in [
        ("train_edge_index", 29),
        ("validation_edge_index", 34),
        ("test_edge_index", 49),
    ]:

        keep = (
            (source_steps <= cutoff)
            & (target_steps <= cutoff)
        )

        expected_edges = edge_index[:, keep]

        assert torch.equal(
            graph[name],
            expected_edges,
        ), f"{name} contains unexpected edges."

    print("[PASS] All chronological edge views are correct.")

    # --------------------------------------------------
    # 9. Label validation
    # --------------------------------------------------

    valid_labels = torch.tensor(
        [-1, 0, 1],
        dtype=y.dtype,
    )

    assert torch.isin(
        y,
        valid_labels,
    ).all()

    print("[PASS] Labels contain only -1, 0 and 1.")

    # --------------------------------------------------
    # 10. Supervised target masks
    # --------------------------------------------------

    train_targets = (split == 0) & (y >= 0)

    validation_targets = (
        (split == 1) & (y >= 0)
    )

    test_targets = (
        (split == 2) & (y >= 0)
    )

    assert not (
        train_targets & validation_targets
    ).any()

    assert not (
        train_targets & test_targets
    ).any()

    assert not (
        validation_targets & test_targets
    ).any()

    combined_targets = (
        train_targets
        | validation_targets
        | test_targets
    )

    assert combined_targets.sum().item() == (
        y >= 0
    ).sum().item()

    assert not (
        combined_targets & (y == -1)
    ).any()

    print(
        "[PASS] Supervised targets are mutually exclusive "
        "and exclude unknown labels."
    )

    # --------------------------------------------------
    # 11. Verify supervised class distributions
    # --------------------------------------------------

    print("\nSUPERVISED TARGET COUNTS")
    print("-" * 40)

    for name, mask in [
        ("Training", train_targets),
        ("Validation", validation_targets),
        ("Test", test_targets),
    ]:

        labels = y[mask]

        licit = int((labels == 0).sum())
        illicit = int((labels == 1).sum())

        print(f"{name} labelled nodes: {len(labels):,}")
        print(f"  Licit:   {licit:,}")
        print(f"  Illicit: {illicit:,}")

    print(
        f"Unlabelled nodes: {(y == -1).sum().item():,}"
    )

    print("\nALL GRAPH INTEGRITY CHECKS PASSED.")


if __name__ == "__main__":
    check_graph()
