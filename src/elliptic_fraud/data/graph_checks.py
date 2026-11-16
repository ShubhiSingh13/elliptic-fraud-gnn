
from pathlib import Path

import torch


GRAPH_FILE = Path("data/processed/elliptic/transaction_graph.pt")


def check_graph():
    print("\nELLIPTIC GRAPH INTEGRITY CHECKS")
    print("=" * 55)

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_FILE}")

    graph = torch.load(GRAPH_FILE, weights_only=True)

    x = graph["x"]
    edge_index = graph["edge_index"]
    y = graph["y"]
    tx_id = graph["tx_id"]
    time_step = graph["time_step"]
    split = graph["split"]

    # 1. Check tensor dimensions
    assert x.shape == (203769, 165), "Unexpected feature dimensions."
    assert edge_index.shape == (2, 234355), "Unexpected edge dimensions."
    assert y.shape[0] == x.shape[0], "Label count does not match node count."
    assert tx_id.shape[0] == x.shape[0], "Transaction ID count mismatch."
    assert time_step.shape[0] == x.shape[0], "Time-step count mismatch."
    assert split.shape[0] == x.shape[0], "Split count mismatch."

    print("[PASS] Tensor dimensions are correct.")

    # 2. Check transaction IDs are unique
    assert torch.unique(tx_id).numel() == tx_id.numel()
    print("[PASS] Transaction IDs are unique.")

    # 3. Check edge indices are valid
    assert edge_index.min() >= 0
    assert edge_index.max() < x.shape[0]
    print("[PASS] Every edge references a valid node.")

    # 4. Check duplicate directed edges
    unique_edges = torch.unique(edge_index, dim=1)
    assert unique_edges.shape[1] == edge_index.shape[1]
    print("[PASS] No duplicate directed edges.")

    # 5. Check edges do not cross time steps
    source_steps = time_step[edge_index[0]]
    target_steps = time_step[edge_index[1]]

    assert torch.equal(source_steps, target_steps)
    print("[PASS] No edges connect different time steps.")

    # 6. Check chronological split assignments
    expected_split = torch.where(
        time_step <= 29,
        0,
        torch.where(time_step <= 34, 1, 2),
    ).to(torch.int8)

    assert torch.equal(split, expected_split)
    print("[PASS] Train, validation and test assignments are correct.")

    # 7. Check cumulative node context masks
    assert torch.equal(graph["train_mask"], time_step <= 29)
    assert torch.equal(graph["validation_mask"], time_step <= 34)
    assert torch.equal(graph["test_mask"], time_step <= 49)
    print("[PASS] Chronological node context masks are correct.")

    # 8. Check chronological edge views
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

        assert torch.equal(graph[name], expected_edges), (
            f"{name} contains unexpected edges."
        )

    print("[PASS] All chronological edge views are correct.")

    # 9. Check label values
    valid_labels = torch.tensor([-1, 0, 1], dtype=y.dtype)
    assert torch.isin(y, valid_labels).all()
    print("[PASS] Labels contain only -1, 0 and 1.")

    # 10. Check unlabelled transactions are excluded from supervised targets
    train_targets = (split == 0) & (y >= 0)
    validation_targets = (split == 1) & (y >= 0)
    test_targets = (split == 2) & (y >= 0)

    assert not ((train_targets | validation_targets | test_targets) & (y == -1)).any()

    print("[PASS] Unlabelled transactions are excluded from supervised targets.")

    print("\nSUPERVISED TARGET COUNTS")
    print("-" * 40)
    print(f"Training labelled nodes: {train_targets.sum().item():,}")
    print(f"Validation labelled nodes: {validation_targets.sum().item():,}")
    print(f"Test labelled nodes: {test_targets.sum().item():,}")
    print(f"Unlabelled nodes: {(y == -1).sum().item():,}")

    print("\nALL GRAPH INTEGRITY CHECKS PASSED.")


if __name__ == "__main__":
    check_graph()
