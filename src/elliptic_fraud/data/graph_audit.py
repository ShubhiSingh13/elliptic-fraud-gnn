
"""Audit the Elliptic transaction graph and its temporal snapshots."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[3]

GRAPH_PATH = ROOT / "data/processed/elliptic/transaction_graph.pt"
SCALER_PATH = ROOT / "data/processed/elliptic/feature_scaler.joblib"
SCALED_FEATURES_PATH = (
    ROOT / "data/processed/elliptic/scaled_features.pt"
)
TRANSACTIONS_PATH = ROOT / "data/interim/elliptic/transactions.parquet"
SPLITS_PATH = ROOT / "data/processed/elliptic/temporal_splits.parquet"

OUTPUT_DIR = ROOT / "artifacts/reports/graph_audit"
CSV_PATH = OUTPUT_DIR / "per_step_graph_audit.csv"
JSON_PATH = OUTPUT_DIR / "graph_audit_summary.json"

EXPECTED_NODES = 203_769
EXPECTED_EDGES = 234_355
EXPECTED_FEATURES = 165
EXPECTED_STEPS = 49
EXPECTED_TRAIN_SCALER_ROWS = 120_804

EXPECTED_LABEL_COUNTS = {
    -1: 157_205,
    0: 42_019,
    1: 4_545,
}


def load_graph():
    """Load the saved graph safely on CPU."""
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(graph, dict):
        raise TypeError("Expected the saved graph to be a dictionary.")

    return graph


def calculate_snapshot_statistics(
    step,
    node_indices,
    source,
    target,
    time_steps,
    labels,
):
    """Calculate structural statistics for one time-step snapshot."""

    num_nodes = len(node_indices)

    # Keep only edges whose source AND target belong to this step.
    edge_mask = (
        (time_steps[source] == step)
        & (time_steps[target] == step)
    )

    step_source_global = source[edge_mask]
    step_target_global = target[edge_mask]

    # Convert global node indices to local snapshot indices.
    global_to_local = np.full(
        len(time_steps),
        -1,
        dtype=np.int64,
    )
    global_to_local[node_indices] = np.arange(num_nodes)

    step_source = global_to_local[step_source_global]
    step_target = global_to_local[step_target_global]

    num_edges = len(step_source)

    self_loops = int(np.sum(step_source == step_target))

    encoded_edges = (
        step_source.astype(np.int64) * num_nodes
        + step_target.astype(np.int64)
    )

    duplicate_edges = int(
        num_edges - len(np.unique(encoded_edges))
    )

    reverse_edges = (
        step_target.astype(np.int64) * num_nodes
        + step_source.astype(np.int64)
    )

    reciprocal_edge_count = int(
        np.isin(reverse_edges, encoded_edges).sum()
    )

    # In-degree: number of incoming edges per node.
    in_degree = np.bincount(
        step_target,
        minlength=num_nodes,
    )

    # Out-degree: number of outgoing edges per node.
    out_degree = np.bincount(
        step_source,
        minlength=num_nodes,
    )

    # Build a sparse adjacency matrix for component analysis.
    adjacency = csr_matrix(
        (
            np.ones(num_edges, dtype=np.int8),
            (step_source, step_target),
        ),
        shape=(num_nodes, num_nodes),
    )

    weak_count, _ = connected_components(
        adjacency,
        directed=True,
        connection="weak",
        return_labels=True,
    )

    strong_count, _ = connected_components(
        adjacency,
        directed=True,
        connection="strong",
        return_labels=True,
    )

    # A directed graph is acyclic iff every strongly connected
    # component contains one node and there are no self-loops.
    is_acyclic = (
        strong_count == num_nodes
        and self_loops == 0
    )

    step_labels = labels[node_indices]

    return {
        "time_step": int(step),
        "nodes": int(num_nodes),
        "edges": int(num_edges),
        "self_loops": self_loops,
        "duplicate_edges": duplicate_edges,
        "reciprocal_edge_count": reciprocal_edge_count,
        "weak_components": int(weak_count),
        "strong_components": int(strong_count),
        "is_acyclic": bool(is_acyclic),
        "isolated_nodes": int(
            np.sum((in_degree + out_degree) == 0)
        ),
        "mean_in_degree": float(in_degree.mean()),
        "mean_out_degree": float(out_degree.mean()),
        "max_in_degree": int(in_degree.max()),
        "max_out_degree": int(out_degree.max()),
        "labelled_licit": int(np.sum(step_labels == 0)),
        "labelled_illicit": int(np.sum(step_labels == 1)),
        "unlabelled": int(np.sum(step_labels == -1)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading graph and supporting files...")

    graph = load_graph()

    transactions = pd.read_parquet(TRANSACTIONS_PATH)
    splits = pd.read_parquet(SPLITS_PATH)
    scaler = joblib.load(SCALER_PATH)
    scaled_features = torch.load(
        SCALED_FEATURES_PATH,
        map_location="cpu",
        weights_only=True,
    )

    x = graph["x"].cpu()
    edge_index = graph["edge_index"].cpu()
    labels = graph["y"].cpu().numpy()
    tx_ids = graph["tx_id"].cpu().numpy()
    time_steps = graph["time_step"].cpu().numpy()
    split = graph["split"].cpu().numpy()

    source = edge_index[0].numpy()
    target = edge_index[1].numpy()

    num_nodes = len(tx_ids)
    num_edges = edge_index.shape[1]

    print("Checking basic graph dimensions...")

    assert num_nodes == EXPECTED_NODES, (
        f"Unexpected node count: {num_nodes}"
    )
    assert num_edges == EXPECTED_EDGES, (
        f"Unexpected edge count: {num_edges}"
    )
    assert x.shape == (EXPECTED_NODES, EXPECTED_FEATURES)
    assert edge_index.shape == (2, EXPECTED_EDGES)
    assert labels.shape == (EXPECTED_NODES,)
    assert len(np.unique(tx_ids)) == num_nodes

    assert np.isfinite(x.numpy()).all(), (
        "Graph features contain NaN or infinity."
    )

    assert set(np.unique(time_steps)) == set(
        range(1, EXPECTED_STEPS + 1)
    )

    label_counts = {
        int(label): int(count)
        for label, count in zip(
            *np.unique(labels, return_counts=True)
        )
    }

    assert label_counts == EXPECTED_LABEL_COUNTS, (
        f"Unexpected label counts: {label_counts}"
    )

    print("Checking transaction ID alignment...")

    assert np.array_equal(
        tx_ids,
        transactions["tx_id"].to_numpy(),
    ), "Graph node IDs do not match transaction row order."

    assert np.array_equal(
        time_steps,
        transactions["time_step"].to_numpy(),
    ), "Graph time steps do not match transaction row order."

    assert len(splits) == num_nodes
    assert np.array_equal(
        tx_ids,
        splits["tx_id"].to_numpy(),
    ), "Graph node IDs do not match split row order."

    print("Checking edge integrity...")

    assert source.min() >= 0 and source.max() < num_nodes
    assert target.min() >= 0 and target.max() < num_nodes

    cross_step_edges = int(
        np.sum(time_steps[source] != time_steps[target])
    )

    assert cross_step_edges == 0, (
        f"Found {cross_step_edges} cross-time-step edges."
    )

    encoded_all_edges = (
        source.astype(np.int64) * num_nodes
        + target.astype(np.int64)
    )

    duplicate_edges = int(
        num_edges - len(np.unique(encoded_all_edges))
    )

    self_loops = int(np.sum(source == target))

    reciprocal_edge_count = int(
        np.isin(
            target.astype(np.int64) * num_nodes
            + source.astype(np.int64),
            encoded_all_edges,
        ).sum()
    )

    print("Checking temporal split masks...")

    expected_split = np.where(
        time_steps <= 29,
        0,
        np.where(time_steps <= 34, 1, 2),
    )

    assert np.array_equal(split, expected_split), (
        "Saved split assignments do not match the planned periods."
    )

    # Context masks are cumulative node masks.
    assert torch.equal(
        graph["train_mask"],
        torch.from_numpy(time_steps <= 29),
    )
    assert torch.equal(
        graph["validation_mask"],
        torch.from_numpy(time_steps <= 34),
    )
    assert torch.equal(
        graph["test_mask"],
        torch.from_numpy(time_steps <= 49),
    )

    print("Checking saved chronological edge views...")

    for name, cutoff in (
        ("train_edge_index", 29),
        ("validation_edge_index", 34),
        ("test_edge_index", 49),
    ):
        expected_mask = (
            (time_steps[source] <= cutoff)
            & (time_steps[target] <= cutoff)
        )

        expected_edges = edge_index[
            :,
            torch.from_numpy(expected_mask),
        ]

        actual_edges = graph[name].cpu()

        assert torch.equal(actual_edges, expected_edges), (
            f"{name} does not match the expected chronological edges."
        )

    print("Checking scaler and scaled feature matrix...")

    assert scaled_features.shape == x.shape, (
        "Scaled feature tensor shape does not match graph features."
    )

    assert torch.isfinite(scaled_features).all(), (
        "Scaled feature tensor contains NaN or infinity."
    )

    scaler_rows = np.asarray(scaler.n_samples_seen_)

    assert np.all(
        scaler_rows == EXPECTED_TRAIN_SCALER_ROWS
    ), (
        "Scaler was not fitted on exactly "
        f"{EXPECTED_TRAIN_SCALER_ROWS} training-period nodes. "
        f"Observed: {scaler_rows}"
    )

    assert scaler.n_features_in_ == EXPECTED_FEATURES

    print("Auditing all 49 time steps...")

    per_step_rows = []

    for step in range(1, EXPECTED_STEPS + 1):
        node_indices = np.flatnonzero(time_steps == step)

        row = calculate_snapshot_statistics(
            step=step,
            node_indices=node_indices,
            source=source,
            target=target,
            time_steps=time_steps,
            labels=labels,
        )

        per_step_rows.append(row)

        print(
            f"Step {step:02d}: "
            f"{row['nodes']:,} nodes, "
            f"{row['edges']:,} edges, "
            f"{row['weak_components']} weak components, "
            f"{row['strong_components']} strong components"
        )

    report = pd.DataFrame(per_step_rows)
    report.to_csv(CSV_PATH, index=False)

    summary = {
        "status": "passed",
        "nodes": int(num_nodes),
        "edges": int(num_edges),
        "features": int(x.shape[1]),
        "time_steps": int(EXPECTED_STEPS),
        "label_counts": label_counts,
        "cross_step_edges": cross_step_edges,
        "duplicate_edges": duplicate_edges,
        "self_loops": self_loops,
        "reciprocal_edge_count": reciprocal_edge_count,
        "scaler_training_rows": int(
            np.asarray(scaler.n_samples_seen_).max()
        ),
        "scaler_features": int(scaler.n_features_in_),
        "all_features_finite": True,
        "all_scaled_features_finite": True,
        "all_snapshot_statistics_generated": (
            len(report) == EXPECTED_STEPS
        ),
        "all_snapshots_acyclic": bool(
            report["is_acyclic"].all()
        ),
        "all_snapshots_have_one_weak_component": bool(
            (report["weak_components"] == 1).all()
        ),
        "total_isolated_nodes": int(
            report["isolated_nodes"].sum()
        ),
        "per_step_report": str(CSV_PATH.relative_to(ROOT)),
    }

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nGraph audit completed successfully.")
    print(f"CSV report:  {CSV_PATH}")
    print(f"JSON report: {JSON_PATH}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
