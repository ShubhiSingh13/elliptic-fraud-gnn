
from pathlib import Path

import pandas as pd
import torch


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/interim/elliptic")
PROCESSED_DIR = Path("data/processed/elliptic")

TRANSACTIONS_FILE = DATA_DIR / "transactions.parquet"
EDGES_FILE = DATA_DIR / "edges.parquet"
LABELS_FILE = DATA_DIR / "labels.parquet"
SPLITS_FILE = PROCESSED_DIR / "temporal_splits.parquet"

OUTPUT_FILE = PROCESSED_DIR / "transaction_graph.pt"

FEATURE_COLUMNS = [
    f"feature_{i:03d}" for i in range(1, 166)
]


# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------

def build_graph():
    print("\nELLIPTIC TRANSACTION GRAPH CONSTRUCTION")
    print("=" * 55)

    transactions = pd.read_parquet(TRANSACTIONS_FILE)
    edges = pd.read_parquet(EDGES_FILE)
    labels = pd.read_parquet(LABELS_FILE)
    splits = pd.read_parquet(SPLITS_FILE)

    # Establish one consistent node-index mapping.
    node_ids = transactions["tx_id"].to_numpy()

    node_to_index = {
        int(tx_id): index
        for index, tx_id in enumerate(node_ids)
    }

    # Convert original transaction IDs into node indices.
    source_indices = edges["txId1"].map(node_to_index)
    target_indices = edges["txId2"].map(node_to_index)

    if source_indices.isna().any() or target_indices.isna().any():
        raise ValueError("An edge references an unknown transaction.")

    edge_index = torch.tensor(
        [
            source_indices.to_numpy(dtype="int64"),
            target_indices.to_numpy(dtype="int64"),
        ],
        dtype=torch.long,
    )

    # Keep numerical features separate from identifiers and time steps.
    x = torch.tensor(
        transactions[FEATURE_COLUMNS].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )

    time_steps = torch.tensor(
        transactions["time_step"].to_numpy(dtype="int64"),
        dtype=torch.long,
    )

    # Keep labels as a separate target tensor.
    label_lookup = labels.set_index("txId")["label"]

    ordered_labels = (
        transactions["tx_id"]
        .map(label_lookup)
        .to_numpy(dtype="int64")
    )

    y = torch.tensor(ordered_labels, dtype=torch.long)

    # Create chronological node masks.
    train_mask = time_steps <= 29
    validation_mask = time_steps <= 34
    test_mask = time_steps <= 49

    # Create chronological edge masks.
    source_steps = time_steps[edge_index[0]]
    target_steps = time_steps[edge_index[1]]

    train_edge_mask = (
        (source_steps <= 29) & (target_steps <= 29)
    )

    validation_edge_mask = (
        (source_steps <= 34) & (target_steps <= 34)
    )

    test_edge_mask = (
        (source_steps <= 49) & (target_steps <= 49)
    )
        # Align split information by transaction ID.
    split_lookup = splits.set_index("tx_id")["split"]
    ordered_splits = transactions["tx_id"].map(split_lookup)

    if ordered_splits.isna().any():
        raise ValueError("Some transactions are missing split assignments.")

    split_codes = ordered_splits.map({
        "train": 0,
        "validation": 1,
        "test": 2,
    }).to_numpy(dtype="int8")

    if (split_codes < 0).any():
        raise ValueError("An invalid split assignment was found.")

    graph = {
        "x": x,
        "edge_index": edge_index,
        "y": y,
        "tx_id": torch.tensor(node_ids, dtype=torch.long),
        "time_step": time_steps,
        "split": torch.tensor(split_codes, dtype=torch.int8),
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,
        "train_edge_index": edge_index[:, train_edge_mask],
        "validation_edge_index": edge_index[:, validation_edge_mask],
        "test_edge_index": edge_index[:, test_edge_mask],
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(graph, OUTPUT_FILE)

    print(f"Total nodes: {x.shape[0]:,}")
    print(f"Node features: {x.shape[1]}")
    print(f"Total edges: {edge_index.shape[1]:,}")

    print("\nChronological graph views:")
    print(f"Training nodes: {train_mask.sum().item():,}")
    print(f"Validation as-of nodes: {validation_mask.sum().item():,}")
    print(f"Test as-of nodes: {test_mask.sum().item():,}")

    print("\nGraph edge counts:")
    print(f"Training edges: {train_edge_mask.sum().item():,}")
    print(f"Validation as-of edges: {validation_edge_mask.sum().item():,}")
    print(f"Test as-of edges: {test_edge_mask.sum().item():,}")

    print(f"\nSaved graph: {OUTPUT_FILE}")
    print("\nGraph construction completed.")


if __name__ == "__main__":
    build_graph()
