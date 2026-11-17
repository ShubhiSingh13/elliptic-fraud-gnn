
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

SPLIT_MAPPING = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------

def build_graph():
    print("\nELLIPTIC TRANSACTION GRAPH CONSTRUCTION")
    print("=" * 55)

    # Load source data.
    transactions = pd.read_parquet(TRANSACTIONS_FILE)
    edges = pd.read_parquet(EDGES_FILE)
    labels = pd.read_parquet(LABELS_FILE)
    splits = pd.read_parquet(SPLITS_FILE)

    # --------------------------------------------------
    # 1. Validate source data
    # --------------------------------------------------

    if transactions["tx_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")

    if labels["txId"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found in labels.")

    if splits["tx_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found in splits.")

    missing_features = [
        column for column in FEATURE_COLUMNS
        if column not in transactions.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    if not set(transactions["tx_id"]).issubset(set(labels["txId"])):
        raise ValueError("Some transactions are missing labels.")

    if not set(transactions["tx_id"]).issubset(set(splits["tx_id"])):
        raise ValueError("Some transactions are missing split assignments.")

    # --------------------------------------------------
    # 2. Establish consistent node-index mapping
    # --------------------------------------------------

    node_ids = transactions["tx_id"].to_numpy()

    node_to_index = {
        int(tx_id): index
        for index, tx_id in enumerate(node_ids)
    }

    # --------------------------------------------------
    # 3. Convert transaction edges into node indices
    # --------------------------------------------------

    source_indices = edges["txId1"].map(node_to_index)
    target_indices = edges["txId2"].map(node_to_index)

    if source_indices.isna().any() or target_indices.isna().any():
        raise ValueError(
            "An edge references an unknown transaction."
        )

    edge_index = torch.tensor(
        [
            source_indices.to_numpy(dtype="int64"),
            target_indices.to_numpy(dtype="int64"),
        ],
        dtype=torch.long,
    )

    if edge_index.numel() == 0:
        raise ValueError("The graph contains no edges.")

    # --------------------------------------------------
    # 4. Create feature tensor
    # --------------------------------------------------

    x = torch.tensor(
        transactions[FEATURE_COLUMNS].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )

    if not torch.isfinite(x).all():
        raise ValueError(
            "Feature tensor contains NaN or infinite values."
        )

    # --------------------------------------------------
    # 5. Create time-step tensor
    # --------------------------------------------------

    time_steps = torch.tensor(
        transactions["time_step"].to_numpy(dtype="int64"),
        dtype=torch.long,
    )

    if not torch.isin(
        time_steps,
        torch.arange(1, 50, dtype=torch.long),
    ).all():
        raise ValueError("Invalid time-step values found.")

    # --------------------------------------------------
    # 6. Align labels by transaction ID
    # --------------------------------------------------

    label_lookup = labels.set_index("txId")["label"]

    ordered_labels = (
        transactions["tx_id"]
        .map(label_lookup)
    )

    if ordered_labels.isna().any():
        raise ValueError("Some transactions have missing labels.")

    ordered_labels = ordered_labels.to_numpy(dtype="int64")

    if not set(ordered_labels).issubset({-1, 0, 1}):
        raise ValueError("Unexpected label values found.")

    y = torch.tensor(
        ordered_labels,
        dtype=torch.long,
    )

    # --------------------------------------------------
    # 7. Create chronological node context masks
    # --------------------------------------------------

    train_mask = time_steps <= 29
    validation_mask = time_steps <= 34
    test_mask = time_steps <= 49

    # --------------------------------------------------
    # 8. Create chronological edge views
    # --------------------------------------------------

    source_steps = time_steps[edge_index[0]]
    target_steps = time_steps[edge_index[1]]

    train_edge_mask = (
        (source_steps <= 29)
        & (target_steps <= 29)
    )

    validation_edge_mask = (
        (source_steps <= 34)
        & (target_steps <= 34)
    )

    test_edge_mask = (
        (source_steps <= 49)
        & (target_steps <= 49)
    )

    # --------------------------------------------------
    # 9. Align split information by transaction ID
    # --------------------------------------------------

    split_lookup = splits.set_index("tx_id")["split"]

    ordered_splits = (
        transactions["tx_id"]
        .map(split_lookup)
    )

    if ordered_splits.isna().any():
        raise ValueError(
            "Some transactions have missing split assignments."
        )

    split_codes = ordered_splits.map(SPLIT_MAPPING)

    if split_codes.isna().any():
        invalid_count = int(split_codes.isna().sum())

        raise ValueError(
            f"{invalid_count} transactions have invalid split assignments."
        )

    split_codes = split_codes.to_numpy(dtype="int8")

    # --------------------------------------------------
    # 10. Build graph object
    # --------------------------------------------------

    graph = {
        "x": x,
        "edge_index": edge_index,
        "y": y,
        "tx_id": torch.tensor(node_ids, dtype=torch.long),
        "time_step": time_steps,
        "split": torch.tensor(split_codes, dtype=torch.int8),

        # Cumulative node context masks.
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,

        # Cumulative chronological edge views.
        "train_edge_index": edge_index[:, train_edge_mask],
        "validation_edge_index": edge_index[:, validation_edge_mask],
        "test_edge_index": edge_index[:, test_edge_mask],
    }

    # --------------------------------------------------
    # 11. Save graph
    # --------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(graph, OUTPUT_FILE)

    # --------------------------------------------------
    # 12. Print summary
    # --------------------------------------------------

    print(f"Total nodes: {x.shape[0]:,}")
    print(f"Node features: {x.shape[1]}")
    print(f"Total edges: {edge_index.shape[1]:,}")

    print("\nChronological graph views")
    print("-" * 40)

    print(
        f"Training context nodes: "
        f"{train_mask.sum().item():,}"
    )

    print(
        f"Validation as-of nodes: "
        f"{validation_mask.sum().item():,}"
    )

    print(
        f"Test as-of nodes: "
        f"{test_mask.sum().item():,}"
    )

    print("\nGraph edge counts")
    print("-" * 40)

    print(
        f"Training edges: "
        f"{train_edge_mask.sum().item():,}"
    )

    print(
        f"Validation as-of edges: "
        f"{validation_edge_mask.sum().item():,}"
    )

    print(
        f"Test as-of edges: "
        f"{test_edge_mask.sum().item():,}"
    )

    print("\nLabel distribution")
    print("-" * 40)

    for label, name in [
        (-1, "Unknown"),
        (0, "Licit"),
        (1, "Illicit"),
    ]:
        count = int((y == label).sum())
        print(f"{name}: {count:,}")

    print(f"\nSaved graph: {OUTPUT_FILE}")
    print("\nGraph construction completed successfully.")


if __name__ == "__main__":
    build_graph()
