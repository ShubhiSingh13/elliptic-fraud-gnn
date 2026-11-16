
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RAW_DIR = Path("data/raw/elliptic")
INTERIM_DIR = Path("data/interim/elliptic")

FEATURES_FILE = RAW_DIR / "elliptic_txs_features.csv"
EDGES_FILE = RAW_DIR / "elliptic_txs_edgelist.csv"
CLASSES_FILE = RAW_DIR / "elliptic_txs_classes.csv"

FEATURES_OUTPUT = INTERIM_DIR / "transactions.parquet"
EDGES_OUTPUT = INTERIM_DIR / "edges.parquet"
CLASSES_OUTPUT = INTERIM_DIR / "labels.parquet"

CHUNK_SIZE = 50_000

FEATURE_COLUMNS = [
    f"feature_{i:03d}" for i in range(1, 166)
]


# --------------------------------------------------
# BUILD TRANSACTION FEATURES
# --------------------------------------------------

def build_features():
    print("\n1. Processing transaction features...")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    processed_chunks = []

    for chunk in pd.read_csv(
        FEATURES_FILE,
        header=None,
        chunksize=CHUNK_SIZE,
    ):
        chunk.columns = (
            ["tx_id", "time_step"] + FEATURE_COLUMNS
        )

        chunk["tx_id"] = chunk["tx_id"].astype("int64")
        chunk["time_step"] = chunk["time_step"].astype("int8")

        chunk[FEATURE_COLUMNS] = chunk[FEATURE_COLUMNS].astype(
            "float32"
        )

        processed_chunks.append(chunk)

    transactions = pd.concat(
        processed_chunks,
        ignore_index=True,
    )

    transactions.to_parquet(
        FEATURES_OUTPUT,
        index=False,
    )

    print(f"Transactions processed: {len(transactions):,}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}")
    print(f"Saved: {FEATURES_OUTPUT}")

    del transactions
    del processed_chunks


# --------------------------------------------------
# BUILD TRANSACTION EDGES
# --------------------------------------------------

def build_edges():
    print("\n2. Processing transaction edges...")

    edges = pd.read_csv(
        EDGES_FILE,
        dtype="int64",
    )

    edges.to_parquet(
        EDGES_OUTPUT,
        index=False,
    )

    print(f"Edges processed: {len(edges):,}")
    print(f"Saved: {EDGES_OUTPUT}")

    del edges


# --------------------------------------------------
# BUILD TRANSACTION LABELS
# --------------------------------------------------

def build_labels():
    print("\n3. Processing transaction labels...")

    classes = pd.read_csv(
        CLASSES_FILE,
        dtype={
            "txId": "int64",
            "class": "string",
        },
    )

    classes["label"] = classes["class"].map(
        {
            "1": 1,
            "2": 0,
            "unknown": -1,
        }
    )

    if classes["label"].isna().any():
        raise ValueError("Unexpected class label found.")

    classes["label"] = classes["label"].astype("int8")

    classes.to_parquet(
        CLASSES_OUTPUT,
        index=False,
    )

    print(f"Labels processed: {len(classes):,}")
    print("\nLabel mapping:")
    print("Illicit = 1")
    print("Licit = 0")
    print("Unknown = -1")
    print(f"Saved: {CLASSES_OUTPUT}")

    del classes


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def build_dataset():
    print("=" * 55)
    print("ELLIPTIC DATASET BUILD PIPELINE")
    print("=" * 55)

    build_features()
    build_edges()
    build_labels()

    print("\n" + "=" * 55)
    print("DATASET BUILD COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    build_dataset()
