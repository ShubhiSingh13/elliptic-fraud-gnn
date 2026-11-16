
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/interim/elliptic")

FEATURES_FILE = DATA_DIR / "transactions.parquet"
EDGES_FILE = DATA_DIR / "edges.parquet"
LABELS_FILE = DATA_DIR / "labels.parquet"


def verify_dataset():
    print("\nELLIPTIC PROCESSED DATASET VERIFICATION")
    print("=" * 50)

    transactions = pd.read_parquet(FEATURES_FILE)
    edges = pd.read_parquet(EDGES_FILE)
    labels = pd.read_parquet(LABELS_FILE)

    print("\n1. TRANSACTION FEATURES")
    print(f"Shape: {transactions.shape}")
    print(f"Columns: {len(transactions.columns)}")
    print(f"Unique IDs: {transactions['tx_id'].nunique():,}")
    print(f"Time steps: {transactions['time_step'].nunique()}")

    print("\n2. GRAPH EDGES")
    print(f"Shape: {edges.shape}")
    print(f"Unique edges: {len(edges.drop_duplicates()):,}")

    print("\n3. TRANSACTION LABELS")
    print(f"Shape: {labels.shape}")
    print("\nLabel distribution:")
    print(labels["label"].value_counts().sort_index())

    print("\nOriginal class distribution:")
    print(labels["class"].value_counts())

    print("\n4. DATA TYPE CHECK")
    print(transactions.dtypes.head(5))

    print("\n5. FEATURE VALIDATION")
    feature_columns = [
        col for col in transactions.columns
        if col.startswith("feature_")
    ]

    print(f"Feature count: {len(feature_columns)}")
    print(
        "Missing feature values:",
        transactions[feature_columns].isna().sum().sum(),
    )

    print("\n" + "=" * 50)
    print("PROCESSED DATASET VERIFICATION COMPLETED")


if __name__ == "__main__":
    verify_dataset()
