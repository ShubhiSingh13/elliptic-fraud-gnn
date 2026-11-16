
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INTERIM_DIR = Path("data/interim/elliptic")
PROCESSED_DIR = Path("data/processed/elliptic")

TRANSACTIONS_FILE = INTERIM_DIR / "transactions.parquet"

SPLIT_OUTPUT = PROCESSED_DIR / "temporal_splits.parquet"

TRAIN_STEPS = range(1, 30)
VALIDATION_STEPS = range(30, 35)
TEST_STEPS = range(35, 50)


# --------------------------------------------------
# CREATE TEMPORAL SPLITS
# --------------------------------------------------

def make_split():
    print("\nELLIPTIC TEMPORAL DATA SPLITTING")
    print("=" * 50)

    transactions = pd.read_parquet(TRANSACTIONS_FILE)

    transactions["split"] = "unassigned"

    transactions.loc[
        transactions["time_step"].isin(TRAIN_STEPS), "split"
    ] = "train"

    transactions.loc[
        transactions["time_step"].isin(VALIDATION_STEPS), "split"
    ] = "validation"

    transactions.loc[
        transactions["time_step"].isin(TEST_STEPS), "split"
    ] = "test"

    if (transactions["split"] == "unassigned").any():
        raise ValueError("Some transactions were not assigned to a split.")

    split_counts = transactions["split"].value_counts()

    print("\nTransaction counts:")

    for split_name in ["train", "validation", "test"]:
        count = int(split_counts[split_name])
        print(f"{split_name.capitalize()}: {count:,}")

    print("\nTime-step ranges:")

    for split_name in ["train", "validation", "test"]:
        subset = transactions[
            transactions["split"] == split_name
        ]

        print(
            f"{split_name.capitalize()}: "
            f"{subset['time_step'].min()}–"
            f"{subset['time_step'].max()}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    transactions[["tx_id", "time_step", "split"]].to_parquet(
        SPLIT_OUTPUT,
        index=False,
    )

    print(f"\nSaved split metadata: {SPLIT_OUTPUT}")
    print("\nTemporal splitting completed successfully.")


if __name__ == "__main__":
    make_split()
