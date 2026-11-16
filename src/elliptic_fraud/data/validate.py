
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/raw/elliptic")

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"

EXPECTED_FEATURE_COLUMNS = 167
EXPECTED_TIME_STEPS = 49
CHUNK_SIZE = 50_000


# --------------------------------------------------
# BASIC FILE VALIDATION
# --------------------------------------------------

def validate_file_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    print(f"FOUND: {path.name}")


# --------------------------------------------------
# FEATURE VALIDATION
# --------------------------------------------------

def validate_features():
    print("\n1. FEATURE DATA VALIDATION")

    transaction_ids = set()
    transaction_steps = {}
    step_counts = {step: 0 for step in range(1, EXPECTED_TIME_STEPS + 1)}

    duplicate_count = 0
    invalid_value_count = 0
    invalid_step_count = 0
    total_rows = 0

    for chunk in pd.read_csv(
        FEATURES_FILE,
        header=None,
        dtype={0: "string"},
        chunksize=CHUNK_SIZE,
    ):
        if len(chunk.columns) != EXPECTED_FEATURE_COLUMNS:
            raise ValueError(
                f"Expected 167 columns, found {len(chunk.columns)}"
            )

        ids = chunk.iloc[:, 0]

        if ids.isna().any():
            raise ValueError("Missing transaction IDs found in feature file.")

        if not ids.str.fullmatch(r"\d+").all():
            raise ValueError("Invalid transaction ID format found.")

        steps = pd.to_numeric(chunk.iloc[:, 1], errors="coerce")

        numeric_features = chunk.iloc[:, 1:].apply(
            pd.to_numeric, errors="coerce"
        )

        invalid_value_count += int(numeric_features.isna().sum().sum())

        invalid_steps = steps.isna() | ~steps.between(1, EXPECTED_TIME_STEPS)

        invalid_step_count += int(invalid_steps.sum())

        for tx_id, step in zip(ids, steps):
            if tx_id in transaction_ids:
                duplicate_count += 1
            else:
                transaction_ids.add(tx_id)

            if pd.notna(step):
                transaction_steps[tx_id] = int(step)

        valid_steps = steps[~invalid_steps].astype(int)
        counts = valid_steps.value_counts()

        for step, count in counts.items():
            step_counts[int(step)] += int(count)

        total_rows += len(chunk)

    print(f"Total feature records: {total_rows:,}")
    print(f"Unique transaction IDs: {len(transaction_ids):,}")
    print(f"Duplicate transaction IDs: {duplicate_count:,}")
    print(f"Missing/invalid numerical values: {invalid_value_count:,}")
    print(f"Invalid time-step records: {invalid_step_count:,}")

    print("\nTransaction distribution by time step:")

    for step, count in step_counts.items():
        print(f"Step {step:02d}: {count:,}")

    return transaction_ids, transaction_steps


# --------------------------------------------------
# CLASS VALIDATION
# --------------------------------------------------

def validate_classes(feature_ids):
    print("\n2. CLASS DATA VALIDATION")

    classes = pd.read_csv(
        CLASSES_FILE,
        dtype={"txId": "string", "class": "string"},
    )

    if list(classes.columns) != ["txId", "class"]:
        raise ValueError("Unexpected class file columns.")

    if classes["txId"].isna().any():
        raise ValueError("Missing transaction IDs in class file.")

    if classes["class"].isna().any():
        raise ValueError("Missing class labels found.")

    duplicate_ids = int(classes["txId"].duplicated().sum())

    class_distribution = classes["class"].value_counts(dropna=False)

    class_ids = set(classes["txId"])

    missing_in_classes = feature_ids - class_ids
    missing_in_features = class_ids - feature_ids

    print(f"Total class records: {len(classes):,}")
    print(f"Duplicate class transaction IDs: {duplicate_ids:,}")

    print("\nClass distribution:")

    for label, count in class_distribution.items():
        print(f"{label}: {count:,}")

    print(f"\nFeature IDs missing in class file: {len(missing_in_classes):,}")
    print(f"Class IDs missing in feature file: {len(missing_in_features):,}")

    return class_ids


# --------------------------------------------------
# EDGE VALIDATION
# --------------------------------------------------

def validate_edges(feature_ids, transaction_steps):
    print("\n3. GRAPH EDGE VALIDATION")

    edge_count = 0
    invalid_source_count = 0
    invalid_target_count = 0
    cross_step_count = 0
    missing_step_count = 0
    duplicate_edge_count = 0

    seen_edges = set()

    for chunk in pd.read_csv(
        EDGES_FILE,
        dtype={"txId1": "string", "txId2": "string"},
        chunksize=CHUNK_SIZE,
    ):
        if list(chunk.columns) != ["txId1", "txId2"]:
            raise ValueError("Unexpected edge file columns.")

        if chunk[["txId1", "txId2"]].isna().any().any():
            raise ValueError("Missing transaction IDs found in edge file.")

        for source, target in zip(chunk["txId1"], chunk["txId2"]):
            edge_count += 1

            if source not in feature_ids:
                invalid_source_count += 1

            if target not in feature_ids:
                invalid_target_count += 1

            edge = (source, target)

            if edge in seen_edges:
                duplicate_edge_count += 1
            else:
                seen_edges.add(edge)

            source_step = transaction_steps.get(source)
            target_step = transaction_steps.get(target)

            if source_step is None or target_step is None:
                missing_step_count += 1
            elif source_step != target_step:
                cross_step_count += 1

    print(f"Total graph edges: {edge_count:,}")
    print(f"Invalid source transaction IDs: {invalid_source_count:,}")
    print(f"Invalid target transaction IDs: {invalid_target_count:,}")
    print(f"Duplicate directed edges: {duplicate_edge_count:,}")
    print(f"Edges with missing time-step information: {missing_step_count:,}")
    print(f"Edges connecting different time steps: {cross_step_count:,}")


# --------------------------------------------------
# COMPLETE DATASET VALIDATION
# --------------------------------------------------

def validate_dataset():
    print("\n" + "=" * 55)
    print("ELLIPTIC DATASET DEEP VALIDATION")
    print("=" * 55)

    for path in [FEATURES_FILE, EDGES_FILE, CLASSES_FILE]:
        validate_file_exists(path)

    feature_ids, transaction_steps = validate_features()

    class_ids = validate_classes(feature_ids)

    validate_edges(feature_ids, transaction_steps)

    print("\n" + "=" * 55)
    print("DATASET DEEP VALIDATION FINISHED")
    print("=" * 55)

    print(f"Feature transaction IDs: {len(feature_ids):,}")
    print(f"Class transaction IDs: {len(class_ids):,}")
    print(f"Time steps represented: {len(set(transaction_steps.values()))}")

    print("\nReview the counts above before using the dataset.")


if __name__ == "__main__":
    validate_dataset()
