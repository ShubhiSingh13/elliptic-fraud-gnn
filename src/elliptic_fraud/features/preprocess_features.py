
from pathlib import Path

import joblib
import torch
from sklearn.preprocessing import StandardScaler

GRAPH_FILE = Path("data/processed/elliptic/transaction_graph.pt")
OUTPUT_DIR = Path("data/processed/elliptic")
FEATURES_OUTPUT = OUTPUT_DIR / "scaled_features.pt"
SCALER_OUTPUT = OUTPUT_DIR / "feature_scaler.joblib"


def preprocess_features():
    print("\nELLIPTIC FEATURE PREPROCESSING")
    print("=" * 55)

    graph = torch.load(GRAPH_FILE, weights_only=True)

    x = graph["x"].numpy()
    train_mask = graph["train_mask"].numpy()

    train_features = x[train_mask]

    print(f"Total transactions: {len(x):,}")
    print(f"Training transactions: {len(train_features):,}")
    print(f"Total features: {x.shape[1]}")

    # Fit the scaler exclusively on training transactions.
    scaler = StandardScaler()
    scaler.fit(train_features)

    # Apply the same fitted scaler to all transactions.
    scaled_features = scaler.transform(x)

    scaled_features = torch.tensor(
        scaled_features,
        dtype=torch.float32,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(scaled_features, FEATURES_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)

    # Verify training-only scaling.
    scaled_train = scaled_features[graph["train_mask"]]

    train_means = scaled_train.mean(dim=0)
    train_stds = scaled_train.std(dim=0)

    print("\nPREPROCESSING VERIFICATION")
    print("-" * 55)
    print(f"Scaled feature shape: {tuple(scaled_features.shape)}")
    print(f"Average training feature mean: {train_means.mean().item():.6f}")
    print(f"Average training feature std: {train_stds.mean().item():.6f}")

    print("\nSaved files:")
    print(FEATURES_OUTPUT)
    print(SCALER_OUTPUT)

    print("\nFeature preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_features()
