
from pathlib import Path

import joblib
import torch

BASE_DIR = Path("data/processed/elliptic")

FEATURES_FILE = BASE_DIR / "scaled_features.pt"
SCALER_FILE = BASE_DIR / "feature_scaler.joblib"
GRAPH_FILE = BASE_DIR / "transaction_graph.pt"


def verify_preprocessing():
    print("\nPREPROCESSING INTEGRITY CHECKS")
    print("=" * 55)

    scaled = torch.load(FEATURES_FILE, weights_only=True)
    scaler = joblib.load(SCALER_FILE)
    graph = torch.load(GRAPH_FILE, weights_only=True)

    train_mask = graph["train_mask"]
    validation_targets = (
        (graph["time_step"] >= 30) & (graph["time_step"] <= 34)
    )
    test_targets = graph["time_step"] >= 35

    assert scaled.shape == graph["x"].shape
    print("[PASS] Feature dimensions match the original graph.")

    assert torch.isfinite(scaled).all()
    print("[PASS] No NaN or infinite scaled values.")

    assert scaler.n_features_in_ == 165
    print("[PASS] Scaler contains all 165 features.")

    assert int(scaler.n_samples_seen_) == int(train_mask.sum())
    print("[PASS] Scaler was fitted on training transactions only.")

    train_scaled = scaled[train_mask]
    means = train_scaled.mean(dim=0)
    stds = train_scaled.std(dim=0)

    assert torch.allclose(
        means,
        torch.zeros_like(means),
        atol=1e-4,
    )
    assert torch.allclose(
        stds,
        torch.ones_like(stds),
        atol=1e-3,
    )
    print("[PASS] Training features have mean 0 and standard deviation 1.")

    assert validation_targets.sum().item() == 15461
    assert test_targets.sum().item() == 67504
    print("[PASS] Validation and test target periods are correct.")

    print("\nFINAL SUMMARY")
    print("-" * 55)
    print(f"Scaled feature shape: {tuple(scaled.shape)}")
    print(f"Training targets: {int(train_mask.sum()):,}")
    print(f"Validation targets: {int(validation_targets.sum()):,}")
    print(f"Test targets: {int(test_targets.sum()):,}")

    print("\nALL PREPROCESSING CHECKS PASSED.")


if __name__ == "__main__":
    verify_preprocessing()
