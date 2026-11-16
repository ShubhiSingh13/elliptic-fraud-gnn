
from pathlib import Path

import torch

GRAPH_FILE = Path("data/processed/elliptic/transaction_graph.pt")


def inspect_train_scaling():
    print("\nTRAINING-ONLY FEATURE STATISTICS")
    print("=" * 55)

    graph = torch.load(GRAPH_FILE, weights_only=True)

    x = graph["x"]
    train_mask = graph["time_step"] <= 29

    train_features = x[train_mask]

    means = train_features.mean(dim=0)
    stds = train_features.std(dim=0)

    print(f"Training transactions: {train_features.shape[0]:,}")
    print(f"Number of features: {train_features.shape[1]}")

    print("\nTRAINING DISTRIBUTION")
    print("-" * 55)
    print(f"Mean across feature means: {means.mean().item():.6f}")
    print(f"Mean across feature stds:  {stds.mean().item():.6f}")
    print(f"Lowest feature mean:       {means.min().item():.6f}")
    print(f"Highest feature mean:      {means.max().item():.6f}")
    print(f"Lowest feature std:        {stds.min().item():.6f}")
    print(f"Highest feature std:       {stds.max().item():.6f}")

    near_standardized = (
        (means.abs() < 0.1) & ((stds - 1).abs() < 0.1)
    )

    print(
        f"\nFeatures approximately standardized: "
        f"{near_standardized.sum().item()}/165"
    )

    print("\nTOP 10 FEATURES WITH LARGEST TRAINING MEANS")
    print("-" * 55)

    largest = torch.argsort(means.abs(), descending=True)[:10]

    for index in largest:
        print(
            f"feature_{index.item() + 1:03d}: "
            f"mean={means[index].item():.4f}, "
            f"std={stds[index].item():.4f}"
        )


if __name__ == "__main__":
    inspect_train_scaling()
