
from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = Path("data/interim/elliptic/transactions.parquet")
OUTPUT_FILE = Path("data/processed/elliptic/feature_profile.csv")

FEATURE_COLUMNS = [
    f"feature_{i:03d}" for i in range(1, 166)
]


def inspect_features():
    print("\nELLIPTIC FEATURE INSPECTION")
    print("=" * 55)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_parquet(DATA_FILE)

    features = df[FEATURE_COLUMNS]

    # Check for invalid numerical values.
    values = features.to_numpy(dtype=np.float64)

    nan_count = int(np.isnan(values).sum())
    infinite_count = int(np.isinf(values).sum())

    print(f"Transactions: {len(df):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Missing values: {nan_count:,}")
    print(f"Infinite values: {infinite_count:,}")

    # Generate descriptive statistics.
    profile = features.describe().T
    profile["unique_values"] = features.nunique()
    profile["is_constant"] = profile["unique_values"] <= 1
    profile["skewness"] = features.skew()
    profile["q01"] = features.quantile(0.01)
    profile["q50"] = features.quantile(0.50)
    profile["q99"] = features.quantile(0.99)
    profile["q999"] = features.quantile(0.999)

    # Identify feature groups.
    profile["feature_group"] = [
        "local" if int(name.split("_")[1]) <= 93 else "aggregated"
        for name in profile.index
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUTPUT_FILE)

    print("\nFEATURE RANGE SUMMARY")
    print("-" * 55)

    print(f"Overall minimum: {values.min():.6f}")
    print(f"Overall maximum: {values.max():.6f}")

    print(f"Constant features: {profile['is_constant'].sum()}")

    print("\nLOCAL FEATURES (first 93)")
    print("-" * 55)
    print(
        profile[profile["feature_group"] == "local"][
            ["min", "max", "mean", "std"]
        ].round(4).to_string()
    )

    print("\nAGGREGATED FEATURES (last 72)")
    print("-" * 55)
    print(
        profile[profile["feature_group"] == "aggregated"][
            ["min", "max", "mean", "std"]
        ].round(4).to_string()
    )
    
    print("\nTOP 15 FEATURES BY ABSOLUTE SKEWNESS")
    print("-" * 55)

    profile["absolute_skewness"] = profile["skewness"].abs()

    print(
        profile.nlargest(15, "absolute_skewness")[
            ["skewness", "min", "q01", "q50", "q99", "q999", "max"]
        ].round(4).to_string()
    )

    print("\nTOP 15 FEATURES BY MAXIMUM VALUE")
    print("-" * 55)

    print(
        profile.nlargest(15, "max")[
            ["min", "q01", "q50", "q99", "q999", "max"]
        ].round(4).to_string()
    )

    print(f"\nFull feature profile saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    inspect_features()
