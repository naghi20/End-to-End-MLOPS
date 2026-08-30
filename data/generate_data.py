"""
generate_data.py
-----------------
Creates a synthetic "customer churn" dataset so the lab works with zero
external downloads / API keys.

Usage:
    python data/generate_data.py --out data/churn.csv --rows 5000
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


FEATURE_NAMES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "num_products",
    "contract_score",
    "usage_score",
    "satisfaction_score",
]


def generate(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=rows,
        n_features=len(FEATURE_NAMES),
        n_informative=6,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.73, 0.27],
        flip_y=0.02,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    df["tenure_months"] = np.clip((df["tenure_months"] * 10 + 24), 0, 72).round(0)
    df["monthly_charges"] = np.clip((df["monthly_charges"] * 25 + 70), 10, 200).round(2)
    df["total_charges"] = (df["monthly_charges"] * df["tenure_months"]).round(2)
    df["support_tickets"] = np.clip((df["support_tickets"] * 2 + 3), 0, 15).round(0)
    df["num_products"] = np.clip((df["num_products"] * 1.5 + 3), 1, 8).round(0)

    df["churn"] = y
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/churn.csv")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")
