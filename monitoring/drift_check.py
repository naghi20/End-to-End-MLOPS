"""
drift_check.py
---------------
Simple statistical drift detector: compares feature mean/std of a new
batch against the training-time baseline.

Usage:
    python monitoring/drift_check.py --mode baseline --data data/churn.csv --out monitoring/baseline.json
    python monitoring/drift_check.py --mode check --data data/new_batch.csv --baseline monitoring/baseline.json
"""
import argparse
import json
import sys

import pandas as pd

FEATURES = [
    "tenure_months", "monthly_charges", "total_charges", "support_tickets",
    "num_products", "contract_score", "usage_score", "satisfaction_score",
]


def build_baseline(df: pd.DataFrame) -> dict:
    return {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in FEATURES if col in df.columns
    }


def check_drift(df: pd.DataFrame, baseline: dict, z_threshold: float = 2.0) -> list:
    alerts = []
    for col, stats in baseline.items():
        if col not in df.columns or stats["std"] == 0:
            continue
        current_mean = df[col].mean()
        z = abs(current_mean - stats["mean"]) / stats["std"]
        if z > z_threshold:
            alerts.append({
                "feature": col,
                "baseline_mean": stats["mean"],
                "current_mean": current_mean,
                "z_shift": round(z, 2),
            })
    return alerts


def main(args):
    df = pd.read_csv(args.data)

    if args.mode == "baseline":
        baseline = build_baseline(df)
        with open(args.out, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline written to {args.out}")
        return

    with open(args.baseline) as f:
        baseline = json.load(f)

    alerts = check_drift(df, baseline, z_threshold=args.z_threshold)
    if alerts:
        print("DRIFT DETECTED:")
        print(json.dumps(alerts, indent=2))
        sys.exit(1)

    print("No significant drift detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "check"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="monitoring/baseline.json")
    parser.add_argument("--baseline", default="monitoring/baseline.json")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    main(parser.parse_args())
