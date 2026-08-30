"""
evaluate.py
-----------
The CI/CD "quality gate": fails the pipeline if the model doesn't clear
a minimum bar.

Usage:
    python src/evaluate.py --metrics artifacts/metrics.json --min-f1 0.70 --min-auc 0.75
"""
import argparse
import json
import sys


def main(args):
    with open(args.metrics) as f:
        metrics = json.load(f)

    print("Evaluating model against quality gate:")
    print(json.dumps(metrics, indent=2))

    failures = []
    if metrics.get("f1", 0) < args.min_f1:
        failures.append(f"f1 {metrics.get('f1'):.4f} < required {args.min_f1}")
    if metrics.get("roc_auc", 0) < args.min_auc:
        failures.append(f"roc_auc {metrics.get('roc_auc'):.4f} < required {args.min_auc}")

    if failures:
        print("QUALITY GATE FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("Quality gate passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    parser.add_argument("--min-f1", type=float, default=0.70)
    parser.add_argument("--min-auc", type=float, default=0.75)
    main(parser.parse_args())
