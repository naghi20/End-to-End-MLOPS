"""
pipeline.py
-----------
Chains data generation -> train/tune -> quality gate into one command.

Usage:
    python orchestration/pipeline.py --tune --trials 15 --register
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd=ROOT):
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Step failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main(args):
    run([sys.executable, "data/generate_data.py", "--out", "data/churn.csv", "--rows", str(args.rows)])
    if args.dvc:
        run(["dvc", "add", "data/churn.csv"])
        run(["dvc", "push"])

    if args.tune:
        cmd = [sys.executable, "tune.py", "--data", "../data/churn.csv",
               "--trials", str(args.trials)]
        if args.register:
            cmd.append("--register")
        run(cmd, cwd=ROOT / "src")
    else:
        cmd = [sys.executable, "train.py", "--data", "../data/churn.csv"]
        if args.register:
            cmd.append("--register")
        run(cmd, cwd=ROOT / "src")

    if not args.tune:
        run([sys.executable, "evaluate.py",
             "--metrics", "artifacts/metrics.json",
             "--min-f1", str(args.min_f1), "--min-auc", str(args.min_auc)],
            cwd=ROOT / "src")

    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--dvc", action="store_true")
    parser.add_argument("--min-f1", type=float, default=0.70)
    parser.add_argument("--min-auc", type=float, default=0.75)
    main(parser.parse_args())
