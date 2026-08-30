"""
train.py
--------
Trains a churn classifier and logs everything to MLflow.

Local run (local sqlite tracking store, no server needed):
    python src/train.py

Pointed at the shared MLflow server on EC2:
    export MLFLOW_TRACKING_URI=http://<EC2_PUBLIC_IP>:5000
    python src/train.py --n-estimators 300 --max-depth 8 --register
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_prep import load_raw, split_and_scale, validate

EXPERIMENT_NAME = "customer-churn"
MODEL_NAME = "churn-classifier"


def train(args) -> dict:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw(args.data)
    validate(df)
    ds = split_and_scale(df, test_size=args.test_size, seed=args.seed)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            class_weight="balanced",
        )
        model.fit(ds.X_train, ds.y_train)
        preds = model.predict(ds.X_test)
        proba = model.predict_proba(ds.X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(ds.y_test, preds),
            "precision": precision_score(ds.y_test, preds),
            "recall": recall_score(ds.y_test, preds),
            "f1": f1_score(ds.y_test, preds),
            "roc_auc": roc_auc_score(ds.y_test, proba),
        }

        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "test_size": args.test_size,
                "seed": args.seed,
                "rows": len(df),
            }
        )
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(ds.y_test, preds)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for (i, j), v in __import__("numpy").ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        mlflow.sklearn.log_model(model, artifact_path="model")

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.joblib")
        joblib.dump(ds.scaler, "artifacts/scaler.joblib")
        with open("artifacts/feature_names.json", "w") as f:
            json.dump(ds.feature_names, f)
        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact("artifacts/scaler.joblib")
        mlflow.log_artifact("artifacts/feature_names.json")

        print(f"run_id={run.info.run_id}")
        print(json.dumps(metrics, indent=2))

        if args.register:
            result = mlflow.register_model(
                f"runs:/{run.info.run_id}/model", MODEL_NAME
            )
            print(f"Registered {MODEL_NAME} version {result.version}")

        return {"run_id": run.info.run_id, **metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--register", action="store_true")
    train(parser.parse_args())
