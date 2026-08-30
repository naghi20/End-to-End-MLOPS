"""
tune.py
-------
Hyperparameter search with Optuna, logged to MLflow.

Usage:
    python src/tune.py --trials 20 --register
"""
import argparse
import os

import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from data_prep import load_raw, split_and_scale, validate

EXPERIMENT_NAME = "customer-churn-tuning"
MODEL_NAME = "churn-classifier"


def objective(trial, ds):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(random_state=42, class_weight="balanced", **params)
        model.fit(ds.X_train, ds.y_train)
        preds = model.predict(ds.X_test)
        score = f1_score(ds.y_test, preds)
        mlflow.log_params(params)
        mlflow.log_metric("f1", score)
        trial.set_user_attr("run_id", mlflow.active_run().info.run_id)
    return score


def main(args):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw(args.data)
    validate(df)
    ds = split_and_scale(df)

    with mlflow.start_run(run_name="optuna-search"):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, ds), n_trials=args.trials)

        mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})
        mlflow.log_metric("best_f1", study.best_value)
        best_run_id = study.best_trial.user_attrs["run_id"]
        print(f"Best trial f1={study.best_value:.4f} params={study.best_params}")
        print(f"Best run_id={best_run_id}")

        if args.register:
            model = RandomForestClassifier(
                random_state=42, class_weight="balanced", **study.best_params
            )
            model.fit(ds.X_train, ds.y_train)
            mlflow.sklearn.log_model(model, artifact_path="model")
            result = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model", MODEL_NAME
            )
            print(f"Registered {MODEL_NAME} version {result.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--register", action="store_true")
    main(parser.parse_args())
