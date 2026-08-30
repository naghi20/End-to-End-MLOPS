"""
data_prep.py
------------
Shared preprocessing so training and the serving API can never drift apart.
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "churn"


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: list


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in {path}")
    return df


def validate(df: pd.DataFrame) -> None:
    assert df.isnull().sum().sum() == 0, "Nulls found in dataset"
    assert (df["tenure_months"] >= 0).all(), "Negative tenure detected"
    assert df[TARGET].isin([0, 1]).all(), "Target must be binary"
    churn_rate = df[TARGET].mean()
    assert 0.05 < churn_rate < 0.6, f"Churn rate {churn_rate:.2%} looks anomalous"


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Dataset:
    feature_names = [c for c in df.columns if c != TARGET]
    X = df[feature_names]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    return Dataset(X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
