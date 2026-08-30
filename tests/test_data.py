import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from data_prep import validate, split_and_scale  # noqa: E402
from generate_data import generate  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return generate(rows=2000, seed=1)


def test_generated_shape(df):
    assert len(df) == 2000
    assert "churn" in df.columns


def test_no_nulls(df):
    assert df.isnull().sum().sum() == 0


def test_churn_rate_is_realistic(df):
    rate = df["churn"].mean()
    assert 0.05 < rate < 0.6, f"churn rate {rate:.2%} out of expected range"


def test_validate_passes_on_good_data(df):
    validate(df)


def test_validate_catches_bad_target():
    bad = pd.DataFrame({"tenure_months": [1, 2], "churn": [0, 2]})
    with pytest.raises(AssertionError):
        validate(bad)


def test_split_and_scale_shapes(df):
    ds = split_and_scale(df, test_size=0.25, seed=1)
    assert len(ds.X_train) + len(ds.X_test) == len(df)
    assert list(ds.X_train.columns) == ds.feature_names
    assert abs(len(ds.X_test) / len(df) - 0.25) < 0.02
