import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from data_prep import split_and_scale
from generate_data import generate


def test_model_beats_naive_baseline():
    df = generate(rows=3000, seed=7)
    ds = split_and_scale(df, seed=7)

    majority_class = ds.y_train.mode()[0]
    naive_preds = [majority_class] * len(ds.y_test)
    naive_f1 = f1_score(ds.y_test, naive_preds, zero_division=0)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=7,
                                    class_weight="balanced")
    model.fit(ds.X_train, ds.y_train)
    preds = model.predict(ds.X_test)
    model_f1 = f1_score(ds.y_test, preds)

    assert model_f1 > naive_f1 + 0.15


def test_model_is_deterministic_given_seed():
    df = generate(rows=1500, seed=3)
    ds = split_and_scale(df, seed=3)

    m1 = RandomForestClassifier(n_estimators=50, random_state=3).fit(ds.X_train, ds.y_train)
    m2 = RandomForestClassifier(n_estimators=50, random_state=3).fit(ds.X_train, ds.y_train)

    assert (m1.predict(ds.X_test) == m2.predict(ds.X_test)).all()
