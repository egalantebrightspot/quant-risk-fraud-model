"""Tests for risk models."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Project root on path for src imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.logistic import LogisticRiskModel
from src.models.train_logistic import run_training
from src.data.loaders import generate_and_save
from src.data.schema import TARGET_FRAUD


def test_logistic_risk_model_fit_predict():
    """LogisticRiskModel fits and returns correct shapes."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 5), columns=[f"X{i}" for i in range(5)])
    y = pd.Series((X["X0"] + X["X1"] > 0).astype(int))

    model = LogisticRiskModel()
    model.fit(X, y)

    assert model.feature_names_in_ == ["X0", "X1", "X2", "X3", "X4"]
    assert model.classes_ is not None

    proba = model.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    pred = model.predict(X)
    assert pred.shape == (200,)
    assert set(pred) <= {0, 1}


def test_logistic_risk_model_coefficients():
    """get_coefficients returns a Series with feature names."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
    y = pd.Series(np.random.randint(0, 2, 100))

    model = LogisticRiskModel()
    model.fit(X, y)
    coef = model.get_coefficients()
    assert isinstance(coef, pd.Series)
    assert list(coef.index) == ["a", "b", "c"]
    assert len(coef) == 3


def test_train_logistic_run_training(tmp_path):
    """run_training loads CSV, trains, evaluates, saves artifact and metrics."""
    data_path = tmp_path / "train.csv"
    generate_and_save(path=data_path, n=500, fraud_rate=0.08, random_state=42)
    out_path = tmp_path / "model.joblib"
    result = run_training(
        data_path=data_path,
        target_column=TARGET_FRAUD,
        output_path=out_path,
        test_size=0.25,
        validation_size=0.2,
        random_state=42,
    )
    assert "summary_uncal" in result
    assert "summary_cal" in result
    assert "artifact_path" in result
    assert "metrics_path" in result
    assert result["artifact_path"].exists()
    assert result["metrics_path"].exists()
    assert "auc" in result["summary_uncal"]
    assert "ks" in result["summary_uncal"]
