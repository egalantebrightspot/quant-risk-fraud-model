"""Tests for training data schema and synthetic risk data generator."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import (
    CORE_FEATURES,
    TARGET_FRAUD,
    TARGET_DEFAULT,
    AUX_TARGETS,
    get_feature_names,
    get_target_name,
)
from src.data.loaders import generate_synthetic_risk_data, generate_and_save
from src.data.synthetic_generator import SyntheticRiskDataGenerator


def test_schema_feature_names():
    assert len(CORE_FEATURES) == 10
    assert "age" in CORE_FEATURES
    assert "fraud_flag" == TARGET_FRAUD
    assert get_feature_names() == CORE_FEATURES
    assert get_target_name(use_default_label=False) == TARGET_FRAUD
    assert get_target_name(use_default_label=True) == TARGET_DEFAULT
    assert AUX_TARGETS == ["loss_given_default", "exposure_at_default"]


def test_generate_synthetic_risk_data_shape_and_columns():
    X, y, aux = generate_synthetic_risk_data(
        n_samples=500,
        positive_fraction=0.05,
        random_state=42,
        include_aux_targets=False,
    )
    assert X.shape == (500, 10)
    assert list(X.columns) == CORE_FEATURES
    assert y.name == TARGET_FRAUD
    assert y.shape == (500,)
    assert set(y) <= {0, 1}
    assert aux is None


def test_generate_synthetic_risk_data_with_aux():
    X, y, aux = generate_synthetic_risk_data(
        n_samples=300,
        positive_fraction=0.03,
        random_state=123,
        include_aux_targets=True,
    )
    assert aux is not None
    assert list(aux.columns) == AUX_TARGETS
    assert aux.shape[0] == 300


def test_generate_synthetic_risk_data_default_label():
    X, y, _ = generate_synthetic_risk_data(
        n_samples=200,
        random_state=0,
        use_default_label=True,
    )
    assert y.name == TARGET_DEFAULT


def test_generate_synthetic_risk_data_positive_rate():
    _, y, _ = generate_synthetic_risk_data(
        n_samples=5_000,
        positive_fraction=0.04,
        random_state=99,
    )
    rate = y.mean()
    assert 0.02 <= rate <= 0.08  # rough band around 4%


def test_synthetic_risk_data_generator_generate():
    gen = SyntheticRiskDataGenerator(n_samples=500, fraud_rate=0.05, random_state=42)
    df = gen.generate()
    assert df.shape == (500, 11)
    assert list(df.columns) == CORE_FEATURES + [TARGET_FRAUD]
    assert df[TARGET_FRAUD].isin([0, 1]).all()
    assert 0.02 <= df[TARGET_FRAUD].mean() <= 0.12  # rough band around 5%


def test_synthetic_risk_data_generator_generate_X_y():
    gen = SyntheticRiskDataGenerator(n_samples=300, fraud_rate=0.03, random_state=0)
    X, y = gen.generate_X_y()
    assert X.shape == (300, 10)
    assert list(X.columns) == CORE_FEATURES
    assert y.name == TARGET_FRAUD
    assert len(y) == 300


def test_generate_and_save(tmp_path):
    path = tmp_path / "training_data.csv"
    df = generate_and_save(path=str(path), n=200, fraud_rate=0.04, random_state=1)
    assert df.shape == (200, 11)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == 200
