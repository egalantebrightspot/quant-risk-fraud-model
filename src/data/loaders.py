"""Data loading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from typing import Tuple, Optional, cast, Union

from src.config import DATA_DIR, RANDOM_STATE
from src.data.schema import (
    CORE_FEATURES,
    DEFAULT_TARGET,
    AUX_TARGETS,
    get_feature_names,
    get_target_name,
)
from src.data.synthetic_generator import SyntheticRiskDataGenerator


def load_csv(
    path: Path,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a CSV with features and a binary target.

    Args:
        path: Path to CSV file.
        target_column: Name of the binary target column (0/1 or similar).
        feature_columns: Optional list of feature column names. If None, all columns
            except target_column are used.

    Returns:
        (X_df, y_series) as DataFrame and Series.
    """
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in {list(df.columns)}")
    y = df[target_column]
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]
    X = df[feature_columns]
    return cast(Tuple[pd.DataFrame, pd.Series], (X, y))


def load_synthetic(
    n_samples: int = 10_000,
    n_features: int = 20,
    n_informative: int = 10,
    n_redundant: int = 5,
    weights: Optional[tuple[float, float]] = (0.9, 0.1),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic binary classification data for PD/fraud-like modeling.

    Useful for development and CI when no real dataset is available.
    Target is 1 = default/fraud, 0 = non-default/non-fraud.

    Args:
        n_samples: Number of samples.
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features.
        weights: Class weights (neg, pos). Default (0.9, 0.1) gives ~10% positive.
        random_state: Random seed.

    Returns:
        (X_df, y_series) with column names X0, X1, ... and target 'target'.
    """
    w = (0.9, 0.1) if weights is None else list(weights)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        weights=w,
        random_state=random_state,
        flip_y=0.02,
    )
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    feature_names = [f"X{i}" for i in range(X_arr.shape[1])]
    X_df = pd.DataFrame(X_arr, columns=feature_names)
    y_series = pd.Series(y_arr, name="target")
    return X_df, y_series


def generate_synthetic_risk_data(
    n_samples: int = 10_000,
    positive_fraction: Optional[float] = 0.03,
    random_state: int = 42,
    include_aux_targets: bool = False,
    use_default_label: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """Generate synthetic risk/fraud data with realistic distributions and label logic.

    Features follow the training data schema (credit/payments/fraud blueprint).
    The binary target is produced from a latent risk score:
      risk_score = β0 + Σ βi*xi + ε,  p = σ(risk_score),  y ~ Bernoulli(p)
    giving smooth probability gradients and realistic ROC/AUC behavior.

    Args:
        n_samples: Number of records.
        positive_fraction: Approximate fraction of positive class (e.g. 0.01–0.05).
                          If None, probabilities are not rescaled (raw sigmoid).
        random_state: Random seed.
        include_aux_targets: If True, return a third value (DataFrame) with LGD and EAD.
        use_default_label: If True, target column is 'default_flag', else 'fraud_flag'.

    Returns:
        (X_df, y_series, aux_df). aux_df is None unless include_aux_targets=True.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples
    names = get_feature_names()
    target_name = get_target_name(use_default_label)

    # ---- Distributions (realistic, correlated where appropriate) ----
    # age: continuous, mildly right-skewed (e.g. 22–70)
    age = 22 + rng.gamma(shape=8, scale=4, size=n)
    age = np.clip(age, 22, 75).astype(np.float64)

    # income: log-normal (thousands)
    log_income = 10.5 + 0.8 * rng.standard_normal(n)
    income = np.exp(log_income)
    income = np.clip(income, 15, 500).astype(np.float64)

    # utilization: ratio [0, 1], beta
    utilization = rng.beta(2, 5, size=n).astype(np.float64)

    # num_trades: count
    num_trades = rng.poisson(8, size=n).astype(np.float64)
    num_trades = np.clip(num_trades, 0, 30)

    # delinq_30d: binary; higher when utilization is high
    p_delinq = 0.05 + 0.25 * utilization + 0.02 * rng.standard_normal(n)
    p_delinq = np.clip(p_delinq, 0, 1)
    delinq_30d = (rng.uniform(size=n) < p_delinq).astype(np.float64)

    # credit_history_length: years, correlated with age
    credit_history_length = np.maximum(
        0,
        0.25 * (age - 22) + 2 * rng.standard_normal(n),
    ).astype(np.float64)
    credit_history_length = np.clip(credit_history_length, 0, 40)

    # transaction_amount: right-skewed, heavy-tail (lognormal)
    transaction_amount = rng.lognormal(mean=5, sigma=1.2, size=n).astype(np.float64)
    transaction_amount = np.clip(transaction_amount, 1, 500)

    # merchant_risk_score: ordinal 0..4
    merchant_risk_score = rng.integers(0, 5, size=n).astype(np.float64)

    # device_trust_score: continuous [0, 1]
    device_trust_score = rng.beta(2, 2, size=n).astype(np.float64)

    # velocity_score: higher = more rapid activity (e.g. 0–10)
    velocity_score = rng.gamma(shape=2, scale=1.2, size=n).astype(np.float64)
    velocity_score = np.clip(velocity_score, 0, 15)

    # Build feature matrix (order = CORE_FEATURES)
    X = np.column_stack([
        age,
        income,
        utilization,
        num_trades,
        delinq_30d,
        credit_history_length,
        transaction_amount,
        merchant_risk_score,
        device_trust_score,
        velocity_score,
    ])

    # ---- Latent risk score: β0 + Σ βi*xi + ε, then p = σ(risk_score) ----
    # Scale features to similar scale for stable coefficients
    age_s = (age - 45) / 15
    income_s = (np.log(income + 1) - 11) / 1.5
    util_s = (utilization - 0.3) / 0.25
    num_trades_s = (num_trades - 8) / 5
    chl_s = (credit_history_length - 10) / 8
    amt_s = (np.log(transaction_amount + 1) - 5) / 1.2
    merch_s = (merchant_risk_score - 2) / 1.5
    device_s = (device_trust_score - 0.5) / 0.3
    vel_s = (velocity_score - 2.5) / 2

    # Risk increases with: utilization, delinq, amount, merchant_risk, velocity
    # Risk decreases with: income, device_trust
    risk_score = (
        -2.5
        + 0.4 * util_s
        + 0.8 * np.asarray(delinq_30d)
        + 0.3 * amt_s
        + 0.35 * merch_s
        - 0.4 * device_s
        + 0.45 * vel_s
        + 0.15 * age_s
        - 0.2 * income_s
        + 0.1 * num_trades_s
        + 0.05 * chl_s
        + 0.5 * rng.standard_normal(n)
    )

    proba = 1.0 / (1.0 + np.exp(-risk_score))

    # Optional rescaling to hit target positive fraction (rare-event)
    if positive_fraction is not None:
        from scipy import special
        # Shift so median predicted prob is near positive_fraction
        q = float(np.quantile(proba, 1 - positive_fraction))
        shift = special.logit(np.clip(positive_fraction, 1e-5, 1 - 1e-5)) - special.logit(np.clip(q, 1e-5, 1 - 1e-5))
        proba = 1.0 / (1.0 + np.exp(-(risk_score + shift)))
        proba = np.clip(proba, 1e-5, 1 - 1e-5)

    y = (rng.uniform(size=n) < proba).astype(np.int64)

    # Optional label noise
    flip = rng.uniform(size=n) < 0.02
    y[flip] = 1 - y[flip]

    X_df = pd.DataFrame(X, columns=names)
    y_series = pd.Series(y, name=target_name)

    aux_df = None
    if include_aux_targets:
        lgd = np.clip(0.2 + 0.5 * rng.beta(2, 3, size=n), 0, 1).astype(np.float64)
        ead = np.clip(utilization * (1 + 0.3 * rng.standard_normal(n)), 0, 2).astype(np.float64)
        aux_df = pd.DataFrame({
            AUX_TARGETS[0]: lgd,
            AUX_TARGETS[1]: ead,
        })

    return (X_df, y_series, aux_df)


def generate_and_save(
    path: Optional[Union[str, Path]] = None,
    n: int = 50_000,
    fraud_rate: float = 0.03,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate synthetic risk data and save to CSV.

    Args:
        path: Output path for CSV. Defaults to data/training_data.csv under project root.
        n: Number of samples to generate.
        fraud_rate: Target fraud rate (e.g. 0.03 for 3%).
        random_state: Random seed for reproducibility.

    Returns:
        The generated DataFrame (same as would be read from the CSV).
    """
    out_path = Path(path) if path is not None else DATA_DIR / "training_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen = SyntheticRiskDataGenerator(
        n_samples=n,
        fraud_rate=fraud_rate,
        random_state=random_state,
    )
    df = gen.generate()
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    """Generate synthetic training data. Run from project root: python -m src.data.loaders"""
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    df = generate_and_save()
    print(f"Generated {len(df):,} rows -> {DATA_DIR / 'training_data.csv'}")
