"""Synthetic risk/fraud data generator (class-based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import cast

from src.data.schema import CORE_FEATURES, TARGET_FRAUD


class SyntheticRiskDataGenerator:
    """Generate synthetic risk/fraud data with configurable size and fraud rate.

    Features match the training data schema. Labels come from a latent risk score
    passed through sigmoid and rescaled to the target fraud rate.
    """

    def __init__(
        self,
        n_samples: int = 50_000,
        fraud_rate: float = 0.03,
        random_state: int = 42,
    ):
        self.n_samples = n_samples
        self.fraud_rate = fraud_rate
        self.rng = np.random.default_rng(random_state)

    def generate(self) -> pd.DataFrame:
        """Generate a DataFrame of features and fraud_flag."""
        n = self.n_samples
        rng = self.rng

        age = rng.normal(42, 12, n).clip(18, 85)
        income = np.exp(rng.normal(10.5, 0.5, n))  # log-normal
        utilization = rng.beta(2, 5, n)
        num_trades = rng.poisson(8, n)
        delinq_30d = rng.binomial(1, 0.08, n)
        credit_history_length = rng.normal(12, 6, n).clip(0, 40)

        transaction_amount = np.exp(rng.normal(4, 1.2, n))
        merchant_risk_score = rng.integers(1, 6, n)
        device_trust_score = rng.beta(5, 2, n)
        velocity_score = rng.gamma(2, 1.5, n)

        # Latent fraud risk score
        risk_score = (
            -3.5
            + 0.015 * utilization * 100
            + 0.8 * delinq_30d
            + 0.002 * transaction_amount
            + 0.4 * (merchant_risk_score - 3)
            - 1.2 * device_trust_score
            + 0.03 * velocity_score
            + rng.normal(0, 1, n)
        )

        # Convert to probability
        p = 1.0 / (1.0 + np.exp(-risk_score))

        # Adjust to target fraud rate
        p_mean = p.mean()
        if p_mean > 0:
            p = p * (self.fraud_rate / p_mean)
        p = np.clip(p, 0.0, 1.0)

        fraud_flag = rng.binomial(1, p)

        df = pd.DataFrame({
            "age": age,
            "income": income,
            "utilization": utilization,
            "num_trades": num_trades,
            "delinq_30d": delinq_30d,
            "credit_history_length": credit_history_length,
            "transaction_amount": transaction_amount,
            "merchant_risk_score": merchant_risk_score,
            "device_trust_score": device_trust_score,
            "velocity_score": velocity_score,
            TARGET_FRAUD: fraud_flag,
        })
        return df

    def generate_X_y(self) -> tuple[pd.DataFrame, pd.Series]:
        """Generate (X, y) for modeling: features and target only."""
        df = self.generate()
        X = df[CORE_FEATURES]
        y = df[TARGET_FRAUD]
        return cast(tuple[pd.DataFrame, pd.Series], (X, y))
