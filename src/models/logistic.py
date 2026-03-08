"""Logistic regression risk model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Optional

from src.config import (
    RANDOM_STATE,
    LOGISTIC_C,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
)


class LogisticRiskModel:
    """Baseline logistic regression for PD / fraud probability estimation.

    Wraps sklearn LogisticRegression with a consistent interface: fit, predict_proba,
    predict. Stores feature names for interpretability and API use.
    """

    def __init__(
        self,
        C: float = LOGISTIC_C,
        max_iter: int = LOGISTIC_MAX_ITER,
        solver: str = LOGISTIC_SOLVER,
        random_state: int = RANDOM_STATE,
    ):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self._estimator = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )
        self.feature_names_in_: Optional[list[str]] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "LogisticRiskModel":
        """Fit the logistic regression model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None
        y_flat = np.asarray(y).ravel()
        self._estimator.fit(X, y_flat)
        self.classes_ = self._estimator.classes_
        return self

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Predict class probabilities. Returns array of shape (n_samples, 2) [P(0), P(1)]."""
        return self._estimator.predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        return self._estimator.predict(X)

    def get_coefficients(self) -> pd.Series:
        """Return feature coefficients (index = feature names if available)."""
        coef = self._estimator.coef_.ravel()
        if self.feature_names_in_ is not None:
            return pd.Series(coef, index=self.feature_names_in_)
        return pd.Series(coef, index=[f"X{i}" for i in range(len(coef))])
