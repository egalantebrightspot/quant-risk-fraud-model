"""Gradient boosting risk model."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from src.config import RANDOM_STATE


class GBMRiskModel:
    """Gradient boosting model for PD / fraud probability (LightGBM)."""

    def __init__(self, random_state: int = RANDOM_STATE, **kwargs):
        import lightgbm as lgb
        self._estimator = lgb.LGBMClassifier(
            random_state=random_state,
            verbosity=-1,
            **kwargs,
        )
        self.feature_names_in_: Optional[list[str]] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "GBMRiskModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None
        y_flat = np.asarray(y).ravel()
        self._estimator.fit(X, y_flat)
        self.classes_ = self._estimator.classes_
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict_proba(X))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(X))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GBMRiskModel":
        return joblib.load(Path(path))
