"""Data preprocessing pipeline."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, cast


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    result = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return cast(Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], result)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fit_on: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features with StandardScaler (zero mean, unit variance).

    Fits on X_train (or fit_on if provided) and transforms both train and test.

    Returns:
        X_train_scaled, X_test_scaled, fitted scaler
    """
    scaler = StandardScaler()
    fit_data = fit_on if fit_on is not None else X_train
    scaler.fit(fit_data)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    return X_train_scaled, X_test_scaled, scaler
