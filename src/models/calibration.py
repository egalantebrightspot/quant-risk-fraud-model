"""Probability calibration for risk models."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattLR
from typing import Literal

from src.config import CALIBRATION_METHOD, RANDOM_STATE


def calibrate_proba(
    uncalibrated_proba: np.ndarray,
    y_true: np.ndarray,
    method: Literal["isotonic", "sigmoid"] = CALIBRATION_METHOD,
) -> np.ndarray:
    """Calibrate probability estimates using a hold-out set.

    Uses sklearn's calibration approach: fit a regressor (isotonic or Platt/sigmoid)
    on (uncalibrated_proba, y_true) and return recalibrated probabilities.

    Args:
        uncalibrated_proba: 1d array of predicted P(positive class).
        y_true: 1d array of true labels.
        method: 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling).

    Returns:
        1d array of calibrated probabilities, same shape as uncalibrated_proba.
    """
    proba = np.asarray(uncalibrated_proba).ravel()
    y = np.asarray(y_true).ravel()

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(proba, y)
        return cal.transform(proba)
    elif method == "sigmoid":
        platt = PlattLR(random_state=RANDOM_STATE, max_iter=1000)
        platt.fit(proba.reshape(-1, 1), y)
        return platt.predict_proba(proba.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"method must be 'isotonic' or 'sigmoid', got {method!r}")


def fit_calibrator(
    uncalibrated_proba: np.ndarray,
    y_true: np.ndarray,
    method: Literal["isotonic", "sigmoid"] = CALIBRATION_METHOD,
):
    """Fit a calibrator on (uncalibrated_proba, y_true) for use at inference.

    Returns an object with a .transform(proba) or .predict_proba(proba) method
    so new probabilities can be calibrated. Suitable for joblib serialization.
    """
    proba = np.asarray(uncalibrated_proba).ravel()
    y = np.asarray(y_true).ravel()

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(proba, y)
        return cal
    elif method == "sigmoid":
        platt = PlattLR(random_state=RANDOM_STATE, max_iter=1000)
        platt.fit(proba.reshape(-1, 1), y)
        return platt
    else:
        raise ValueError(f"method must be 'isotonic' or 'sigmoid', got {method!r}")


def apply_calibration(proba: np.ndarray, calibrator) -> np.ndarray:
    """Apply a fitted calibrator to probability estimates."""
    proba = np.asarray(proba).ravel()
    if hasattr(calibrator, "transform"):
        return calibrator.transform(proba)
    # Platt LR
    return calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
