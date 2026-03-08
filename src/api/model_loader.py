"""Load trained model artifact from artifacts/ for scoring."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import DEFAULT_MODEL_PATH
from src.data.schema import CORE_FEATURES
from src.models.calibration import apply_calibration


def load_scoring_artifact(path: Path | None = None) -> dict[str, Any]:
    """Load the baseline logistic artifact (model, scaler, calibrator).

    Supports two formats:
    - Full pipeline: joblib dict with model, scaler, calibrator, feature_names_in.
    - Simple model: joblib dump of LogisticRiskModel only (no scaling/calibration).

    Returns a dict with keys: model, scaler (optional), calibrator (optional),
    feature_names_in. Raises FileNotFoundError if the artifact is missing.
    """
    path = path or DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run training first, e.g.: "
            "python -m src.models.train_logistic"
        )
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj
    # Simple artifact: LogisticRiskModel only
    return {
        "model": obj,
        "scaler": None,
        "calibrator": None,
        "feature_names_in": getattr(obj, "feature_names_in_", None) or CORE_FEATURES,
    }


def score_one(
    artifact: dict[str, Any],
    feature_row: dict[str, float],
    use_calibration: bool = True,
) -> float:
    """Compute fraud/default probability for a single feature row.

    Args:
        artifact: Loaded artifact from load_scoring_artifact().
        feature_row: Dict of feature name -> value (e.g. from ScoreRequest.to_feature_row()).
        use_calibration: If True and artifact has a calibrator, apply it.

    Returns:
        Probability of positive class (fraud/default) in [0, 1].
    """
    model = artifact["model"]
    scaler = artifact.get("scaler")
    names = artifact.get("feature_names_in") or CORE_FEATURES
    X = pd.DataFrame([feature_row])[names]
    if scaler is not None:
        X = scaler.transform(X)
    proba = model.predict_proba(X)[0, 1]
    calibrator = artifact.get("calibrator")
    if use_calibration and calibrator is not None:
        proba = float(apply_calibration(proba.reshape(1), calibrator)[0])
    return float(proba)
