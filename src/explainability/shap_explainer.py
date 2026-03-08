"""SHAP values and feature importance explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.schema import CORE_FEATURES


def get_feature_contributions(
    artifact: dict[str, Any],
    feature_row: dict[str, float],
) -> dict[str, float]:
    """Per-feature contribution to the score (SHAP-like for linear model).

    For logistic regression uses coefficient * (scaled) feature value as
    contribution. For GBM or other models, returns empty dict unless
    a fitted explainer is present in the artifact.

    Returns:
        Dict mapping feature name to contribution (positive = increases risk).
    """
    model = artifact.get("model")
    if model is None:
        return {}
    scaler = artifact.get("scaler")
    names = artifact.get("feature_names_in") or CORE_FEATURES
    X = pd.DataFrame([feature_row])[names]
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    # LogisticRiskModel or sklearn LogisticRegression
    if hasattr(model, "get_coefficients"):
        coef = model.get_coefficients()
        contrib = (coef * X_scaled[0]).to_dict()
        return {k: round(float(v), 6) for k, v in contrib.items()}
    if hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        contrib = {names[i]: round(float(coef[i] * X_scaled[0, i]), 6) for i in range(len(names))}
        return contrib

    # Optional: use fitted SHAP explainer from artifact
    explainer = artifact.get("shap_explainer")
    if explainer is not None and hasattr(explainer, "shap_values"):
        import numpy as np
        X_df = pd.DataFrame([feature_row])[names]
        if scaler is not None:
            X_df = pd.DataFrame(scaler.transform(X_df), columns=names)
        sv = explainer.shap_values(X_df)
        if isinstance(sv, list):
            sv = sv[1]  # positive class
        if sv.ndim == 2:
            sv = sv[0]
        return {names[i]: round(float(sv[i]), 6) for i in range(len(names))}

    return {}
