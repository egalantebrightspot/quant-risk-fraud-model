"""SHAP values and feature importance explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.schema import CORE_FEATURES


def _gbm_shap_values(model: Any, X: pd.DataFrame, names: list[str]) -> dict[str, float]:
    """Compute SHAP values for a tree-based GBM (e.g. LightGBM) using TreeExplainer."""
    import shap

    # GBMRiskModel wraps _estimator (LGBMClassifier), which has .booster_
    raw = getattr(model, "_estimator", model)
    booster = getattr(raw, "booster_", None)
    if booster is None:
        return {}
    explainer = shap.TreeExplainer(
        booster,
        feature_perturbation="tree_path_dependent",
    )
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]  # positive class (fraud/default)
    if hasattr(sv, "ndim") and sv.ndim == 2:
        sv = sv[0]
    return {names[i]: round(float(sv[i]), 6) for i in range(len(names))}


def get_feature_contributions(
    artifact: dict[str, Any],
    feature_row: dict[str, float],
) -> dict[str, float]:
    """Per-feature contribution to the score (SHAP-like).

    For logistic regression uses coefficient * (scaled) feature value.
    For GBM (LightGBM) uses SHAP TreeExplainer at score time.
    Otherwise uses artifact['shap_explainer'] if present.

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

    # GBM (e.g. GBMRiskModel / LightGBM): use TreeExplainer
    if getattr(model, "_estimator", None) is not None:
        return _gbm_shap_values(model, X, names)
    if hasattr(model, "booster_"):
        return _gbm_shap_values(model, X, names)

    # Optional: use fitted SHAP explainer from artifact
    explainer = artifact.get("shap_explainer")
    if explainer is not None and hasattr(explainer, "shap_values"):
        X_df = pd.DataFrame([feature_row])[names]
        if scaler is not None:
            X_df = pd.DataFrame(scaler.transform(X_df), columns=names)
        sv = explainer.shap_values(X_df)
        if isinstance(sv, list):
            sv = sv[1]  # positive class
        if hasattr(sv, "ndim") and sv.ndim == 2:
            sv = sv[0]
        return {names[i]: round(float(sv[i]), 6) for i in range(len(names))}

    return {}
