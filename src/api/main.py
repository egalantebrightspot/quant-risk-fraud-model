"""FastAPI scoring service.

Loads the trained model from artifacts/, validates input with Pydantic,
and returns structured fraud/default probability, risk tier, optional
SHAP explanations, expected loss (when LGD/EAD provided), and batch scoring.

Run from project root:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Requires artifacts/baseline_logistic.joblib (from python -m src.models.train_logistic).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from src.api.model_loader import load_scoring_artifact, score_one
from src.api.schemas import (
    ScoreRequest,
    ScoreResponse,
    probability_to_risk_tier,
    probability_to_risk_tier_letter,
    probability_to_risk_tier_numeric,
)
from src.config import DEFAULT_MODEL_PATH, GBM_MODEL_PATH
from src.explainability.shap_explainer import get_feature_contributions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup (logistic required, GBM optional)."""
    app.state.artifact_error = None
    try:
        app.state.artifact = load_scoring_artifact(DEFAULT_MODEL_PATH, model_type="logistic")
    except FileNotFoundError as e:
        app.state.artifact = None
        app.state.artifact_error = str(e)
    try:
        app.state.artifact_gbm = load_scoring_artifact(GBM_MODEL_PATH, model_type="gbm")
    except FileNotFoundError:
        app.state.artifact_gbm = None
    yield
    app.state.artifact = None
    app.state.artifact_gbm = None


app = FastAPI(
    title="Risk & Fraud Scoring API",
    description="PD/fraud probability, risk tiers (A–E / 1–5), SHAP explanations, expected loss, batch scoring.",
    version="0.1.0",
    lifespan=lifespan,
)


def _artifact_for_model(app: FastAPI, model: str) -> dict[str, Any] | None:
    if model == "gbm":
        return getattr(app.state, "artifact_gbm", None)
    return getattr(app.state, "artifact", None)


def _build_response(
    probability: float,
    request: ScoreRequest,
    artifact: dict[str, Any],
    feature_row: dict[str, float],
    include_shap: bool,
) -> ScoreResponse:
    risk_tier = probability_to_risk_tier(probability)
    risk_tier_letter = probability_to_risk_tier_letter(probability)
    risk_tier_numeric = probability_to_risk_tier_numeric(probability)
    fraud_flag = 1 if probability >= 0.5 else 0
    expected_loss = None
    if request.loss_given_default is not None and request.exposure_at_default is not None:
        expected_loss = round(
            probability * request.loss_given_default * request.exposure_at_default, 6
        )
    shap_values = None
    if include_shap:
        shap_values = get_feature_contributions(artifact, feature_row)
    return ScoreResponse(
        probability=round(probability, 6),
        risk_tier=risk_tier,
        risk_tier_letter=risk_tier_letter,
        risk_tier_numeric=risk_tier_numeric,
        fraud_flag=fraud_flag,
        expected_loss=expected_loss,
        shap_values=shap_values,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Health check; reports whether models are loaded."""
    logistic_loaded = getattr(app.state, "artifact", None) is not None
    gbm_loaded = getattr(app.state, "artifact_gbm", None) is not None
    if logistic_loaded or gbm_loaded:
        return {
            "status": "ok",
            "logistic": "loaded" if logistic_loaded else "not_loaded",
            "gbm": "loaded" if gbm_loaded else "not_loaded",
        }
    return {
        "status": "degraded",
        "model": "not_loaded",
        "hint": getattr(app.state, "artifact_error", "Run training to create artifact."),
    }


@app.post("/score", response_model=ScoreResponse)
def score(
    request: ScoreRequest,
    model: str = Query("logistic", description="Model to use: logistic or gbm"),
    include_shap: bool = Query(False, description="Include SHAP-based feature contributions"),
) -> ScoreResponse:
    """Compute fraud/default probability and risk tier for one record.

    Optional: LGD/EAD in body for expected loss; include_shap=true for explanations.
    """
    artifact = _artifact_for_model(app, model)
    if artifact is None:
        if model == "gbm":
            raise HTTPException(
                status_code=503,
                detail="GBM model not loaded. Train and save GBM artifact to artifacts/gbm_model.joblib.",
            )
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training and ensure artifact exists.",
        )
    feature_row = request.to_feature_row()
    probability = score_one(artifact, feature_row, use_calibration=True)
    return _build_response(probability, request, artifact, feature_row, include_shap)


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(
    requests: list[ScoreRequest],
    model: str = Query("logistic", description="Model to use: logistic or gbm"),
    include_shap: bool = Query(False, description="Include SHAP-based feature contributions"),
) -> list[ScoreResponse]:
    """Bulk risk evaluation: score multiple records in one request."""
    artifact = _artifact_for_model(app, model)
    if artifact is None:
        if model == "gbm":
            raise HTTPException(status_code=503, detail="GBM model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    responses = []
    for req in requests:
        feature_row = req.to_feature_row()
        probability = score_one(artifact, feature_row, use_calibration=True)
        responses.append(
            _build_response(probability, req, artifact, feature_row, include_shap)
        )
    return responses
