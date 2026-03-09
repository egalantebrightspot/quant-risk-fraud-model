"""FastAPI scoring service.

Loads the trained model from artifacts/, validates input with Pydantic,
and returns structured fraud/default probability, risk tier, optional
SHAP explanations, expected loss (when LGD/EAD provided), and batch scoring.

Run from project root:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Requires artifacts/baseline_logistic.joblib (from python -m src.models.train_logistic).
"""

import json
import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query

from src.api.model_loader import load_scoring_artifact, score_one
from src.api.schemas import (
    DecisionResponse,
    ExplainResponse,
    ModelInfoResponse,
    ReasonCode,
    ScoreRequest,
    ScoreResponse,
    decision_summary,
    probability_to_decision,
    probability_to_risk_tier,
    probability_to_risk_tier_letter,
    probability_to_risk_tier_numeric,
    top_contributors_from_shap,
)
from src.config import DEFAULT_MODEL_PATH, GBM_MODEL_PATH, MODEL_VERSION
from src.data.schema import CORE_FEATURES
from src.explainability.reason_codes import build_reason_codes
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


def _artifact_for_model(app: FastAPI, model: str) -> Optional[dict[str, Any]]:
    if model == "gbm":
        return getattr(app.state, "artifact_gbm", None)
    return getattr(app.state, "artifact", None)


def _model_info_from_artifact(
    artifact: dict[str, Any],
    model_type: str,
    artifact_path: Path,
) -> ModelInfoResponse:
    """Build model metadata from loaded artifact and optional .metrics.json."""
    feature_list = list(artifact.get("feature_names_in") or CORE_FEATURES)
    calibrator = artifact.get("calibrator")
    cal_method = artifact.get("calibration_method", "isotonic")
    calibration_status = (
        f"calibrated ({cal_method})" if calibrator is not None else "uncalibrated"
    )
    model_version = MODEL_VERSION
    training_date = None
    auc = None
    ks = None
    metrics_path = artifact_path.with_suffix(".metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            model_version = data.get("model_version", model_version)
            training_date = data.get("trained_at")
            auc = data.get("auc_calibrated") or data.get("auc")
            ks = data.get("ks_calibrated") or data.get("ks")
        except Exception:
            pass
    if training_date is None and artifact_path.exists():
        try:
            mtime = artifact_path.stat().st_mtime
            training_date = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return ModelInfoResponse(
        model_type=model_type,
        model_version=model_version,
        training_date=training_date,
        auc=auc,
        ks=ks,
        feature_list=feature_list,
        calibration_status=calibration_status,
    )


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
    top_contributors = None
    if include_shap:
        shap_values = get_feature_contributions(artifact, feature_row)
        top_contributors = top_contributors_from_shap(shap_values, top_n=5)
    return ScoreResponse(
        probability=round(probability, 6),
        risk_tier=risk_tier,
        risk_tier_letter=risk_tier_letter,
        risk_tier_numeric=risk_tier_numeric,
        fraud_flag=fraud_flag,
        expected_loss=expected_loss,
        shap_values=shap_values,
        top_contributors=top_contributors,
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


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info(
    model: str = Query("logistic", description="Model to describe: logistic or gbm"),
) -> ModelInfoResponse:
    """Model metadata for governance and monitoring.

    Returns model version, training date, AUC/KS metrics (if saved),
    feature list, and calibration status.
    """
    if model == "gbm":
        artifact = getattr(app.state, "artifact_gbm", None)
        path = GBM_MODEL_PATH
    else:
        artifact = getattr(app.state, "artifact", None)
        path = DEFAULT_MODEL_PATH
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=f"{model} model not loaded. Train and load artifact first.",
        )
    return _model_info_from_artifact(artifact, model, path)


# Sample payloads for Swagger UI
_EXAMPLE_LOW_RISK = {
    "age": 45,
    "income": 85,
    "utilization": 0.15,
    "num_trades": 6,
    "delinq_30d": 0,
    "credit_history_length": 15,
    "transaction_amount": 75,
    "merchant_risk_score": 1,
    "device_trust_score": 0.9,
    "velocity_score": 0.8,
}
_EXAMPLE_HIGH_RISK = {
    "age": 28,
    "income": 35,
    "utilization": 0.85,
    "num_trades": 12,
    "delinq_30d": 1,
    "credit_history_length": 3,
    "transaction_amount": 450,
    "merchant_risk_score": 4,
    "device_trust_score": 0.2,
    "velocity_score": 8.0,
}
_EXAMPLE_WITH_LGD_EAD = {
    **_EXAMPLE_LOW_RISK,
    "loss_given_default": 0.4,
    "exposure_at_default": 100.0,
}
_EXAMPLE_BATCH = [_EXAMPLE_LOW_RISK, _EXAMPLE_HIGH_RISK]


@app.post("/score", response_model=ScoreResponse)
def score(
    request: ScoreRequest = Body(
        ...,
        examples=[
            {"summary": "Low risk", "value": _EXAMPLE_LOW_RISK},
            {"summary": "High risk", "value": _EXAMPLE_HIGH_RISK},
            {"summary": "With LGD/EAD (expected loss)", "value": _EXAMPLE_WITH_LGD_EAD},
        ],
    ),
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


@app.post("/decide", response_model=DecisionResponse)
def decide(
    request: ScoreRequest = Body(
        ...,
        examples=[
            {"summary": "Low risk", "value": _EXAMPLE_LOW_RISK},
            {"summary": "High risk", "value": _EXAMPLE_HIGH_RISK},
        ],
    ),
    model: str = Query("logistic", description="Model to use: logistic or gbm"),
    approve_below: float = Query(0.15, ge=0, lt=1, description="Approve when probability < this"),
    decline_above: float = Query(0.40, ge=0, le=1, description="Decline when probability > this"),
) -> DecisionResponse:
    """Map probability into risk tiers and decision outcome (Approve / Review / Decline).

    Decision engine: returns Tier A–E, fraud risk level (Low/Medium/High),
    and actionable decision. Optional thresholds for approve/decline.
    """
    if approve_below >= decline_above:
        raise HTTPException(
            status_code=400,
            detail="approve_below must be less than decline_above.",
        )
    artifact = _artifact_for_model(app, model)
    if artifact is None:
        if model == "gbm":
            raise HTTPException(status_code=503, detail="GBM model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    feature_row = request.to_feature_row()
    probability = score_one(artifact, feature_row, use_calibration=True)
    prob = round(probability, 6)
    tier_letter = probability_to_risk_tier_letter(probability)
    tier_numeric = probability_to_risk_tier_numeric(probability)
    risk_tier = probability_to_risk_tier(probability)
    fraud_risk_level = risk_tier.capitalize()  # low -> Low, medium -> Medium, high -> High
    decision = probability_to_decision(probability, approve_below=approve_below, decline_above=decline_above)
    fraud_flag = 1 if probability >= 0.5 else 0
    summary = decision_summary(decision, fraud_risk_level, tier_letter)
    return DecisionResponse(
        probability=prob,
        tier_letter=tier_letter,
        tier_numeric=tier_numeric,
        fraud_risk_level=fraud_risk_level,
        decision=decision,
        fraud_flag=fraud_flag,
        summary=summary,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(
    request: ScoreRequest = Body(
        ...,
        examples=[
            {"summary": "Low risk", "value": _EXAMPLE_LOW_RISK},
            {"summary": "High risk", "value": _EXAMPLE_HIGH_RISK},
        ],
    ),
    model: str = Query("logistic", description="Model to use: logistic or gbm"),
    include_reason_codes: bool = Query(True, description="Include industry-standard reason codes"),
    top_n_reason_codes: int = Query(10, ge=1, le=20, description="Max reason codes to return"),
) -> ExplainResponse:
    """Model explainability: SHAP values, contributions sorted by magnitude, optional reason codes.

    Dedicated route for regulated risk systems. Returns local explanations
    so the model is auditable and transparent.
    """
    artifact = _artifact_for_model(app, model)
    if artifact is None:
        if model == "gbm":
            raise HTTPException(
                status_code=503,
                detail="GBM model not loaded. Train and save GBM artifact first.",
            )
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training and ensure artifact exists.",
        )
    feature_row = request.to_feature_row()
    probability = score_one(artifact, feature_row, use_calibration=True)
    shap_values = get_feature_contributions(artifact, feature_row)
    if not shap_values:
        raise HTTPException(
            status_code=501,
            detail="Explainability not available for this model.",
        )
    contributions_sorted = top_contributors_from_shap(shap_values, top_n=None)
    risk_tier = probability_to_risk_tier(probability)
    risk_tier_letter = probability_to_risk_tier_letter(probability)
    risk_tier_numeric = probability_to_risk_tier_numeric(probability)

    reason_codes_list = None
    if include_reason_codes:
        raw_codes = build_reason_codes(
            [{"feature": c.feature, "contribution": c.contribution} for c in contributions_sorted],
            top_n=top_n_reason_codes,
        )
        reason_codes_list = [ReasonCode(**r) for r in raw_codes]

    return ExplainResponse(
        probability=round(probability, 6),
        risk_tier=risk_tier,
        risk_tier_letter=risk_tier_letter,
        risk_tier_numeric=risk_tier_numeric,
        shap_values=shap_values,
        contributions_sorted=contributions_sorted,
        reason_codes=reason_codes_list,
    )


@app.post("/score/batch")
def score_batch(
    requests: list[ScoreRequest] = Body(
        ...,
        examples=[
            {"summary": "Low risk + High risk", "value": _EXAMPLE_BATCH},
        ],
    ),
    model: str = Query("logistic", description="Model to use: logistic or gbm"),
    include_shap: bool = Query(False, description="Include SHAP-based feature contributions"),
    format: str = Query("full", description="Response format: full (list of score objects) or minimal (vector of probabilities only)"),
):
    """Batch scoring: list of applicants/transactions → vector of scores.

    Accepts a list of applicants or transactions and returns scores for each.
    Useful for portfolio scoring, backtesting, monitoring, and bulk fraud sweeps.
    Mirrors enterprise scoring services.

    - **full** (default): list of ScoreResponse (probability, risk_tier, risk_tier_letter, etc.).
    - **minimal**: { "scores": [float, ...], "count": N } for high-throughput backtesting.
    """
    if format not in ("full", "minimal"):
        raise HTTPException(status_code=400, detail="format must be 'full' or 'minimal'.")
    artifact = _artifact_for_model(app, model)
    if artifact is None:
        if model == "gbm":
            raise HTTPException(status_code=503, detail="GBM model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    responses = []
    scores_only = []
    for req in requests:
        feature_row = req.to_feature_row()
        probability = score_one(artifact, feature_row, use_calibration=True)
        prob = round(probability, 6)
        scores_only.append(prob)
        if format == "full":
            responses.append(
                _build_response(probability, req, artifact, feature_row, include_shap)
            )
    if format == "minimal":
        return {"scores": scores_only, "count": len(scores_only)}
    return responses
