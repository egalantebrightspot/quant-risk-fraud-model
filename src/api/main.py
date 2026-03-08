"""FastAPI scoring service.

Loads the trained model from artifacts/, validates input with Pydantic,
and returns structured fraud/default probability and risk tier.

Run from project root:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Requires artifacts/baseline_logistic.joblib (from python -m src.models.train_logistic).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.model_loader import load_scoring_artifact, score_one
from src.api.schemas import ScoreRequest, ScoreResponse, probability_to_risk_tier
from src.config import DEFAULT_MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup."""
    try:
        app.state.artifact = load_scoring_artifact(DEFAULT_MODEL_PATH)
    except FileNotFoundError as e:
        app.state.artifact = None
        app.state.artifact_error = str(e)
    yield
    app.state.artifact = None


app = FastAPI(
    title="Risk & Fraud Scoring API",
    description="PD/fraud probability and risk tier for credit and fraud analytics.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check; reports whether the model is loaded."""
    if getattr(app.state, "artifact", None) is not None:
        return {"status": "ok", "model": "loaded"}
    return {
        "status": "degraded",
        "model": "not_loaded",
        "hint": getattr(app.state, "artifact_error", "Run training to create artifact."),
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Compute fraud/default probability and risk tier for one record.

    Validates input features, runs the trained model (with scaling and
    calibration when available), and returns a structured response.
    """
    artifact: dict[str, Any] | None = getattr(app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training and ensure artifact exists.",
        )
    feature_row = request.to_feature_row()
    probability = score_one(artifact, feature_row, use_calibration=True)
    risk_tier = probability_to_risk_tier(probability)
    fraud_flag = 1 if probability >= 0.5 else 0
    return ScoreResponse(
        probability=round(probability, 6),
        risk_tier=risk_tier,
        fraud_flag=fraud_flag,
    )
