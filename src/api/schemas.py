"""API request/response schemas.

Pydantic models for strict input validation and deterministic JSON outputs,
matching production risk and fraud scoring services.
"""

from pydantic import BaseModel, Field

from src.data.schema import CORE_FEATURES


class ScoreRequest(BaseModel):
    """Input features for a single score request.

    Field order and names match the training schema (CORE_FEATURES).
    """

    age: float = Field(..., ge=18, le=100, description="Applicant/transaction age (years)")
    income: float = Field(..., gt=0, description="Income (e.g. thousands)")
    utilization: float = Field(..., ge=0, le=1, description="Credit utilization ratio [0, 1]")
    num_trades: float = Field(..., ge=0, description="Number of credit lines/trades")
    delinq_30d: float = Field(..., ge=0, le=1, description="Delinquency flag (0 or 1)")
    credit_history_length: float = Field(..., ge=0, description="Credit history (years)")
    transaction_amount: float = Field(..., ge=0, description="Transaction amount")
    merchant_risk_score: float = Field(..., ge=0, le=10, description="Merchant risk (ordinal)")
    device_trust_score: float = Field(..., ge=0, le=1, description="Device trust [0, 1]")
    velocity_score: float = Field(..., ge=0, description="Velocity / rapid-activity score")

    model_config = {"extra": "forbid"}

    def to_feature_row(self) -> dict[str, float]:
        """Return a dict of feature name -> value in schema order for scoring."""
        return {name: getattr(self, name) for name in CORE_FEATURES}


class ScoreResponse(BaseModel):
    """Structured scoring response for downstream systems."""

    probability: float = Field(..., ge=0, le=1, description="Fraud/default probability")
    risk_tier: str = Field(..., description="Risk band: low, medium, high")
    fraud_flag: int = Field(..., ge=0, le=1, description="Binary decision at 0.5 threshold")

    model_config = {"extra": "forbid"}


def probability_to_risk_tier(probability: float) -> str:
    """Map probability to a risk tier label."""
    if probability < 0.1:
        return "low"
    if probability < 0.3:
        return "medium"
    return "high"
