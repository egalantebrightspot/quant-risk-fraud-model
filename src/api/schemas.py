"""API request/response schemas.

Pydantic models for strict input validation and deterministic JSON outputs,
matching production risk and fraud scoring services.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.data.schema import CORE_FEATURES


class ScoreRequest(BaseModel):
    """Input features for a single score request.

    Field order and names match the training schema (CORE_FEATURES).
    Optional LGD/EAD for expected loss when provided.
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
    loss_given_default: Optional[float] = Field(None, ge=0, le=1, description="LGD for expected loss (optional)")
    exposure_at_default: Optional[float] = Field(None, ge=0, description="EAD for expected loss (optional)")

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
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
        },
    }

    def to_feature_row(self) -> dict[str, float]:
        """Return a dict of feature name -> value in schema order for scoring (excludes LGD/EAD)."""
        return {name: getattr(self, name) for name in CORE_FEATURES}


class ScoreResponse(BaseModel):
    """Structured scoring response for downstream systems."""

    probability: float = Field(..., ge=0, le=1, description="Fraud/default probability (PD)")
    risk_tier: str = Field(..., description="Risk band: low, medium, high")
    risk_tier_letter: str = Field(..., description="Risk grade A (best) to E (worst)")
    risk_tier_numeric: int = Field(..., ge=1, le=5, description="Risk bucket 1 (best) to 5 (worst)")
    fraud_flag: int = Field(..., ge=0, le=1, description="Binary decision at 0.5 threshold")
    expected_loss: Optional[float] = Field(None, description="PD * LGD * EAD when LGD/EAD provided")
    shap_values: Optional[dict[str, float]] = Field(None, description="Per-feature contribution to score")

    model_config = {"extra": "forbid"}


def probability_to_risk_tier(probability: float) -> str:
    """Map probability to a risk tier label (low / medium / high)."""
    if probability < 0.1:
        return "low"
    if probability < 0.3:
        return "medium"
    return "high"


def probability_to_risk_tier_letter(probability: float) -> str:
    """Map probability to letter grade A (lowest risk) to E (highest risk)."""
    if probability < 0.05:
        return "A"
    if probability < 0.15:
        return "B"
    if probability < 0.30:
        return "C"
    if probability < 0.50:
        return "D"
    return "E"


def probability_to_risk_tier_numeric(probability: float) -> int:
    """Map probability to bucket 1 (lowest risk) to 5 (highest risk)."""
    if probability < 0.05:
        return 1
    if probability < 0.15:
        return 2
    if probability < 0.30:
        return 3
    if probability < 0.50:
        return 4
    return 5
