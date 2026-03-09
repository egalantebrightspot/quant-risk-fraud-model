"""API request/response schemas.

Pydantic models with optional fields, defaults, type coercion, and schema
versioning so the API can evolve without breaking clients.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from src.data.schema import CORE_FEATURES

# Feature versioning: bump when request/response contract changes
SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = ("1.0",)


def _coerce_to_float(value: Any) -> Optional[float]:
    """Coerce int, float, or numeric string to float; None stays None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
    return None


class TopContributor(BaseModel):
    """Single top contributing feature for explainability."""

    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="Contribution to score (positive = increases risk)")


class ModelInfoResponse(BaseModel):
    """Model metadata for governance and monitoring (/model_info)."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Response schema version")
    model_type: str = Field(..., description="logistic or gbm")
    model_version: str = Field(..., description="Model/artifact version")
    training_date: Optional[str] = Field(default=None, description="ISO date when model was trained")
    auc: Optional[float] = Field(default=None, description="Test AUC (calibrated if applicable)")
    ks: Optional[float] = Field(default=None, description="Test KS statistic")
    feature_list: list[str] = Field(..., description="Ordered list of input features")
    calibration_status: str = Field(
        ...,
        description="calibrated (isotonic/sigmoid) or uncalibrated",
    )

    model_config = {"extra": "ignore"}


class ScoreRequest(BaseModel):
    """Input features for a single score request.

    Supports schema versioning (schema_version), optional LGD/EAD with defaults,
    and type coercion (int/string -> float). Extra keys are ignored for evolution.
    """

    # Feature versioning: client can send schema_version to request a contract version
    schema_version: Optional[str] = Field(
        default=SCHEMA_VERSION,
        description="Request schema version (e.g. 1.0). Enables API evolution.",
    )
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
    loss_given_default: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="LGD for expected loss (optional)",
    )
    exposure_at_default: Optional[float] = Field(
        default=None,
        ge=0,
        description="EAD for expected loss (optional)",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_and_unwrap(cls, data: Any) -> Any:
        """Unwrap Swagger-style body; coerce int/str to float for numeric fields."""
        if not isinstance(data, dict):
            return data
        if set(data.keys()) <= {"summary", "value"} and "value" in data and isinstance(data["value"], dict):
            data = data["value"]
        out = dict(data)
        numeric_keys = (
            "age", "income", "utilization", "num_trades", "delinq_30d",
            "credit_history_length", "transaction_amount", "merchant_risk_score",
            "device_trust_score", "velocity_score", "loss_given_default", "exposure_at_default",
        )
        for key in numeric_keys:
            if key not in out:
                continue
            val = _coerce_to_float(out[key])
            if val is not None:
                out[key] = val
            elif key in ("loss_given_default", "exposure_at_default"):
                out[key] = None
        # Default schema_version if missing
        if "schema_version" not in out or out["schema_version"] is None:
            out["schema_version"] = SCHEMA_VERSION
        return out

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "schema_version": "1.0",
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
    """Structured scoring response; includes schema_version for evolution."""

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Response schema version for client compatibility",
    )
    probability: float = Field(..., ge=0, le=1, description="Fraud/default probability (PD)")
    risk_tier: str = Field(..., description="Risk band: low, medium, high")
    risk_tier_letter: str = Field(..., description="Risk grade A (best) to E (worst)")
    risk_tier_numeric: int = Field(..., ge=1, le=5, description="Risk bucket 1 (best) to 5 (worst)")
    fraud_flag: int = Field(..., ge=0, le=1, description="Binary decision at 0.5 threshold")
    expected_loss: Optional[float] = Field(default=None, description="PD * LGD * EAD when LGD/EAD provided")
    shap_values: Optional[dict[str, float]] = Field(default=None, description="Per-feature contribution to score")
    top_contributors: Optional[list[TopContributor]] = Field(
        default=None,
        description="Top contributing features by |contribution| (when include_shap=true)",
    )

    model_config = {"extra": "ignore"}


class ReasonCode(BaseModel):
    """Industry-standard reason code for model transparency and auditability."""

    code: str = Field(..., description="Standard reason code (e.g. HIGH_UTILIZATION)")
    description: str = Field(..., description="Human-readable explanation")
    direction: str = Field(..., description="increases_risk or decreases_risk")
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="Numeric contribution to score")


class ExplainResponse(BaseModel):
    """Dedicated explainability response: SHAP, contributions by magnitude, optional reason codes."""

    probability: float = Field(..., ge=0, le=1, description="Predicted fraud/default probability")
    risk_tier: str = Field(..., description="Risk band (low / medium / high)")
    risk_tier_letter: str = Field(..., description="Risk grade A–E")
    risk_tier_numeric: int = Field(..., ge=1, le=5, description="Risk bucket 1–5")
    shap_values: dict[str, float] = Field(..., description="Local SHAP values per feature")
    contributions_sorted: list[TopContributor] = Field(
        ...,
        description="Feature contributions sorted by magnitude (descending |contribution|)",
    )
    reason_codes: Optional[list[ReasonCode]] = Field(
        None,
        description="Industry-standard reason codes (when include_reason_codes=true)",
    )

    model_config = {"extra": "forbid"}


def top_contributors_from_shap(
    shap_values: dict[str, float],
    top_n: Optional[int] = None,
) -> list[TopContributor]:
    """Return features by absolute contribution (descending). If top_n is None, return all."""
    if not shap_values:
        return []
    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    if top_n is not None:
        sorted_features = sorted_features[:top_n]
    return [TopContributor(feature=k, contribution=round(v, 6)) for k, v in sorted_features]


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


def probability_to_decision(
    probability: float,
    approve_below: float = 0.15,
    decline_above: float = 0.40,
) -> str:
    """Map probability to actionable decision: approve, review, or decline.

    Default thresholds: approve when PD < 0.15, decline when PD > 0.40, else review.
    """
    if probability < approve_below:
        return "approve"
    if probability > decline_above:
        return "decline"
    return "review"


def decision_summary(decision: str, fraud_risk_level: str, tier_letter: str) -> str:
    """Human-readable one-line summary for the decision outcome."""
    level = fraud_risk_level.capitalize()
    return f"{decision.capitalize()} — {level} fraud risk (Tier {tier_letter})"


class DecisionResponse(BaseModel):
    """Decision-engine response with schema_version for evolution."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Response schema version")
    probability: float = Field(..., ge=0, le=1, description="Fraud/default probability (PD)")
    tier_letter: str = Field(..., description="Risk grade A (best) to E (worst)")
    tier_numeric: int = Field(..., ge=1, le=5, description="Risk bucket 1 (best) to 5 (worst)")
    fraud_risk_level: str = Field(
        ...,
        description="Fraud risk level: Low, Medium, or High",
    )
    decision: str = Field(
        ...,
        description="Actionable outcome: approve, review, or decline",
    )
    fraud_flag: int = Field(..., ge=0, le=1, description="Binary fraud indicator (1 if PD >= 0.5)")
    summary: str = Field(
        ...,
        description="Short human-readable summary of the decision outcome",
    )

    model_config = {"extra": "ignore"}
