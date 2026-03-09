"""Industry-standard reason codes for model transparency and auditability.

Maps feature names and contribution direction to standard codes used in
regulated risk and fraud systems.
"""

from __future__ import annotations

from typing import Any

from src.data.schema import CORE_FEATURES


# (code_increases_risk, description_increases_risk, code_decreases_risk, description_decreases_risk)
REASON_CODE_MAP: dict[str, tuple[str, str, str, str]] = {
    "age": (
        "AGE_RISK_FACTOR",
        "Age contributes to higher risk",
        "AGE_MITIGATING",
        "Age contributes to lower risk",
    ),
    "income": (
        "LOW_INCOME",
        "Lower income increases risk",
        "HIGH_INCOME",
        "Higher income decreases risk",
    ),
    "utilization": (
        "HIGH_UTILIZATION",
        "High credit utilization increases risk",
        "LOW_UTILIZATION",
        "Low utilization decreases risk",
    ),
    "num_trades": (
        "ELEVATED_NUM_TRADES",
        "Number of trades increases risk",
        "LOW_NUM_TRADES",
        "Fewer trades decreases risk",
    ),
    "delinq_30d": (
        "DELINQUENCY_FLAG",
        "Recent delinquency increases risk",
        "NO_DELINQUENCY",
        "No delinquency decreases risk",
    ),
    "credit_history_length": (
        "SHORT_CREDIT_HISTORY",
        "Short credit history increases risk",
        "LONG_CREDIT_HISTORY",
        "Longer credit history decreases risk",
    ),
    "transaction_amount": (
        "HIGH_TRANSACTION_AMOUNT",
        "High transaction amount increases risk",
        "LOW_TRANSACTION_AMOUNT",
        "Lower amount decreases risk",
    ),
    "merchant_risk_score": (
        "HIGH_MERCHANT_RISK",
        "High merchant risk score increases risk",
        "LOW_MERCHANT_RISK",
        "Low merchant risk decreases risk",
    ),
    "device_trust_score": (
        "LOW_DEVICE_TRUST",
        "Low device trust increases risk",
        "HIGH_DEVICE_TRUST",
        "High device trust decreases risk",
    ),
    "velocity_score": (
        "HIGH_VELOCITY",
        "High velocity/activity increases risk",
        "LOW_VELOCITY",
        "Lower velocity decreases risk",
    ),
}


def build_reason_codes(
    contributions_sorted: list[dict[str, Any]],
    top_n: int | None = 10,
) -> list[dict[str, Any]]:
    """Build industry-standard reason codes from contributions sorted by magnitude.

    Args:
        contributions_sorted: List of {"feature": str, "contribution": float}.
        top_n: Max number of reason codes (default 10). None = all.

    Returns:
        List of dicts with keys: code, description, direction, feature, contribution.
    """
    out: list[dict[str, Any]] = []
    seen = 0
    for item in contributions_sorted:
        if top_n is not None and seen >= top_n:
            break
        feature = item.get("feature")
        contribution = item.get("contribution", 0.0)
        if feature in REASON_CODE_MAP:
            mapping = REASON_CODE_MAP[feature]
        else:
            code_inc = f"{feature.upper()}_INCREASES_RISK"
            code_dec = f"{feature.upper()}_DECREASES_RISK"
            desc_inc = f"{feature} contributes to higher risk"
            desc_dec = f"{feature} contributes to lower risk"
            mapping = (code_inc, desc_inc, code_dec, desc_dec)
        if contribution > 0:
            code, description = mapping[0], mapping[1]
            direction = "increases_risk"
        else:
            code, description = mapping[2], mapping[3]
            direction = "decreases_risk"
        out.append({
            "code": code,
            "description": description,
            "direction": direction,
            "feature": feature,
            "contribution": round(contribution, 6),
        })
        seen += 1
    return out
