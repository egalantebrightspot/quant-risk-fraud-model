#!/usr/bin/env python3
"""Test the scoring API with payloads for all new features."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

# Use API_PORT env var, or 8001 (8000 often blocked on Windows), then 8000
_PORT = int(os.environ.get("API_PORT", "8001"))
BASE = f"http://127.0.0.1:{_PORT}"

# Low-risk and high-risk single records
LOW_RISK = {
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
HIGH_RISK = {
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

def main():
    print(f"Using API base: {BASE}\n")
    with httpx.Client(timeout=30.0) as client:
        print("=== GET /health ===")
        r = client.get(f"{BASE}/health")
        print(r.json())
        print()

        print("=== POST /score (low risk) ===")
        r = client.post(f"{BASE}/score", json=LOW_RISK)
        print(r.status_code, r.json())
        print()

        print("=== POST /score (high risk) ===")
        r = client.post(f"{BASE}/score", json=HIGH_RISK)
        print(r.status_code, r.json())
        print()

        print("=== POST /score?include_shap=true (with SHAP + top contributors) ===")
        r = client.post(f"{BASE}/score", json=LOW_RISK, params={"include_shap": True})
        d = r.json()
        print("probability:", d.get("probability"), "risk_tier_letter:", d.get("risk_tier_letter"))
        print("top_contributors:", d.get("top_contributors"))
        print()

        print("=== POST /score with LGD/EAD (expected loss) ===")
        payload = {**LOW_RISK, "loss_given_default": 0.4, "exposure_at_default": 100.0}
        r = client.post(f"{BASE}/score", json=payload)
        d = r.json()
        print("probability:", d.get("probability"), "expected_loss:", d.get("expected_loss"))
        print()

        print("=== POST /score/batch (low + high risk) ===")
        r = client.post(f"{BASE}/score/batch", json=[LOW_RISK, HIGH_RISK])
        results = r.json()
        for i, item in enumerate(results):
            print(f"  [{i}] probability={item.get('probability')} risk_tier_letter={item.get('risk_tier_letter')} risk_tier_numeric={item.get('risk_tier_numeric')}")
        print()

        print("=== POST /score?model=gbm (optional GBM) ===")
        r = client.post(f"{BASE}/score", json=LOW_RISK, params={"model": "gbm"})
        print(r.status_code, r.json() if r.status_code == 200 else r.text[:200])
        print()

        print("=== POST /score?model=gbm&include_shap=true (GBM + top contributors) ===")
        r = client.post(f"{BASE}/score", json=HIGH_RISK, params={"model": "gbm", "include_shap": True})
        d = r.json() if r.status_code == 200 else {}
        print(r.status_code, "top_contributors:", d.get("top_contributors") if d else r.text[:200])

if __name__ == "__main__":
    main()
