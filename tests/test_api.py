"""Tests for FastAPI scoring service."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.main import app
from src.api.schemas import ScoreRequest, ScoreResponse, probability_to_risk_tier


@pytest.fixture
def client():
    return TestClient(app)


def test_health_returns_ok_or_degraded(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] in ("ok", "degraded")


def test_score_request_schema():
    req = ScoreRequest(
        age=40,
        income=50,
        utilization=0.3,
        num_trades=6,
        delinq_30d=0,
        credit_history_length=10,
        transaction_amount=200,
        merchant_risk_score=2,
        device_trust_score=0.7,
        velocity_score=2.0,
    )
    row = req.to_feature_row()
    assert row["age"] == 40
    assert row["utilization"] == 0.3


def test_score_response_and_risk_tier():
    assert probability_to_risk_tier(0.05) == "low"
    assert probability_to_risk_tier(0.2) == "medium"
    assert probability_to_risk_tier(0.5) == "high"
    resp = ScoreResponse(probability=0.15, risk_tier="medium", fraud_flag=0)
    assert resp.fraud_flag == 0


def test_score_endpoint(client):
    payload = {
        "age": 35,
        "income": 60,
        "utilization": 0.2,
        "num_trades": 5,
        "delinq_30d": 0,
        "credit_history_length": 8,
        "transaction_amount": 100,
        "merchant_risk_score": 2,
        "device_trust_score": 0.8,
        "velocity_score": 1.5,
    }
    r = client.post("/score", json=payload)
    if r.status_code == 503:
        pytest.skip("Model artifact not loaded (run training first)")
    assert r.status_code == 200
    data = r.json()
    assert "probability" in data
    assert "risk_tier" in data
    assert "fraud_flag" in data
    assert 0 <= data["probability"] <= 1
    assert data["risk_tier"] in ("low", "medium", "high")
    assert data["fraud_flag"] in (0, 1)


def test_score_rejects_invalid_input(client):
    r = client.post("/score", json={"age": 10})  # missing fields, age out of range
    assert r.status_code == 422
