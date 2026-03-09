"""Structured logging and audit trails for fraud and credit systems.

Logs: incoming requests, scores returned, latency, errors, feature drift indicators.
Output is structured (JSON) for ingestion by log aggregators and compliance.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from src.data.schema import CORE_FEATURES

# Expected bounds per feature for drift detection (min, max) or None for unbounded
FEATURE_BOUNDS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "age": (18.0, 100.0),
    "income": (0.0, None),
    "utilization": (0.0, 1.0),
    "num_trades": (0.0, None),
    "delinq_30d": (0.0, 1.0),
    "credit_history_length": (0.0, None),
    "transaction_amount": (0.0, None),
    "merchant_risk_score": (0.0, 10.0),
    "device_trust_score": (0.0, 1.0),
    "velocity_score": (0.0, None),
}

AUDIT_LOGGER = "risk_fraud_audit"
LOG_FORMAT = "%(message)s"


# Standard LogRecord attributes to skip when building structured payload
_STD_ATTRS = frozenset(
    {"name", "msg", "args", "created", "filename", "funcName", "levelname",
     "levelno", "lineno", "module", "msecs", "pathname", "process", "processName",
     "relativeCreated", "stack_info", "exc_info", "exc_text", "thread", "threadName",
     "message", "taskName"}
)


class StructuredFormatter(logging.Formatter):
    """Format log records as single-line JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and v is not None:
                log_dict[k] = v
        return json.dumps(log_dict)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(AUDIT_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_request(
    request_id: str,
    method: str,
    path: str,
    query: Optional[str] = None,
) -> None:
    """Log incoming request for audit trail."""
    logger = _get_logger()
    extra: dict[str, Any] = {"request_id": request_id, "method": method, "path": path}
    if query:
        extra["query"] = query
    logger.info("request", extra=extra)


def log_response(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """Log response and latency."""
    logger = _get_logger()
    logger.info(
        "response",
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )


def log_score_audit(
    request_id: str,
    endpoint: str,
    model: str,
    probability: float,
    risk_tier: str,
    risk_tier_letter: str,
    latency_ms: float,
    decision: Optional[str] = None,
    batch_size: Optional[int] = None,
    drift: Optional[dict[str, Any]] = None,
) -> None:
    """Log score/decision for audit trail (scores returned)."""
    logger = _get_logger()
    extra: dict[str, Any] = {
        "request_id": request_id,
        "endpoint": endpoint,
        "model": model,
        "probability": round(probability, 6),
        "risk_tier": risk_tier,
        "risk_tier_letter": risk_tier_letter,
        "latency_ms": round(latency_ms, 2),
    }
    if decision is not None:
        extra["decision"] = decision
    if batch_size is not None:
        extra["batch_size"] = batch_size
    if drift is not None:
        extra["drift"] = drift
    logger.info("score_audit", extra=extra)


def log_error(
    request_id: str,
    path: str,
    status_code: int,
    error_type: str,
    detail: str,
    latency_ms: Optional[float] = None,
) -> None:
    """Log errors for audit and alerting."""
    logger = _get_logger()
    extra: dict[str, Any] = {
        "request_id": request_id,
        "path": path,
        "status_code": status_code,
        "error_type": error_type,
        "detail": detail,
    }
    if latency_ms is not None:
        extra["latency_ms"] = round(latency_ms, 2)
    logger.warning("error", extra=extra)


def feature_drift_indicators(feature_row: dict[str, float]) -> dict[str, Any]:
    """Compute drift indicators: out-of-bounds count and list of features outside expected bounds."""
    out_of_bounds: list[str] = []
    for name in CORE_FEATURES:
        if name not in feature_row:
            continue
        bounds = FEATURE_BOUNDS.get(name, (None, None))
        if bounds == (None, None):
            continue
        lo, hi = bounds
        val = feature_row[name]
        try:
            v = float(val)
        except (TypeError, ValueError):
            out_of_bounds.append(name)
            continue
        if lo is not None and v < lo:
            out_of_bounds.append(name)
        elif hi is not None and v > hi:
            out_of_bounds.append(name)
    return {
        "out_of_bounds_count": len(out_of_bounds),
        "features_out_of_bounds": out_of_bounds if out_of_bounds else None,
    }
