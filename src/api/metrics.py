"""Prometheus metrics for risk/fraud API.

Exposes request count, latency, error rate, average score, and score distribution
for Prometheus scraping and Grafana dashboards. Used to detect drift and model degradation.
"""

from prometheus_client import Counter, Histogram, REGISTRY, generate_latest, CONTENT_TYPE_LATEST

# Request count by path, method, and status class (2xx, 4xx, 5xx)
REQUESTS_TOTAL = Counter(
    "risk_api_requests_total",
    "Total HTTP requests",
    ["path", "method", "status_class"],
)

# Request latency in seconds (for percentile and average in Grafana)
REQUEST_LATENCY = Histogram(
    "risk_api_request_latency_seconds",
    "Request latency in seconds",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Score (probability) distribution by endpoint and model — average and distribution over time
SCORE_DISTRIBUTION = Histogram(
    "risk_api_score",
    "Scored probability (PD) distribution for drift and degradation detection",
    ["endpoint", "model"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def status_class(status_code: int) -> str:
    """Return 2xx, 4xx, or 5xx for labeling."""
    if status_code < 400:
        return "2xx"
    if status_code < 500:
        return "4xx"
    return "5xx"


def record_request(path: str, method: str, status_code: int, latency_seconds: float) -> None:
    """Record a completed request for count and latency metrics."""
    sc = status_class(status_code)
    REQUESTS_TOTAL.labels(path=path, method=method, status_class=sc).inc()
    REQUEST_LATENCY.labels(path=path).observe(latency_seconds)


def record_score(endpoint: str, model: str, probability: float) -> None:
    """Record a score for average and distribution metrics."""
    SCORE_DISTRIBUTION.labels(endpoint=endpoint, model=model).observe(probability)


def get_metrics() -> bytes:
    """Return Prometheus text format for GET /metrics."""
    return generate_latest(REGISTRY)


def get_content_type() -> str:
    """Content-Type for metrics response."""
    return CONTENT_TYPE_LATEST
