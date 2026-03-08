"""Training data schema (feature blueprint) for credit, payments, and fraud risk.

Mirrors production datasets: interpretable, statistically rich, and real-world relevant.
"""

# Core applicant/transaction features (order matches generator)
CORE_FEATURES = [
    "age",                    # continuous, mildly right-skewed
    "income",                  # log-normal
    "utilization",             # ratio [0, 1]
    "num_trades",              # count of credit lines
    "delinq_30d",              # binary delinquency flag
    "credit_history_length",   # years
    "transaction_amount",      # right-skewed, heavy-tail
    "merchant_risk_score",     # categorical/ordinal (0..K)
    "device_trust_score",      # continuous [0, 1]
    "velocity_score",          # rapid-fire transaction behavior
]

# Primary target (binary)
TARGET_FRAUD = "fraud_flag"
TARGET_DEFAULT = "default_flag"

# Optional auxiliary targets for expected loss (PD * LGD * EAD)
AUX_TARGETS = [
    "loss_given_default",   # LGD
    "exposure_at_default",  # EAD
]

# Default target name for modeling
DEFAULT_TARGET = TARGET_FRAUD


def get_feature_names() -> list[str]:
    """Return the ordered list of core feature names."""
    return list(CORE_FEATURES)


def get_target_name(use_default_label: bool = False) -> str:
    """Return the primary target column name."""
    return TARGET_DEFAULT if use_default_label else TARGET_FRAUD


def get_aux_target_names() -> list[str]:
    """Return optional auxiliary target names for LGD/EAD."""
    return list(AUX_TARGETS)
