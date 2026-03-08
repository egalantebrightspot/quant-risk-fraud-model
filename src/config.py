"""Configuration for the risk and fraud scoring model."""

from pathlib import Path

# Project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data and artifacts
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "baseline_logistic.joblib"

# Modeling defaults
RANDOM_STATE = 42
TEST_SIZE = 0.25
VALIDATION_SIZE = 0.2  # of training set, for calibration

# Logistic regression defaults
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000
LOGISTIC_SOLVER = "lbfgs"

# Calibration
CALIBRATION_METHOD = "isotonic"  # "isotonic" or "sigmoid" (Platt)
