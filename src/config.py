"""Configuration for the risk and fraud scoring model."""

from pathlib import Path

# Project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data and artifacts
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Model and artifact paths (keeps API and training consistent)
LOGISTIC_MODEL_PATH = ARTIFACTS_DIR / "logistic_model.joblib"  # simple model from train_logistic_model()
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "baseline_logistic.joblib"  # full pipeline: model + scaler + calibrator
GBM_MODEL_PATH = ARTIFACTS_DIR / "gbm_model.joblib"             # future: gradient boosting model
SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"   # future: fitted SHAP explainer

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

# Model metadata (governance / model_info endpoint)
MODEL_VERSION = "1.0"
