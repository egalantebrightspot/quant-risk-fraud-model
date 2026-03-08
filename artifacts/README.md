# Model artifact directory

This folder stores trained models and related artifacts so the **API**, **evaluation**, **governance**, and **explainability** layers stay consistent and don’t depend on ad‑hoc paths.

## Layout

| Artifact | Description |
|----------|-------------|
| `logistic_model.joblib` | Logistic regression model from the simple training flow (`train_logistic_model()` or `--simple`). Load with `LogisticRiskModel.load()`. |
| `baseline_logistic.joblib` | Full pipeline bundle: model + scaler + calibrator + metadata. Used by the FastAPI scoring endpoint and evaluation. |
| `baseline_logistic.metrics.json` | Training metrics (AUC, KS) for the baseline model. |
| `gbm_model.joblib` | *(Future)* Gradient boosting model. |
| `shap_explainer.joblib` | *(Future)* Fitted SHAP explainer for model interpretability. |

## Why it matters

- **API** needs a trained model to serve PD/fraud scores.
- **Evaluation notebooks** need predictions for ROC, KS, lift, and calibration plots.
- **Governance** needs group-level performance metrics from a fixed model.
- **Explainability** needs a fitted model and (optionally) a saved SHAP explainer.

Training the baseline and saving it here is the hinge for everything downstream; the next step is the FastAPI scoring endpoint that loads from this directory and returns scores.

## Creating artifacts

- **Simple logistic:** `python -m src.models.train_logistic --simple` → `logistic_model.joblib`
- **Full pipeline (with calibration):** `python -m src.models.train_logistic` → `baseline_logistic.joblib` + `baseline_logistic.metrics.json`

Paths are defined in `src/config.py` (`ARTIFACTS_DIR`, `LOGISTIC_MODEL_PATH`, `DEFAULT_MODEL_PATH`, etc.).
