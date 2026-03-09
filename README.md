# Quantitative Risk & Fraud Scoring Model

A production-grade risk and fraud scoring engine that combines statistical and machine learning models to estimate default/fraud probability, expected loss, and risk tiers. Designed to look and feel like a real decisioning service used in credit, payments, and fraud analytics.

---

## How to run

From the project root (with a virtualenv activated and `pip install -r requirements.txt`):

```bash
# 1. Generate synthetic training data (optional; training can generate if missing)
python -m src.data.loaders

# 2. Train the logistic model (saves to artifacts/baseline_logistic.joblib)
python -m src.models.train_logistic
# Or generate data then train in one go:
python -m src.models.train_logistic --synthetic

# 3. Serve the scoring API
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` for the interactive API.

---

## Overview

This project implements a full risk modeling pipeline:

- Data preprocessing and feature engineering
- Multiple risk models (logistic regression, gradient boosting, optional Poisson/Count models)
- Probability of default (PD) / fraud likelihood estimation
- Expected loss and risk tiering
- Model evaluation (ROC, AUC, KS, lift, calibration)
- Bias and fairness checks
- Explainability (SHAP, feature importance)
- FastAPI scoring service

The goal is to demonstrate quantitative modeling depth, governance, and production-quality engineering.

---

## Features

- **Multiple models:** Logistic regression, gradient boosting, optional Poisson/Count model  
- **Risk metrics:** PD, expected loss, risk tiers, scorecards  
- **Evaluation:** ROC/AUC, KS, lift curves, calibration plots  
- **Governance:** Basic bias/fairness checks across groups  
- **Explainability:** SHAP values, feature importance, per-sample explanations  
- **API:** FastAPI endpoint for real-time scoring  
- **Reproducibility:** Notebooks for EDA, modeling, and evaluation  

---

## Repository structure

```text
quant-risk-fraud-model/
│
├── artifacts/          # Trained models, calibration, SHAP explainers (see artifacts/README.md)
│   ├── logistic_model.joblib
│   ├── baseline_logistic.joblib
│   └── ...
├── data/               # Training data (e.g. training_data.csv)
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── logistic.py
│   │   ├── gbm.py
│   │   └── calibration.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   ├── governance/
│   │   ├── bias_checks.py
│   │   └── stability.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   └── config.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_governance_explainability.ipynb
│
├── tests/
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_api.py
│
├── README.md
├── requirements.txt
└── pyproject.toml