#!/usr/bin/env python3
"""
Train the baseline logistic regression risk model.

Loads data (synthetic by default), preprocesses, trains, calibrates, evaluates,
and saves the model artifact for the API. Run from project root:

    python train_baseline.py

Optional: pass a CSV path and target column for real data:

    python train_baseline.py --data path/to/data.csv --target target_col
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path when running as script
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np

from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    CALIBRATION_METHOD,
)
from src.data.loaders import load_csv, load_synthetic, generate_synthetic_risk_data
from src.data.synthetic_generator import SyntheticRiskDataGenerator
from src.data.preprocessing import train_test_split_data, scale_features
from src.models.logistic import LogisticRiskModel
from src.models.calibration import fit_calibrator, apply_calibration
from src.evaluation.metrics import evaluation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline logistic risk model")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to CSV (optional). If not set, use synthetic data.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Target column name when using CSV (default: target)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output path for model artifact (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        choices=["isotonic", "sigmoid"],
        default=CALIBRATION_METHOD,
        help="Calibration method (default: isotonic)",
    )
    parser.add_argument(
        "--legacy-synthetic",
        action="store_true",
        help="Use legacy load_synthetic (generic features) instead of schema-based risk data.",
    )
    args = parser.parse_args()

    # Load data
    if args.data is not None:
        if not args.data.exists():
            raise FileNotFoundError(f"Data file not found: {args.data}")
        X, y = load_csv(args.data, target_column=args.target)
        print(f"Loaded {len(X)} rows from {args.data}")
    elif args.legacy_synthetic:
        X, y = load_synthetic(random_state=RANDOM_STATE)
        print("Using legacy synthetic data (10k samples, 20 features)")
    else:
        gen = SyntheticRiskDataGenerator(
            n_samples=10_000,
            fraud_rate=0.03,
            random_state=RANDOM_STATE,
        )
        X, y = gen.generate_X_y()
        print("Using SyntheticRiskDataGenerator (10k samples, fraud_rate=3%)")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Train / calibration split (from train)
    X_tr, X_cal, y_tr, y_cal = train_test_split_data(
        X_train, y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    # Scale features (fit on train only)
    X_tr_s, X_cal_s, scaler = scale_features(X_tr, X_cal)
    _, X_test_s, _ = scale_features(X_test, X_test, fit_on=X_tr)

    # Train baseline model
    model = LogisticRiskModel()
    model.fit(X_tr_s, y_tr)
    print("Fitted LogisticRiskModel")

    # Calibration (placeholder step)
    proba_cal = model.predict_proba(X_cal_s)[:, 1]
    calibrator = fit_calibrator(proba_cal, y_cal.values, method=args.calibration)
    print(f"Fitted {args.calibration} calibration on validation set")

    # Evaluate on test set
    proba_test = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)
    summary_uncal = evaluation_summary(y_test.values, y_pred, proba_test)
    print("\n--- Test set (uncalibrated) ---")
    print(f"  AUC: {summary_uncal['auc']:.4f}  KS: {summary_uncal['ks']:.4f}")
    print(f"  Accuracy: {summary_uncal['accuracy']:.4f}  F1: {summary_uncal['f1']:.4f}")
    print("  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"  {summary_uncal['confusion_matrix'].tolist()}")

    proba_cal_test = apply_calibration(proba_test, calibrator)
    y_pred_cal = (proba_cal_test >= 0.5).astype(int)
    summary_cal = evaluation_summary(y_test.values, y_pred_cal, proba_cal_test)
    print("\n--- Test set (calibrated) ---")
    print(f"  AUC: {summary_cal['auc']:.4f}  KS: {summary_cal['ks']:.4f}")
    print(f"  Accuracy: {summary_cal['accuracy']:.4f}  F1: {summary_cal['f1']:.4f}")

    # Save artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "calibrator": calibrator,
        "calibration_method": args.calibration,
        "feature_names_in": model.feature_names_in_,
    }
    joblib.dump(artifact, args.output)
    print(f"\nSaved artifact to {args.output}")

    # Save a small JSON summary next to the model (optional)
    summary_path = args.output.with_suffix(".metrics.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "auc_uncalibrated": summary_uncal["auc"],
                "ks_uncalibrated": summary_uncal["ks"],
                "auc_calibrated": summary_cal["auc"],
                "ks_calibrated": summary_cal["ks"],
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {summary_path}")


if __name__ == "__main__":
    main()
