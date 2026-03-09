"""Train the baseline logistic regression risk model.

Loads the synthetic (or any) dataset, splits train/test, trains the model,
evaluates (AUC, KS), and saves the artifact for the API. This is the first
complete modeling loop and unlocks evaluation, calibration, explainability,
and the API layer.

Run from project root:
    python -m src.models.train_logistic
    python -m src.models.train_logistic --data data/training_data.csv --target fraud_flag
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Union, cast

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_MODEL_PATH,
    LOGISTIC_MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    CALIBRATION_METHOD,
    DATA_DIR,
    MODEL_VERSION,
)
from src.data.loaders import load_csv
from src.data.preprocessing import train_test_split_data, scale_features
from src.data.schema import TARGET_FRAUD
from src.models.logistic import LogisticRiskModel
from src.models.calibration import fit_calibrator, apply_calibration
from src.evaluation.metrics import evaluation_summary


DEFAULT_DATA_PATH = DATA_DIR / "training_data.csv"


def train_logistic_model(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    model_path: Union[str, Path] = LOGISTIC_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> float:
    """Simple training loop: load CSV, split, fit, evaluate AUC, save model.

    Uses fraud_flag as target. No feature scaling or calibration; for the full
    pipeline use run_training() instead.
    """
    data_path = Path(data_path)
    model_path = Path(model_path)
    df = pd.read_csv(data_path)
    X = df.drop(TARGET_FRAUD, axis=1)
    y = df[TARGET_FRAUD]
    X_train, X_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        ),
    )
    model = LogisticRiskModel(random_state=random_state)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, probs[:, 1])
    print(f"AUC: {auc:.4f}")
    model.save(model_path)
    return float(auc)


def run_training(
    data_path: Path,
    target_column: str = TARGET_FRAUD,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
    output_path: Path = DEFAULT_MODEL_PATH,
    calibration_method: Literal["isotonic", "sigmoid"] = CALIBRATION_METHOD,
) -> dict:
    """Run the full training loop: load → split → train → evaluate → save.

    Args:
        data_path: Path to CSV with features and target.
        target_column: Name of the binary target column.
        test_size: Fraction of data for test set (e.g. 0.25).
        validation_size: Fraction of training data for calibration (e.g. 0.2).
        random_state: Seed for splits and model.
        output_path: Where to save the joblib artifact.
        calibration_method: 'isotonic' or 'sigmoid'.

    Returns:
        Dict with keys: summary_uncal, summary_cal, artifact_path, metrics_path.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    X, y = load_csv(data_path, target_column=target_column)

    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    X_tr, X_cal, y_tr, y_cal = train_test_split_data(
        X_train, y_train,
        test_size=validation_size,
        random_state=random_state,
    )

    X_tr_s, X_cal_s, scaler = scale_features(X_tr, X_cal)
    _, X_test_s, _ = scale_features(X_test, X_test, fit_on=X_tr)

    model = LogisticRiskModel(random_state=random_state)
    model.fit(X_tr_s, y_tr)

    proba_cal = model.predict_proba(X_cal_s)[:, 1]
    calibrator = fit_calibrator(proba_cal, y_cal.values, method=calibration_method)

    proba_test = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)
    summary_uncal = evaluation_summary(y_test.values, y_pred, proba_test)

    proba_cal_test = apply_calibration(proba_test, calibrator)
    y_pred_cal = (proba_cal_test >= 0.5).astype(int)
    summary_cal = evaluation_summary(y_test.values, y_pred_cal, proba_cal_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "calibrator": calibrator,
        "calibration_method": calibration_method,
        "feature_names_in": model.feature_names_in_,
    }
    joblib.dump(artifact, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "auc_uncalibrated": summary_uncal["auc"],
        "ks_uncalibrated": summary_uncal["ks"],
        "auc_calibrated": summary_cal["auc"],
        "ks_calibrated": summary_cal["ks"],
        "auc": summary_cal["auc"],
        "ks": summary_cal["ks"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "summary_uncal": summary_uncal,
        "summary_cal": summary_cal,
        "artifact_path": output_path,
        "metrics_path": metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline logistic risk model from dataset (e.g. synthetic CSV)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_FRAUD,
        help=f"Target column name (default: {TARGET_FRAUD})",
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
        "--simple",
        action="store_true",
        help="Use simple train_logistic_model() flow (no scaling/calibration, model.save).",
    )
    args = parser.parse_args()

    if args.simple:
        train_logistic_model(
            data_path=args.data,
            model_path=LOGISTIC_MODEL_PATH,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        return

    result = run_training(
        data_path=args.data,
        target_column=args.target,
        output_path=args.output,
        calibration_method=args.calibration,
    )

    summary_uncal = result["summary_uncal"]
    summary_cal = result["summary_cal"]

    print(f"Loaded data from {args.data}")
    print("Fitted LogisticRiskModel + calibration")
    print("\n--- Test set (uncalibrated) ---")
    print(f"  AUC: {summary_uncal['auc']:.4f}  KS: {summary_uncal['ks']:.4f}")
    print(f"  Accuracy: {summary_uncal['accuracy']:.4f}  F1: {summary_uncal['f1']:.4f}")
    print("  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"  {summary_uncal['confusion_matrix'].tolist()}")
    print("\n--- Test set (calibrated) ---")
    print(f"  AUC: {summary_cal['auc']:.4f}  KS: {summary_cal['ks']:.4f}")
    print(f"  Accuracy: {summary_cal['accuracy']:.4f}  F1: {summary_cal['f1']:.4f}")
    print(f"\nSaved artifact to {result['artifact_path']}")
    print(f"Saved metrics to {result['metrics_path']}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()
