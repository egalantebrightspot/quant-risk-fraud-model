"""Train the GBM (LightGBM) risk model and save to artifacts for the API.

Uses the same data schema as the logistic pipeline. Run from project root:

    python -m src.models.train_gbm
    python -m src.models.train_gbm --data data/training_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, GBM_MODEL_PATH, RANDOM_STATE
from src.data.schema import CORE_FEATURES, TARGET_FRAUD
from src.models.gbm import GBMRiskModel


DEFAULT_DATA_PATH = DATA_DIR / "training_data.csv"


def train_gbm(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = GBM_MODEL_PATH,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> float:
    """Load CSV, split, fit GBM, evaluate AUC, save model for API."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}. Generate with generate_and_save() first.")
    df = pd.read_csv(data_path)
    for col in CORE_FEATURES + [TARGET_FRAUD]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}. Expected CORE_FEATURES + {TARGET_FRAUD}.")
    X = df[CORE_FEATURES]
    y = df[TARGET_FRAUD]
    X_train, X_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y),
    )
    model = GBMRiskModel(random_state=random_state)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"GBM AUC: {auc:.4f}")
    print(f"Saved to {output_path}")
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GBM risk model for API scoring.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to CSV")
    parser.add_argument("--output", type=Path, default=GBM_MODEL_PATH, help="Output joblib path")
    args = parser.parse_args()
    train_gbm(data_path=args.data, output_path=args.output)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()
