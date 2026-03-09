#!/usr/bin/env python3
"""Training → artifact → API loop: train logistic model and save artifact for the API.

Generates synthetic data if needed, trains the model, prints AUC/KS, and saves
artifacts/baseline_logistic.joblib. The API (src.api.main) loads this artifact at startup.

Run from project root:
    python train_logistic.py              # use data/training_data.csv (or generate if missing)
    python train_logistic.py --synthetic  # generate synthetic data then train
    python train_logistic.py --synthetic-only --synthetic-n 10000   # only generate CSV

Equivalently:  python -m src.models.train_logistic [options]
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train_logistic import main

if __name__ == "__main__":
    main()
