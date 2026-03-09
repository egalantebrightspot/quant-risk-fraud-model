#!/usr/bin/env python3
"""Generate a ROC curve image for the README. Run from project root: python scripts/generate_readme_roc.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Synthetic ROC-like curve for a plausible risk model (no training needed)
np.random.seed(42)
n = 500
y_true = np.random.binomial(1, 0.1, n)
y_proba = np.clip(0.05 + 0.4 * y_true + 0.3 * np.random.randn(n).cumsum() * 0.02 + np.random.rand(n) * 0.3, 0, 1)
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve — risk model")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = ROOT / "docs" / "roc_example.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
