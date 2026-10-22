"""Draw the baseline vs improved model comparison from model/results.json.

Usage: python scripts/plot_comparison.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "model" / "results.json").read_text())

metrics = ["Accuracy", "Balanced accuracy", "Macro AUC"]
baseline = [r["baseline_cnn_accuracy"], r["baseline_cnn_balanced_accuracy"], r["baseline_cnn_auc"]]
improved = [r["test_accuracy"], r["balanced_accuracy"], r["macro_auc"]]

x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(8.5, 4.2))
b1 = ax.bar(x - 0.2, baseline, 0.38, label="Baseline CNN", color="#93c5fd")
b2 = ax.bar(x + 0.2, improved, 0.38, label="Improved CNN", color="#2563eb")
for bars in (b1, b2):
    ax.bar_label(bars, labels=[f"{v:.1%}" if i < 2 else f"{v:.3f}" for i, v in enumerate(bars.datavalues)],
                 padding=3, fontsize=10)
# Majority-class reference only applies to plain accuracy
ax.hlines(r["majority_baseline"], -0.45, 0.45, color="#374151", ls="--", lw=1.2)
ax.text(-0.43, r["majority_baseline"] - 0.06, f"always predict 'nv': {r['majority_baseline']:.1%}",
        color="#374151", fontsize=8.5, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.set_xticks(x, metrics); ax.set_ylim(0, 1.05); ax.set_ylabel("Score on test set")
ax.spines[["top", "right"]].set_visible(False); ax.legend(loc="upper left", frameon=False)
ax.set_title("Baseline vs improved CNN on the held-out test set")
plt.tight_layout()
plt.savefig(ROOT / "screenshots" / "model_comparison.png", dpi=110)
print("saved screenshots/model_comparison.png")
