from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", index=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    _plot_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")
    return metrics


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("HAM10000 Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
