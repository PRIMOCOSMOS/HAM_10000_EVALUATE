from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score


def predict_with_thresholds(probs: np.ndarray, labels: list[str], thresholds: dict[str, float]) -> np.ndarray:
    adjusted = probs.copy()
    for i, cls in enumerate(labels):
        thr = float(thresholds.get(cls, 0.5))
        adjusted[:, i] = probs[:, i] / max(thr, 1e-6)

    pred_idx = np.argmax(adjusted, axis=1)
    return np.array([labels[i] for i in pred_idx], dtype=object)


def search_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    minority_classes: tuple[str, ...],
    recall_floor: float,
    lo: float,
    hi: float,
    steps: int,
) -> dict[str, float]:
    grid = np.linspace(lo, hi, steps)
    best = {cls: 0.5 for cls in labels}
    order = [c for c in labels if c in minority_classes] + [c for c in labels if c not in minority_classes]

    for target in order:
        best_thr = best[target]
        best_obj = -1e9

        for thr in grid:
            trial = dict(best)
            trial[target] = float(thr)
            y_pred = predict_with_thresholds(probs, labels, trial)

            macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            rec = recall_score(y_true, y_pred, labels=[target], average=None, zero_division=0)[0]
            penalty = 0.0
            if target in minority_classes and rec < recall_floor:
                penalty = (recall_floor - rec)

            objective = float(macro_f1 - penalty)
            if objective > best_obj:
                best_obj = objective
                best_thr = float(thr)

        best[target] = best_thr

    return best