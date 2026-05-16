from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ExperimentConfig, Paths
from reproducer.cache import create_memory
from reproducer.data import load_metadata
from reproducer.features import extract_handcrafted_features
from reproducer.metrics import calculate_metrics, save_confusion_plot, save_metrics
from reproducer.models import build_models
from reproducer.sampling import apply_sampling


def run_reproduction(paths: Paths, exp: ExperimentConfig) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    memory = create_memory(paths.cache_dir)

    metadata = load_metadata(paths)
    metadata["dx"].value_counts().sort_index().to_csv(paths.output_dir / "class_distribution_original.csv")

    x, y = extract_handcrafted_features(metadata, exp, paths.cache_dir / "features")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=exp.test_size,
        random_state=exp.random_state,
        stratify=y,
    )

    models = build_models(random_state=exp.random_state, n_jobs=exp.n_jobs)
    labels = sorted(np.unique(y).tolist())
    leaderboard_rows: list[dict[str, float | str]] = []

    cached_sampling = memory.cache(apply_sampling)

    for strategy in exp.sampling_strategies:
        sx, sy = cached_sampling(x_train, y_train, strategy, exp.random_state)
        _save_distribution(sy, paths.output_dir / f"class_distribution_{strategy}.csv")

        for name, model in models.items():
            run_dir = paths.output_dir / strategy / name
            run_dir.mkdir(parents=True, exist_ok=True)

            model.fit(sx, sy)
            pred = model.predict(x_test)

            metrics = calculate_metrics(y_test, pred)
            save_metrics(metrics, run_dir / "metrics.json")
            save_confusion_plot(y_test, pred, labels=labels, path=run_dir / "confusion_matrix.png")
            joblib.dump(model, run_dir / "model.joblib")

            leaderboard_rows.append(
                {
                    "sampling": strategy,
                    "model": name,
                    "accuracy": float(metrics["accuracy"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "precision_macro": float(metrics["precision_macro"]),
                    "recall_macro": float(metrics["recall_macro"]),
                }
            )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by=["sampling", "accuracy"], ascending=[True, False]
    )
    leaderboard.to_csv(paths.output_dir / "leaderboard.csv", index=False)


def _save_distribution(labels: np.ndarray, path: Path) -> None:
    values, counts = np.unique(labels, return_counts=True)
    rows = ["label,count"] + [f"{l},{c}" for l, c in zip(values.tolist(), counts.tolist())]
    path.write_text("\n".join(rows), encoding="utf-8")
