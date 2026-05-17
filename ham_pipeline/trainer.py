from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .balancing import apply_sampling
from .config import PipelineConfig
from .data import build_balanced_coverage_rounds, load_metadata, train_test_split_df
from .evaluate import evaluate_and_save
from .features import build_feature_columns, extract_features_for_row
from .model import build_classifier


def _extract_matrix(df: pd.DataFrame, config: PipelineConfig) -> np.ndarray:
    iterator = (row for _, row in df.iterrows())

    if config.n_jobs == 1:
        feats = [extract_features_for_row(row, config) for row in tqdm(iterator, total=len(df))]
    else:
        feats = Parallel(n_jobs=config.n_jobs, backend="loky")(
            delayed(extract_features_for_row)(row, config) for row in tqdm(iterator, total=len(df))
        )

    return np.vstack(feats).astype(np.float32)


def _fit_and_predict(
    config: PipelineConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[object, np.ndarray, list[str] | None]:
    model = build_classifier(config)

    label_classes: list[str] | None = None
    if config.model_name == "ann":
        label_encoder = LabelEncoder()
        y_fit = label_encoder.fit_transform(y_train)
        model.fit(X_train, y_fit)
        y_pred_encoded = model.predict(X_val).astype(int)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        label_classes = label_encoder.classes_.tolist()
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    return model, y_pred, label_classes


def _run_single_experiment(
    config: PipelineConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float]:
    train_df["dx"].value_counts().sort_index().rename_axis("class").to_frame("count").to_csv(
        output_dir / "class_distribution_train.csv"
    )
    val_df["dx"].value_counts().sort_index().rename_axis("class").to_frame("count").to_csv(
        output_dir / "class_distribution_val.csv"
    )

    feature_columns = build_feature_columns(config)
    X_train = _extract_matrix(train_df, config)
    X_val = _extract_matrix(val_df, config)
    y_train = train_df["dx"].to_numpy()
    y_val = val_df["dx"].to_numpy()

    X_train_bal, y_train_bal = apply_sampling(X_train, y_train, config)
    model, y_pred, label_classes = _fit_and_predict(config, X_train_bal, y_train_bal, X_val)

    labels = sorted(pd.concat([train_df["dx"], val_df["dx"]], axis=0).unique().tolist())
    metrics = evaluate_and_save(y_val, y_pred, labels, output_dir)

    joblib.dump(model, output_dir / "model.joblib")
    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    if label_classes is not None:
        with (output_dir / "label_classes.json").open("w", encoding="utf-8") as f:
            json.dump(label_classes, f, ensure_ascii=False, indent=2)

    return metrics


def run_training(config: PipelineConfig) -> None:
    output_dir: Path = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not any([config.include_lbp_roi, config.include_lbp_bimf1, config.include_glcm, config.include_hsv, config.include_abcd]):
        raise ValueError("At least one feature group must be enabled.")
    if config.split_mode == "stratified_ratio" and not (0.0 < config.val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1) when split_mode='stratified_ratio'.")

    df = load_metadata(config)

    if not config.coverage_enabled:
        train_df, val_df = train_test_split_df(config, df)
        metrics = _run_single_experiment(config, train_df, val_df, output_dir)

        with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2, default=str)

        print("Training complete")
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        return

    # Coverage mode: iterate balanced subsets until majority class samples are covered.
    round_dfs = build_balanced_coverage_rounds(config, df)
    if not round_dfs:
        raise ValueError("Coverage mode produced zero rounds.")

    rows: list[dict[str, float | int]] = []
    best_acc = -1.0
    best_round_dir: Path | None = None

    for round_idx, round_df in enumerate(round_dfs, start=1):
        round_dir = output_dir / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        train_df, val_df = train_test_split_df(config, round_df)
        metrics = _run_single_experiment(config, train_df, val_df, round_dir)

        row = {"round": round_idx, **metrics}
        rows.append(row)

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_round_dir = round_dir

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "coverage_metrics.csv", index=False)

    summary = {
        "rounds": len(round_dfs),
        "accuracy_mean": float(metrics_df["accuracy"].mean()),
        "accuracy_std": float(metrics_df["accuracy"].std(ddof=0)),
        "f1_macro_mean": float(metrics_df["f1_macro"].mean()),
        "best_round": int(metrics_df.loc[metrics_df["accuracy"].idxmax(), "round"]),
        "best_accuracy": float(metrics_df["accuracy"].max()),
    }
    with (output_dir / "coverage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if best_round_dir is not None:
        for name in ["model.joblib", "feature_columns.json", "run_config.json", "label_classes.json"]:
            src = best_round_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2, default=str)

    print("Coverage training complete")
    print(f"Rounds: {len(round_dfs)} | Mean acc: {summary['accuracy_mean']:.4f} | Best acc: {summary['best_accuracy']:.4f}")
