from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PipelineConfig


def _resolve_image_path(image_id: str, roots: tuple[Path, Path]) -> Path:
    filename = f"{image_id}.jpg"
    for root in roots:
        candidate = root / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image for id={image_id}")


def load_metadata(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.metadata_csv())
    df = df[["image_id", "dx"]].copy()

    roots = config.image_roots()
    df["image_path"] = df["image_id"].apply(lambda image_id: _resolve_image_path(image_id, roots))
    return df


def _limit_per_class(df: pd.DataFrame, limit: int | None, random_state: int) -> pd.DataFrame:
    if limit is None:
        return df.reset_index(drop=True)

    sampled = []
    for _, group in df.groupby("dx"):
        take_n = min(limit, len(group))
        sampled.append(group.sample(n=take_n, random_state=random_state))
    return pd.concat(sampled, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _strict_balanced_split(config: PipelineConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    required = config.train_per_class + config.val_per_class
    for label, group in df.groupby("dx"):
        if len(group) < required:
            raise ValueError(
                f"Class '{label}' has {len(group)} samples, but strict split requires {required}. "
                "Increase per-class-limit, reduce train/val-per-class, or use split_mode='stratified_ratio'."
            )

        picked = group.sample(n=required, random_state=config.random_state)
        train_parts.append(picked.iloc[: config.train_per_class])
        test_parts.append(picked.iloc[config.train_per_class : required])

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=config.random_state)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=config.random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_test_split_df(config: PipelineConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.split_mode == "paper_fixed_count":
        df = _limit_per_class(df, config.per_class_limit, config.random_state)
        return _strict_balanced_split(config, df)
    if config.split_mode != "stratified_ratio":
        raise ValueError(f"Unsupported split_mode={config.split_mode}")

    train_df, test_df = train_test_split(
        df,
        test_size=config.val_ratio,
        random_state=config.random_state,
        stratify=df["dx"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_balanced_coverage_rounds(config: PipelineConfig, df: pd.DataFrame) -> list[pd.DataFrame]:
    if config.per_class_limit is None:
        raise ValueError("coverage mode requires per_class_limit to be set")

    grouped = {label: group.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True) for label, group in df.groupby("dx")}
    rounds_per_class = {label: int(np.ceil(len(group) / config.per_class_limit)) for label, group in grouped.items()}
    auto_rounds = max(rounds_per_class.values())
    total_rounds = auto_rounds if config.coverage_max_rounds <= 0 else min(config.coverage_max_rounds, auto_rounds)

    round_dfs: list[pd.DataFrame] = []
    rng = np.random.default_rng(config.random_state)
    for round_idx in range(total_rounds):
        parts: list[pd.DataFrame] = []
        for _, group in grouped.items():
            start = round_idx * config.per_class_limit
            end = start + config.per_class_limit

            if start < len(group):
                chunk = group.iloc[start:end].copy()
                if len(chunk) < config.per_class_limit:
                    refill = group.sample(
                        n=config.per_class_limit - len(chunk),
                        replace=True,
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    )
                    chunk = pd.concat([chunk, refill], axis=0)
            else:
                chunk = group.sample(
                    n=config.per_class_limit,
                    replace=True,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )

            parts.append(chunk)

        round_df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=config.random_state + round_idx)
        round_dfs.append(round_df.reset_index(drop=True))

    return round_dfs
