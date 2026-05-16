from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import Paths


def load_metadata(paths: Paths) -> pd.DataFrame:
    if not paths.metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata: {paths.metadata_csv}")

    df = pd.read_csv(paths.metadata_csv)
    required_cols = {"image_id", "dx"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Metadata must include columns: {sorted(required_cols)}")

    df = df[["image_id", "dx"]].copy()
    df["image_path"] = df["image_id"].map(lambda image_id: _resolve_image_path(paths.image_dirs, image_id))
    if df["image_path"].isna().any():
        missing = df[df["image_path"].isna()]["image_id"].head(20).tolist()
        raise FileNotFoundError(f"Missing image files for ids (sample): {missing}")

    return df


def _resolve_image_path(image_dirs: tuple[Path, ...], image_id: str) -> str | None:
    filename = f"{image_id}.jpg"
    for folder in image_dirs:
        path = folder / filename
        if path.exists():
            return str(path)
    return None
