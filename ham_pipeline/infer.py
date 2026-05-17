from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .config import PipelineConfig
from .features import extract_features_for_inference


def _load_run_config(artifacts_dir: Path) -> PipelineConfig:
    with (artifacts_dir / "run_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return PipelineConfig(**cfg)


def predict_one(artifacts_dir: Path, image_path: Path) -> dict[str, object]:
    config = _load_run_config(artifacts_dir)
    model = joblib.load(artifacts_dir / "model.joblib")
    label_classes_path = artifacts_dir / "label_classes.json"
    label_classes: list[str] | None = None
    if label_classes_path.exists():
        with label_classes_path.open("r", encoding="utf-8") as f:
            label_classes = json.load(f)

    x = extract_features_for_inference(image_path=str(image_path), config=config)
    x = np.expand_dims(x, axis=0)

    pred_raw = model.predict(x)[0]
    if label_classes is not None:
        pred_idx = int(pred_raw)
        pred_label = label_classes[pred_idx]
    else:
        pred_label = str(pred_raw)

    result: dict[str, object] = {"prediction": pred_label}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        if label_classes is not None:
            classes = label_classes
        else:
            classes = [str(k) for k in model.classes_.tolist()]
        result["probabilities"] = {str(k): float(v) for k, v in zip(classes, proba)}

    return result
