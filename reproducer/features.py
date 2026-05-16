from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from config import ExperimentConfig
from reproducer.cache import stable_hash


def extract_handcrafted_features(
    df: pd.DataFrame,
    exp: ExperimentConfig,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = stable_hash(
        {
            "image_size": exp.image_size,
            "num_rows": int(len(df)),
            "ids_head": df["image_id"].head(10).tolist(),
            "ids_tail": df["image_id"].tail(10).tolist(),
        }
    )
    feat_file = cache_dir / f"features_{key}.npz"
    label_file = cache_dir / f"labels_{key}.npy"

    if feat_file.exists() and label_file.exists():
        with np.load(feat_file) as packed:
            features = packed["x"]
        labels = np.load(label_file)
        return features, labels

    rows = list(df.itertuples(index=False))
    vectors = Parallel(n_jobs=exp.n_jobs, prefer="processes")(
        delayed(_extract_one)(Path(r.image_path), exp.image_size)
        for r in tqdm(rows, total=len(rows), desc="Extracting features")
    )
    x = np.vstack(vectors).astype(np.float32)
    y = df["dx"].to_numpy()

    np.savez_compressed(feat_file, x=x)
    np.save(label_file, y)
    return x, y


def _extract_one(path: Path, image_size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 论文是传统机器学习路线，这里用稳定的颜色+纹理统计特征作为可复现输入。
    hist_rgb = _color_hist(rgb, bins=16)
    hist_hsv = _color_hist(hsv, bins=16)
    hist_lab = _color_hist(lab, bins=16)

    mean_std = np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1))], axis=0)

    edge = cv2.Canny(gray, 60, 160)
    edge_ratio = np.array([edge.mean() / 255.0], dtype=np.float32)

    lbp_like = _local_binary_like(gray)

    return np.concatenate([hist_rgb, hist_hsv, hist_lab, mean_std, edge_ratio, lbp_like], axis=0)


def _color_hist(img: np.ndarray, bins: int) -> np.ndarray:
    out = []
    for c in range(img.shape[2]):
        h = cv2.calcHist([img], [c], None, [bins], [0, 256]).flatten()
        h = h / (h.sum() + 1e-8)
        out.append(h)
    return np.concatenate(out, axis=0).astype(np.float32)


def _local_binary_like(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    mag_hist, _ = np.histogram(mag, bins=12, range=(0, 255), density=True)
    ang_hist, _ = np.histogram(ang, bins=12, range=(0, 360), density=True)
    return np.concatenate([mag_hist, ang_hist], axis=0).astype(np.float32)
