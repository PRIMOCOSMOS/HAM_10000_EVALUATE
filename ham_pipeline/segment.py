from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import chan_vese

from .config import PipelineConfig


def chan_vese_mask(bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    cv_result = chan_vese(
        gray,
        mu=0.2,
        lambda1=config.cv_lambda1,
        lambda2=config.cv_lambda2,
        tol=1e-3,
        max_num_iter=config.cv_iter,
        dt=0.5,
        init_level_set="checkerboard",
        extended_output=False,
    )
    mask = (cv_result > 0).astype(np.uint8)

    # Keep largest connected component as lesion candidate.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.ones_like(mask, dtype=np.uint8)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    lesion = (labels == largest).astype(np.uint8)

    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return lesion
