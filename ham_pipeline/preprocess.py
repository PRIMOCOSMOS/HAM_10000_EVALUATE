from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig


def read_and_resize(image_path: str, image_size: int) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)


def remove_hair_artifacts(bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (config.blackhat_kernel_size, config.blackhat_kernel_size),
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Telea inpainting propagates local texture into thin hair-occluded regions.
    cleaned = cv2.inpaint(bgr, mask, config.inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned


def illumination_correction_lab_clahe(bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_image(image_path: str, config: PipelineConfig) -> np.ndarray:
    image = read_and_resize(image_path, config.image_size)
    if not config.preprocessing_enabled:
        return image
    if config.enable_hair_removal:
        image = remove_hair_artifacts(image, config)
    if config.enable_clahe:
        image = illumination_correction_lab_clahe(image, config)
    return image
