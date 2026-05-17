from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .config import PipelineConfig
from .preprocess import preprocess_image
from .segment import chan_vese_mask


def _safe_hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def _mean_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.blur(img, (ksize, ksize))


def _envelope_max_min(img: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    upper = cv2.dilate(img, kernel)
    lower = cv2.erode(img, kernel)
    return upper, lower


def _enforce_extrema_coincidence(
    source: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    local_max = source >= cv2.dilate(source, kernel)
    local_min = source <= cv2.erode(source, kernel)

    upper_adj = upper.copy()
    lower_adj = lower.copy()
    upper_adj[local_max] = source[local_max]
    lower_adj[local_min] = source[local_min]
    return upper_adj, lower_adj


def _downsample_by_power_of_two(img: np.ndarray, power: int) -> np.ndarray:
    if power <= 0:
        return img.copy()
    factor = 2**power
    h, w = img.shape
    new_h = max(1, h // factor)
    new_w = max(1, w // factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _upsample_to_shape(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def mremd_decompose(gray_u8: np.ndarray, config: PipelineConfig) -> tuple[list[np.ndarray], np.ndarray]:
    residual = gray_u8.astype(np.float32)
    bimfs: list[np.ndarray] = []

    for n in range(1, config.mremd_max_bimfs + 1):
        power = n - 1
        work = _downsample_by_power_of_two(residual, power)
        if min(work.shape) < config.mremd_window_size:
            break

        work = _mean_filter(work, config.mremd_smoothing_kernel)
        upper, lower = _envelope_max_min(work, config.mremd_window_size)

        upper = _mean_filter(upper, config.mremd_smoothing_kernel)
        lower = _mean_filter(lower, config.mremd_smoothing_kernel)
        upper, lower = _enforce_extrema_coincidence(work, upper, lower, config.mremd_window_size)

        mean_env = (upper + lower) * 0.5
        mean_env_up = _upsample_to_shape(mean_env, residual.shape)
        mean_env_up = _mean_filter(mean_env_up, config.mremd_smoothing_kernel)

        bimf = residual - mean_env_up
        bimfs.append(bimf)
        residual = mean_env_up

        if np.std(residual) < 1e-3:
            break

    return bimfs, residual


def _bimf_to_uint8(bimf: np.ndarray) -> np.ndarray:
    bimf_norm = cv2.normalize(bimf, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return bimf_norm.astype(np.uint8)


def _lbp_hist(
    gray_u8: np.ndarray,
    mask_u8: np.ndarray,
    points: int,
    radius: int,
    method: str,
    bins: int,
) -> np.ndarray:
    lbp = local_binary_pattern(gray_u8, P=points, R=radius, method=method)
    values = lbp[mask_u8 > 0]
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    return _safe_hist(values, bins=bins, value_range=(0, bins))


def _glcm_stats(gray_u8: np.ndarray, mask_u8: np.ndarray, config: PipelineConfig) -> np.ndarray:
    quant = (gray_u8 / 32).astype(np.uint8)
    fill_value = int(np.median(quant[mask_u8 > 0])) if np.any(mask_u8 > 0) else 0
    quant_masked = np.where(mask_u8 > 0, quant, fill_value).astype(np.uint8)

    glcm = graycomatrix(
        quant_masked,
        distances=list(config.glcm_distances),
        angles=list(config.glcm_angles),
        levels=8,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "homogeneity", "energy", "correlation"]
    stats = [graycoprops(glcm, prop).mean() for prop in props]
    return np.array(stats, dtype=np.float32)


def _hsv_histogram(bgr: np.ndarray, mask_u8: np.ndarray, bins: tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_vals = hsv[..., 0][mask_u8 > 0]
    s_vals = hsv[..., 1][mask_u8 > 0]
    v_vals = hsv[..., 2][mask_u8 > 0]

    h_hist = _safe_hist(h_vals, bins=bins[0], value_range=(0, 180))
    s_hist = _safe_hist(s_vals, bins=bins[1], value_range=(0, 256))
    v_hist = _safe_hist(v_vals, bins=bins[2], value_range=(0, 256))
    return np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)


def _abcd_features(mask_u8: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    mask = (mask_u8 > 0).astype(np.uint8)
    area = float(mask.sum())

    if area <= 1:
        return np.zeros(4, dtype=np.float32)

    h, w = mask.shape

    left = mask[:, : w // 2].astype(np.int16)
    right = np.fliplr(mask[:, w - w // 2 :]).astype(np.int16)
    min_w = min(left.shape[1], right.shape[1])
    asym_h = np.mean(np.abs(left[:, :min_w] - right[:, :min_w]))

    top = mask[: h // 2, :].astype(np.int16)
    bottom = np.flipud(mask[h - h // 2 :, :]).astype(np.int16)
    min_h = min(top.shape[0], bottom.shape[0])
    asym_v = np.mean(np.abs(top[:min_h, :] - bottom[:min_h, :]))
    asymmetry = float((asym_h + asym_v) / 2.0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([asymmetry, 0.0, 0.0, 0.0], dtype=np.float32)

    perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), closed=True))
    compactness = (perimeter**2) / (4.0 * np.pi * area + 1e-7)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    roi_pixels = rgb[mask > 0]
    color_var = float(np.std(roi_pixels, axis=0).mean()) if len(roi_pixels) > 0 else 0.0

    eq_diameter = float(np.sqrt((4.0 * area) / np.pi)) / max(h, w)
    return np.array([asymmetry, compactness, color_var, eq_diameter], dtype=np.float32)


def build_feature_columns(config: PipelineConfig) -> list[str]:
    cols: list[str] = []

    if config.include_lbp_roi:
        cols.extend([f"lbp_roi_{i}" for i in range(config.lbp_bins)])
    if config.include_lbp_bimf1:
        cols.extend([f"lbp_bimf1_{i}" for i in range(config.lbp_bins)])

    if config.include_glcm:
        cols.extend(["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"])

    if config.include_hsv:
        h_bins, s_bins, v_bins = config.hsv_bins
        cols.extend([f"hsv_h_{i}" for i in range(h_bins)])
        cols.extend([f"hsv_s_{i}" for i in range(s_bins)])
        cols.extend([f"hsv_v_{i}" for i in range(v_bins)])

    if config.include_abcd:
        cols.extend(["abcd_asymmetry", "abcd_border_irregularity", "abcd_color_var", "abcd_diameter"])

    return cols


def extract_features_for_row(row: pd.Series, config: PipelineConfig) -> np.ndarray:
    bgr = preprocess_image(str(row["image_path"]), config)
    mask = chan_vese_mask(bgr, config)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bimfs, _ = mremd_decompose(gray, config)
    bimf1_u8 = gray if not bimfs else _bimf_to_uint8(bimfs[0])

    blocks: list[np.ndarray] = []

    if config.include_lbp_roi:
        blocks.append(
            _lbp_hist(
                gray,
                mask,
                points=config.lbp_points,
                radius=config.lbp_radius,
                method=config.lbp_method,
                bins=config.lbp_bins,
            )
        )

    if config.include_lbp_bimf1:
        blocks.append(
            _lbp_hist(
                bimf1_u8,
                mask,
                points=config.lbp_points,
                radius=config.lbp_radius,
                method=config.lbp_method,
                bins=config.lbp_bins,
            )
        )

    if config.include_glcm:
        blocks.append(_glcm_stats(gray, mask, config))

    if config.include_hsv:
        blocks.append(_hsv_histogram(bgr, mask, config.hsv_bins))

    if config.include_abcd:
        blocks.append(_abcd_features(mask, bgr))

    if not blocks:
        raise ValueError("No feature blocks enabled. Enable at least one feature group in PipelineConfig.")

    return np.concatenate(blocks).astype(np.float32)


def extract_features_for_inference(image_path: str, config: PipelineConfig) -> np.ndarray:
    row = pd.Series({"image_path": image_path})
    return extract_features_for_row(row, config)
