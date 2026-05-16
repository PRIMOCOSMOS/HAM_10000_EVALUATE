"""
GLCM纹理特征
用于传统方法中的多特征融合（GCLM/GLCM + SVM）。
"""

import cv2
import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
except Exception:
    graycomatrix = None
    graycoprops = None


def compute(image_rgb: np.ndarray, mask: np.ndarray, levels: int = 32, **kwargs) -> dict:
    """
    返回:
        contrast, dissimilarity, homogeneity, energy, correlation, asm, entropy
        texture_irregularity: 异质性综合分数(0-1)
    """
    mask_bool = mask > 0
    lesion_area = int(np.sum(mask_bool))
    if lesion_area == 0:
        return _empty_result()

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if graycomatrix is None or graycoprops is None:
        # skimage不可用时降级
        vals = gray[mask_bool].astype(np.float64)
        if vals.size == 0:
            return _empty_result()
        std_norm = np.clip(np.std(vals) / 64.0, 0.0, 1.0)
        return {
            "contrast": float(std_norm),
            "dissimilarity": float(std_norm),
            "homogeneity": float(1.0 - std_norm),
            "energy": float(max(0.0, 1.0 - std_norm)),
            "correlation": 0.0,
            "asm": float(max(0.0, 1.0 - std_norm)),
            "entropy": float(std_norm),
            "texture_irregularity": float(std_norm),
        }

    # 仅病灶区域参与
    q = np.floor(gray.astype(np.float64) / (256.0 / levels)).astype(np.uint8)
    q = np.clip(q, 0, levels - 1)

    # 将病灶外设为0不会破坏统计，因为后续只在bbox内计算并结合mask过滤
    q_masked = q.copy()
    q_masked[~mask_bool] = 0

    ys, xs = np.where(mask_bool)
    y1, y2 = np.min(ys), np.max(ys) + 1
    x1, x2 = np.min(xs), np.max(xs) + 1

    roi = q_masked[y1:y2, x1:x2]
    roi_mask = mask_bool[y1:y2, x1:x2]

    if roi.size == 0 or np.sum(roi_mask) < 32:
        return _empty_result()

    # 病灶外像素统一填充为病灶中位值，减少边界伪纹理
    med = int(np.median(roi[roi_mask]))
    roi_filled = roi.copy()
    roi_filled[~roi_mask] = med

    distances = [1, 2, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        roi_filled,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    asm = float(np.mean(graycoprops(glcm, "ASM")))

    # 熵
    p = glcm.astype(np.float64)
    p = p / np.sum(p)
    entropy = float(-np.sum(p * np.log2(p + 1e-12)))

    # 异质性综合分
    contrast_n = np.clip(contrast / 8.0, 0.0, 1.0)
    dissim_n = np.clip(dissimilarity / 3.0, 0.0, 1.0)
    entropy_n = np.clip(entropy / 8.0, 0.0, 1.0)
    homo_inv = 1.0 - np.clip(homogeneity, 0.0, 1.0)
    texture_irregularity = float(np.clip(
        0.30 * contrast_n + 0.25 * dissim_n + 0.25 * entropy_n + 0.20 * homo_inv, 0.0, 1.0
    ))

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "asm": asm,
        "entropy": entropy,
        "texture_irregularity": texture_irregularity,
    }


def _empty_result() -> dict:
    return {
        "contrast": 0.0,
        "dissimilarity": 0.0,
        "homogeneity": 0.0,
        "energy": 0.0,
        "correlation": 0.0,
        "asm": 0.0,
        "entropy": 0.0,
        "texture_irregularity": 0.0,
    }