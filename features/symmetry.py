"""
结构与颜色对称性评估
医学依据：良性痣(nv)通常双轴对称，黑色素瘤(mel)通常至少单轴不对称。
         Menzies方法将"对称性缺失"作为黑色素瘤的阴性特征之一。
BIP方法：沿主轴和次轴分别翻转病灶，计算翻转前后的结构相似度（Dice系数），
         分别对形态和颜色分布做对称性评估。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    评估病灶的形态和颜色对称性。

    返回:
        symmetry_score: float (0-2), 0=双轴对称, 1=单轴不对称, 2=双轴不对称
        shape_symmetry: float (0-1), 形态对称性（1=完美对称）
        color_symmetry: float (0-1), 颜色分布对称性（1=完美对称）
    """
    mask_bool = mask > 0
    if np.sum(mask_bool) == 0:
        return {"symmetry_score": 1.0, "shape_symmetry": 0.5, "color_symmetry": 0.5}

    # 获取病灶最小外接矩形确定主轴方向
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return {"symmetry_score": 1.0, "shape_symmetry": 0.5, "color_symmetry": 0.5}

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    center, (w, h), angle = rect

    # 将病灶旋转到主轴水平
    rows, cols = mask.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    mask_rotated = cv2.warpAffine(mask, rot_matrix, (cols, rows), flags=cv2.INTER_NEAREST)
    img_rotated = cv2.warpAffine(image_rgb, rot_matrix, (cols, rows))

    # 裁剪到病灶区域
    cx, cy = int(center[0]), int(center[1])
    half_size = int(max(w, h) / 2) + 10
    y1, y2 = max(0, cy - half_size), min(rows, cy + half_size)
    x1, x2 = max(0, cx - half_size), min(cols, cx + half_size)

    mask_crop = mask_rotated[y1:y2, x1:x2]
    img_crop = img_rotated[y1:y2, x1:x2]

    if mask_crop.size == 0 or img_crop.size == 0:
        return {"symmetry_score": 1.0, "shape_symmetry": 0.5, "color_symmetry": 0.5}

    # 主轴对称性（左右翻转, axis=1）
    shape_sym_major = _flip_similarity(mask_crop, axis=1)
    color_sym_major = _color_flip_similarity(img_crop, mask_crop, axis=1)

    # 次轴对称性（上下翻转, axis=0）
    shape_sym_minor = _flip_similarity(mask_crop, axis=0)
    color_sym_minor = _color_flip_similarity(img_crop, mask_crop, axis=0)

    # 综合
    shape_symmetry = (shape_sym_major + shape_sym_minor) / 2.0
    color_symmetry = (color_sym_major + color_sym_minor) / 2.0

    # 不对称评分：每轴综合对称性<0.7视为不对称，贡献1分
    asym_major = 1.0 if (shape_sym_major + color_sym_major) / 2 < 0.7 else 0.0
    asym_minor = 1.0 if (shape_sym_minor + color_sym_minor) / 2 < 0.7 else 0.0
    symmetry_score = asym_major + asym_minor

    return {
        "symmetry_score": float(symmetry_score),
        "shape_symmetry": float(np.clip(shape_symmetry, 0, 1)),
        "color_symmetry": float(np.clip(color_symmetry, 0, 1)),
    }


def _flip_similarity(binary_mask: np.ndarray, axis: int) -> float:
    """二值掩膜翻转后的Dice系数"""
    flipped = cv2.flip(binary_mask, axis)
    intersection = np.sum((binary_mask > 0) & (flipped > 0))
    total = np.sum(binary_mask > 0) + np.sum(flipped > 0)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def _color_flip_similarity(image: np.ndarray, mask: np.ndarray, axis: int) -> float:
    """颜色分布翻转后的归一化互相关"""
    img_float = image.astype(np.float64)
    flipped = cv2.flip(img_float, axis)
    mask_bool = mask > 0
    mask_flipped = cv2.flip(mask, axis) > 0

    overlap = mask_bool & mask_flipped
    if np.sum(overlap) < 100:
        return 0.5

    gray = np.mean(img_float, axis=2)
    gray_flipped = np.mean(flipped, axis=2)

    a = gray[overlap] - np.mean(gray[overlap])
    b = gray_flipped[overlap] - np.mean(gray_flipped[overlap])

    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if norm < 1e-6:
        return 1.0

    ncc = np.sum(a * b) / norm
    return float((ncc + 1.0) / 2.0)