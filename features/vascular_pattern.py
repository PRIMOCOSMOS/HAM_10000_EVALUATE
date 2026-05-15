"""
血管形态分析
医学依据：不同病变有特征性血管模式：
         - 点状(dotted)血管 → akiec/mel
         - 线状不规则(linear irregular) → mel
         - 树枝状(arborizing) → bcc（由arborizing_vessels.py单独处理）
         - 多形性(polymorphous，多种形态混合) → mel（七点检查法主要标准，2分）
BIP方法：红色通道Frangi血管增强 → 骨架化 → 分析形态（点状vs线状）→
         评估多形性（多种形态共存程度）。
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    分析病灶内的血管形态模式。

    返回:
        polymorphism_score: float (0-1), 血管多形性得分
        vessel_density: float, 血管区域占病灶面积比
        morphology_type: str, 主要血管形态 ('dotted'/'linear'/'mixed'/'none')
        dotted_ratio: float, 点状血管占比
        linear_ratio: float, 线状血管占比
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    # 提取红色通道（血管在红色通道中最明显）
    red = image_rgb[:, :, 0].astype(np.float64)
    green = image_rgb[:, :, 1].astype(np.float64)

    # 血管增强：红色通道减去绿色通道突出血管
    vessel_enhanced = red - green
    vessel_enhanced[~mask_bool] = 0
    vessel_enhanced = np.clip(vessel_enhanced, 0, 255)

    # 自适应阈值提取血管候选区域
    valid = vessel_enhanced[mask_bool]
    if np.std(valid) < 1e-6:
        return _empty_result()

    thresh = np.mean(valid) + 1.5 * np.std(valid)
    vessel_binary = ((vessel_enhanced > thresh) & mask_bool).astype(np.uint8)

    # 形态学开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_binary = cv2.morphologyEx(vessel_binary, cv2.MORPH_OPEN, kernel)

    vessel_area = np.sum(vessel_binary)
    vessel_density = vessel_area / lesion_area

    if vessel_area < 30:
        return _empty_result()

    # --- 骨架化分析形态 ---
    skeleton = skeletonize(vessel_binary > 0).astype(np.uint8)

    # 区分点状 vs 线状：
    # 点状血管：连通域面积小且近圆形
    # 线状血管：连通域细长
    contours, _ = cv2.findContours(vessel_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dotted_count = 0
    linear_count = 0
    total_vessel_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        total_vessel_area += area
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        # 点状：圆形度高，面积小
        if circularity > 0.5 and area < 100:
            dotted_count += 1
        # 线状：圆形度低（细长）
        elif circularity < 0.3:
            linear_count += 1

    total_count = dotted_count + linear_count
    if total_count == 0:
        return _empty_result()

    dotted_ratio = dotted_count / total_count
    linear_ratio = linear_count / total_count

    # 多形性评分：两种形态都存在时得分高
    polymorphism_score = 2.0 * min(dotted_ratio, linear_ratio)
    polymorphism_score = min(1.0, polymorphism_score + vessel_density * 0.5)

    # 主要形态类型
    if dotted_ratio > 0.7:
        morphology_type = "dotted"
    elif linear_ratio > 0.7:
        morphology_type = "linear"
    elif total_count > 5:
        morphology_type = "mixed"
    else:
        morphology_type = "none"

    return {
        "polymorphism_score": float(np.clip(polymorphism_score, 0, 1)),
        "vessel_density": float(vessel_density),
        "morphology_type": morphology_type,
        "dotted_ratio": float(dotted_ratio),
        "linear_ratio": float(linear_ratio),
    }


def _empty_result():
    return {
        "polymorphism_score": 0.0,
        "vessel_density": 0.0,
        "morphology_type": "none",
        "dotted_ratio": 0.0,
        "linear_ratio": 0.0,
    }