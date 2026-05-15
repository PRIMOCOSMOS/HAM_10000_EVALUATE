"""
中央白色斑片 + 周围网络检测（皮肤纤维瘤判定）
医学依据：皮肤纤维瘤(df)最典型模式是"周围淡色素网络 + 中央白色瘢痕样斑片"，
         中央白色对应真皮乳头层纤维化，周围网络对应边缘色素沉着的表皮突。
BIP方法：距离变换做径向分区 → 中央区高亮低饱和度检测 →
         周围区淡网络验证 → 径向亮度梯度计算。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测中央白色斑片和周围淡色素网络的空间组合模式。

    返回:
        has_central_white_patch: bool
        has_peripheral_network: bool
        df_pattern_score: float (0-1)
        radial_gradient: float, 正值=中心亮于边缘
        white_area_ratio: float, 白色区占中央区面积比
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    # --- 步骤1：径向分区（距离变换） ---
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = np.max(dist)
    if max_dist < 1:
        return _empty_result()

    # 中央区：距离>50%最大距离的区域
    central_mask = dist > (max_dist * 0.5)
    # 周围区：病灶内但不在中央区
    peripheral_mask = mask_bool & (~central_mask)

    central_area = np.sum(central_mask)
    peripheral_area = np.sum(peripheral_mask)

    if central_area == 0 or peripheral_area == 0:
        return _empty_result()

    # --- 步骤2：中央白色斑片检测 ---
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1].astype(np.float64) / 255.0

    # 中央区内的高亮度、低饱和度像素
    lesion_l_mean = np.mean(l_channel[mask_bool])
    lesion_l_std = np.std(l_channel[mask_bool])

    white_condition = (
        (l_channel > lesion_l_mean + 1.2 * lesion_l_std) &
        (s_channel < 0.15) &
        central_mask
    )
    white_area = np.sum(white_condition)
    white_area_ratio = white_area / central_area

    has_central_white = white_area_ratio > 0.2

    # --- 步骤3：周围淡网络验证（简化版Gabor检测） ---
    peripheral_l = l_channel.copy()
    peripheral_l[~peripheral_mask] = 0

    # 黑帽变换检测周围区的细线结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(peripheral_l.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel)
    tophat[~peripheral_mask] = 0

    # 周围区网络强度
    peripheral_network_intensity = np.mean(tophat[peripheral_mask])
    has_peripheral_network = peripheral_network_intensity > 3.0

    # --- 步骤4：径向亮度梯度 ---
    central_mean_l = np.mean(l_channel[central_mask])
    peripheral_mean_l = np.mean(l_channel[peripheral_mask])
    radial_gradient = (central_mean_l - peripheral_mean_l) / max(lesion_l_std, 1e-6)

    # --- 步骤5：整体形态（圆形度） ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * cv2.contourArea(cnt) / (perimeter ** 2)

    # --- 综合评分 ---
    df_pattern_score = (
        0.3 * min(1.0, white_area_ratio / 0.4) +
        0.2 * min(1.0, peripheral_network_intensity / 8.0) +
        0.25 * min(1.0, max(0, radial_gradient) / 2.0) +
        0.25 * circularity
    )

    return {
        "has_central_white_patch": bool(has_central_white),
        "has_peripheral_network": bool(has_peripheral_network),
        "df_pattern_score": float(np.clip(df_pattern_score, 0, 1)),
        "radial_gradient": float(radial_gradient),
        "white_area_ratio": float(white_area_ratio),
    }


def _empty_result():
    return {
        "has_central_white_patch": False,
        "has_peripheral_network": False,
        "df_pattern_score": 0.0,
        "radial_gradient": 0.0,
        "white_area_ratio": 0.0,
    }