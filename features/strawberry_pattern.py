"""
草莓样模式检测
医学依据：草莓样模式(strawberry pattern)是光化性角化病(akiec)的特征性模式，
         由红色背景（红色假网络）中突出的白色/黄色毛囊开口组成。
BIP方法：检测红色背景区域 → 在红色背景中寻找小的圆形白色/黄色点（毛囊口）→
         计算"红色背景中白点"的模式得分。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测草莓样模式。

    返回:
        strawberry_score: float (0-1)
        red_background_ratio: float, 红色背景占病灶面积比
        follicular_opening_count: int, 毛囊开口数量
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return {"strawberry_score": 0.0, "red_background_ratio": 0.0,
                "follicular_opening_count": 0}

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)

    # --- 红色/粉红色背景检测 ---
    # 红色/粉红色：H在两端或低饱和度红
    red_bg = (
        (((h <= 15) | (h >= 165)) & (s > 40) & (v > 80)) &
        mask_bool
    )
    red_background_ratio = np.sum(red_bg) / lesion_area

    # --- 在红色背景中检测白色/黄色毛囊开口 ---
    # 毛囊开口：小的高亮度、低饱和度圆形点
    # 先在红色区域内做白帽变换
    red_region_l = l_channel.copy()
    red_region_l[~red_bg] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(red_region_l.astype(np.uint8), cv2.MORPH_TOPHAT, kernel)

    # 阈值化
    valid = tophat[red_bg]
    if len(valid) == 0 or np.std(valid) < 1e-6:
        return {"strawberry_score": 0.0, "red_background_ratio": float(red_background_ratio),
                "follicular_opening_count": 0}

    thresh = np.mean(valid) + 1.5 * np.std(valid)
    bright_in_red = (tophat > thresh).astype(np.uint8)

    # 连通域筛选
    contours, _ = cv2.findContours(bright_in_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    follicular_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < 3 or area > 60:
            continue
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.5:
            follicular_count += 1

    # --- 综合评分 ---
    # 草莓样模式 = 红色背景充分 + 毛囊开口数量足够
    bg_score = min(1.0, red_background_ratio / 0.5)
    opening_score = min(1.0, follicular_count / 15.0)
    strawberry_score = bg_score * 0.4 + opening_score * 0.6

    return {
        "strawberry_score": float(np.clip(strawberry_score, 0, 1)),
        "red_background_ratio": float(red_background_ratio),
        "follicular_opening_count": int(follicular_count),
    }