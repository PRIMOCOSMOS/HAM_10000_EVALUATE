"""
粟粒样囊肿检测
医学依据：粟粒样囊肿(milia-like cysts)是脂溢性角化病(bkl)的标志性结构，
         表现为白色/黄白色圆形亮点，对应表皮内角质假囊肿。
BIP方法：形态学白帽变换提取局部亮点 → 圆形度和尺寸筛选 →
         亮度显著性验证。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测粟粒样囊肿结构。

    返回:
        milia_score: float (0-1)
        milia_count: int, 检测到的囊肿数量
        milia_density: float, 囊肿密度（数量/病灶面积*10000）
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return {"milia_score": 0.0, "milia_count": 0, "milia_density": 0.0}

    # 转Lab取L通道
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)

    # 白帽变换：提取比周围亮的小圆形结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(l_channel.astype(np.uint8), cv2.MORPH_TOPHAT, kernel)
    tophat[~mask_bool] = 0

    # 自适应阈值
    valid = tophat[mask_bool]
    thresh = np.mean(valid) + 2.0 * np.std(valid)
    bright_spots = (tophat > thresh).astype(np.uint8)

    # 连通域分析
    contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    milia_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # 尺寸筛选：对应0.1-0.5mm，假设图像分辨率下约5-80像素面积
        if area < 5 or area > 80:
            continue
        if perimeter < 1:
            continue

        # 圆形度筛选
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.6:
            continue

        # 亮度显著性验证：该区域平均亮度应显著高于周围
        spot_mask = np.zeros_like(bright_spots)
        cv2.drawContours(spot_mask, [cnt], -1, 1, cv2.FILLED)
        spot_mean = np.mean(l_channel[spot_mask > 0])

        # 周围环形区域
        dilated = cv2.dilate(spot_mask, np.ones((5, 5), np.uint8))
        surround = (dilated > 0) & (spot_mask == 0) & mask_bool
        if np.sum(surround) > 0:
            surround_mean = np.mean(l_channel[surround])
            if spot_mean - surround_mean < 10:
                continue

        milia_count += 1

    milia_density = milia_count / lesion_area * 10000
    milia_score = min(1.0, milia_count / 8.0)

    return {
        "milia_score": float(np.clip(milia_score, 0, 1)),
        "milia_count": int(milia_count),
        "milia_density": float(milia_density),
    }