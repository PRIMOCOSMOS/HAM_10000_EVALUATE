"""
粉刺样开口检测
医学依据：粉刺样开口(comedo-like openings)是脂溢性角化病(bkl)的标志性结构，
         表现为棕黑色圆形暗点/暗孔，对应表皮内陷填充角蛋白。
BIP方法：形态学黑帽变换提取局部暗点 → 圆形度和尺寸筛选 →
         与色素网络网孔区分（粉刺样开口更大、更暗、更孤立）。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测粉刺样开口结构。

    返回:
        comedo_score: float (0-1)
        comedo_count: int, 检测到的开口数量
        comedo_density: float, 开口密度（数量/病灶面积*10000）
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return {"comedo_score": 0.0, "comedo_count": 0, "comedo_density": 0.0}

    # 转Lab取L通道
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)

    # 黑帽变换：提取比周围暗的小圆形结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(l_channel.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel)
    blackhat[~mask_bool] = 0

    # 自适应阈值
    valid = blackhat[mask_bool]
    if np.std(valid) < 1e-6:
        return {"comedo_score": 0.0, "comedo_count": 0, "comedo_density": 0.0}

    thresh = np.mean(valid) + 2.0 * np.std(valid)
    dark_spots = (blackhat > thresh).astype(np.uint8)

    # 连通域分析
    contours, _ = cv2.findContours(dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    comedo_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # 尺寸筛选：粉刺样开口比网络网孔更大，约10-150像素面积
        if area < 10 or area > 150:
            continue
        if perimeter < 1:
            continue

        # 圆形度筛选（允许略椭圆）
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.5:
            continue

        # 暗度验证：该区域应显著暗于周围
        spot_mask = np.zeros_like(dark_spots)
        cv2.drawContours(spot_mask, [cnt], -1, 1, cv2.FILLED)
        spot_mean = np.mean(l_channel[spot_mask > 0])

        dilated = cv2.dilate(spot_mask, np.ones((7, 7), np.uint8))
        surround = (dilated > 0) & (spot_mask == 0) & mask_bool
        if np.sum(surround) > 0:
            surround_mean = np.mean(l_channel[surround])
            if surround_mean - spot_mean < 15:
                continue

        comedo_count += 1

    comedo_density = comedo_count / lesion_area * 10000
    comedo_score = min(1.0, comedo_count / 8.0)

    return {
        "comedo_score": float(np.clip(comedo_score, 0, 1)),
        "comedo_count": int(comedo_count),
        "comedo_density": float(comedo_density),
    }