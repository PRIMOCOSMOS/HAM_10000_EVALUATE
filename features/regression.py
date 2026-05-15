"""
回归结构检测
医学依据：回归结构包括白色瘢痕样区域（纤维化）和蓝灰色胡椒粒样颗粒（噬黑素细胞），
         是七点检查法的次要标准(1分)，提示黑色素瘤部分自发消退。
BIP方法：白色瘢痕区检测（高L低S无网络区域）+ 蓝灰颗粒检测（细小蓝灰点），
         两者共存时判定为回归结构。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测回归结构（白色瘢痕 + 蓝灰颗粒）。

    返回:
        has_regression: bool
        regression_score: float (0-1)
        scar_area_ratio: float, 白色瘢痕区占病灶面积比
        peppering_score: float, 蓝灰颗粒强度得分
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    l_channel = lab[:, :, 0].astype(np.float64)
    a_channel = lab[:, :, 1].astype(np.float64)
    b_channel = lab[:, :, 2].astype(np.float64)
    s_channel = hsv[:, :, 1].astype(np.float64) / 255.0

    # --- 白色瘢痕区检测 ---
    lesion_l_mean = np.mean(l_channel[mask_bool])
    lesion_l_std = np.std(l_channel[mask_bool])

    # 高亮度 + 低饱和度 + 在病灶内部（非边缘）
    scar_mask = (
        (l_channel > lesion_l_mean + 1.0 * lesion_l_std) &
        (s_channel < 0.12) &
        mask_bool
    )

    # 排除病灶边缘的白色（可能是正常皮肤渗入）
    eroded = cv2.erode(mask, np.ones((11, 11), np.uint8))
    scar_mask = scar_mask & (eroded > 0)

    scar_area = np.sum(scar_mask)
    scar_area_ratio = scar_area / lesion_area

    # --- 蓝灰颗粒（peppering）检测 ---
    # 蓝灰色：a接近中性(128)，b偏低(<120)，L中等
    blue_gray_mask = (
        (l_channel > 60) & (l_channel < 160) &
        (a_channel > 120) & (a_channel < 138) &
        (b_channel > 90) & (b_channel < 122) &
        mask_bool
    )

    # 检测细小点状结构（形态学白帽变换提取小亮点在蓝灰区域中）
    # 这里用蓝灰区域的面积和碎片化程度来评估
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue_gray_uint8 = blue_gray_mask.astype(np.uint8) * 255

    # 开运算去除大块连通区域，保留细小颗粒
    opened = cv2.morphologyEx(blue_gray_uint8, cv2.MORPH_OPEN, kernel_small)
    # 原图减去开运算结果 = 细小颗粒
    granules = blue_gray_uint8 - opened
    granule_area = np.sum(granules > 0)

    peppering_score = min(1.0, granule_area / (lesion_area * 0.05))

    # --- 综合判定 ---
    # 回归结构 = 白色瘢痕 + 蓝灰颗粒 共存
    has_scar = scar_area_ratio > 0.05
    has_peppering = peppering_score > 0.2

    regression_score = (
        min(1.0, scar_area_ratio / 0.15) * 0.5 +
        peppering_score * 0.5
    )

    has_regression = has_scar and has_peppering

    return {
        "has_regression": bool(has_regression),
        "regression_score": float(np.clip(regression_score, 0, 1)),
        "scar_area_ratio": float(scar_area_ratio),
        "peppering_score": float(peppering_score),
    }


def _empty_result():
    return {
        "has_regression": False,
        "regression_score": 0.0,
        "scar_area_ratio": 0.0,
        "peppering_score": 0.0,
    }