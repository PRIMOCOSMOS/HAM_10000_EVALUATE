"""
颜色种类与分布分析
医学依据：良性痣通常1-2种颜色，黑色素瘤常有5-6种颜色。
         诊断性6色空间：白/红/浅棕/深棕/蓝灰/黑。
BIP方法：将病灶像素映射到预定义的6色空间，统计各色占比，
         计算颜色数量和颜色分布熵。
"""

import cv2
import numpy as np


# 6色空间在Lab中的近似范围 (L, a, b各通道的[min, max])
# OpenCV Lab: L∈[0,255], a∈[0,255](128=中性), b∈[0,255](128=中性)
COLOR_DEFINITIONS = {
    "white":       {"L": (180, 255), "a": (115, 145), "b": (115, 145)},
    "red":         {"L": (50, 180),  "a": (145, 220), "b": (130, 200)},
    "light_brown": {"L": (120, 190), "a": (130, 155), "b": (135, 175)},
    "dark_brown":  {"L": (40, 120),  "a": (125, 160), "b": (130, 170)},
    "blue_gray":   {"L": (50, 150),  "a": (115, 135), "b": (90, 120)},
    "black":       {"L": (0, 60),    "a": (110, 140), "b": (110, 140)},
}

# 占比超过此阈值才计入
COLOR_PRESENCE_THRESH = 0.05


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    分析病灶内的颜色种类和分布。

    返回:
        color_count: int, 存在的颜色种类数 (1-6)
        color_score: float (0-1), 颜色多样性得分（归一化到0-1）
        color_entropy: float, 颜色分布熵
        color_ratios: dict, 各颜色占比
    """
    mask_bool = mask > 0
    lesion_area = int(np.sum(mask_bool))
    if lesion_area == 0:
        return _empty_result()

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    l_vals = lab[:, :, 0][mask_bool].astype(np.float64)
    a_vals = lab[:, :, 1][mask_bool].astype(np.float64)
    b_vals = lab[:, :, 2][mask_bool].astype(np.float64)

    # 对每种颜色计算占比
    color_ratios = {}
    for name, ranges in COLOR_DEFINITIONS.items():
        l_min, l_max = ranges["L"]
        a_min, a_max = ranges["a"]
        b_min, b_max = ranges["b"]

        in_range = (
            (l_vals >= l_min) & (l_vals <= l_max) &
            (a_vals >= a_min) & (a_vals <= a_max) &
            (b_vals >= b_min) & (b_vals <= b_max)
        )
        color_ratios[name] = float(np.sum(in_range)) / lesion_area

    # 统计超过阈值的颜色数量
    color_count = sum(1 for r in color_ratios.values() if r > COLOR_PRESENCE_THRESH)
    color_count = max(color_count, 1)  # 至少1种

    # 颜色分布熵
    probs = np.array([r for r in color_ratios.values() if r > 0])
    if len(probs) > 0:
        probs = probs / probs.sum()  # 归一化为概率分布
        color_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    else:
        color_entropy = 0.0

    # 多样性得分：颜色数归一化到0-1
    color_score = min(1.0, (color_count - 1) / 5.0)

    return {
        "color_count": color_count,
        "color_score": float(color_score),
        "color_entropy": color_entropy,
        "color_ratios": color_ratios,
    }


def _empty_result():
    return {
        "color_count": 0,
        "color_score": 0.0,
        "color_entropy": 0.0,
        "color_ratios": {},
    }