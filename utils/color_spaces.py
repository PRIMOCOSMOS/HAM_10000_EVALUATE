"""
颜色空间转换与诊断性6色映射工具
提供RGB↔Lab↔HSV的便捷转换，以及将像素映射到皮肤镜诊断性6色空间的功能。
"""

import cv2
import numpy as np


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """RGB转Lab，返回float64格式 (L:0-255, a:0-255, b:0-255)"""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)


def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """RGB转HSV，返回float64格式 (H:0-180, S:0-255, V:0-255)"""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float64)


def lab_to_rgb(image_lab: np.ndarray) -> np.ndarray:
    """Lab转RGB"""
    lab_uint8 = np.clip(image_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)


# 诊断性6色空间定义（Lab范围）
SIX_COLOR_RANGES = {
    "white":       {"L": (180, 255), "a": (115, 145), "b": (115, 145)},
    "red":         {"L": (50, 180),  "a": (145, 220), "b": (130, 200)},
    "light_brown": {"L": (120, 190), "a": (130, 155), "b": (135, 175)},
    "dark_brown":  {"L": (40, 120),  "a": (125, 160), "b": (130, 170)},
    "blue_gray":   {"L": (50, 150),  "a": (115, 135), "b": (90, 120)},
    "black":       {"L": (0, 60),    "a": (110, 140), "b": (110, 140)},
}


def map_to_six_colors(image_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    将病灶像素映射到诊断性6色空间，返回各颜色占比。

    参数:
        image_rgb: RGB图像 (H, W, 3), uint8
        mask: 病灶掩膜 (H, W), uint8

    返回:
        dict: 各颜色名称 → 占比 (float)
    """
    mask_bool = mask > 0
    lesion_area = int(np.sum(mask_bool))
    if lesion_area == 0:
        return {name: 0.0 for name in SIX_COLOR_RANGES}

    lab = rgb_to_lab(image_rgb)
    l_vals = lab[:, :, 0][mask_bool]
    a_vals = lab[:, :, 1][mask_bool]
    b_vals = lab[:, :, 2][mask_bool]

    ratios = {}
    for name, ranges in SIX_COLOR_RANGES.items():
        l_min, l_max = ranges["L"]
        a_min, a_max = ranges["a"]
        b_min, b_max = ranges["b"]

        in_range = (
            (l_vals >= l_min) & (l_vals <= l_max) &
            (a_vals >= a_min) & (a_vals <= a_max) &
            (b_vals >= b_min) & (b_vals <= b_max)
        )
        ratios[name] = float(np.sum(in_range)) / lesion_area

    return ratios


def extract_channel(image_rgb: np.ndarray, space: str, channel: int) -> np.ndarray:
    """
    提取指定颜色空间的指定通道。

    参数:
        image_rgb: RGB图像
        space: 'lab', 'hsv', 'rgb'
        channel: 通道索引 (0, 1, 2)

    返回:
        单通道图像 float64
    """
    if space == "lab":
        converted = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    elif space == "hsv":
        converted = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    elif space == "rgb":
        converted = image_rgb
    else:
        raise ValueError(f"Unsupported color space: {space}")

    return converted[:, :, channel].astype(np.float64)