"""
红色腔隙检测（血管来源判定）
医学依据：血管瘤(vasc)的标志性结构是红色/蓝红色腔隙（lacunae），
         圆形或椭圆形、边界清晰的红色区域，对应真皮浅层扩张的血管腔。
BIP方法：HSV空间筛选红色/蓝红色区域 → 连通域分析 →
         圆形度和凸度筛选腔隙结构 → 统计数量和面积比。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测红色/蓝红色腔隙结构。

    返回:
        has_lacunae: bool
        lacunae_score: float (0-1)
        lacunae_count: int
        lacunae_area_ratio: float, 腔隙占病灶面积比
        dominant_color: str, 主要腔隙颜色 ('red'/'blue-red'/'dark')
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    # 转HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # --- 红色区域提取 ---
    # OpenCV中H范围0-180，红色在两端
    red_mask = (((h <= 10) | (h >= 170)) & (s > 75) & (v > 50))

    # --- 蓝红色区域提取 ---
    blue_red_mask = ((h >= 130) & (h <= 170) & (s > 50) & (v > 30))

    # 合并候选区域，限制在病灶内
    candidate = ((red_mask | blue_red_mask) & mask_bool).astype(np.uint8) * 255

    # 轻微形态学闭运算填补小间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    # --- 连通域分析 ---
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lacunae_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # 面积过滤：排除过小噪点和过大区域
        if area < 30 or area > lesion_area * 0.4:
            continue
        if perimeter < 1:
            continue

        # 圆形度
        circularity = 4 * np.pi * area / (perimeter ** 2)
        # 凸度
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexity = area / max(hull_area, 1)

        # 腔隙筛选条件：圆形度>0.5 且 凸度>0.75
        if circularity > 0.5 and convexity > 0.75:
            lacunae_list.append({
                "area": area,
                "circularity": circularity,
                "convexity": convexity,
            })

    lacunae_count = len(lacunae_list)
    lacunae_total_area = sum(l["area"] for l in lacunae_list)
    lacunae_area_ratio = lacunae_total_area / lesion_area

    # 判定主要颜色
    dominant_color = _determine_dominant_color(image_rgb, candidate, mask_bool)

    # 综合评分
    count_score = min(1.0, lacunae_count / 6.0)
    area_score = min(1.0, lacunae_area_ratio / 0.4)
    lacunae_score = count_score * 0.5 + area_score * 0.5

    has_lacunae = lacunae_count >= 3 and lacunae_area_ratio > 0.05

    return {
        "has_lacunae": bool(has_lacunae),
        "lacunae_score": float(np.clip(lacunae_score, 0, 1)),
        "lacunae_count": lacunae_count,
        "lacunae_area_ratio": float(lacunae_area_ratio),
        "dominant_color": dominant_color,
    }


def _determine_dominant_color(image_rgb, candidate_mask, mask_bool):
    """判断腔隙区域的主要颜色类型"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    region = (candidate_mask > 0) & mask_bool

    if np.sum(region) == 0:
        return "none"

    mean_h = np.mean(hsv[:, :, 0][region])
    mean_v = np.mean(hsv[:, :, 2][region])

    if mean_v < 80:
        return "dark"
    elif mean_h <= 10 or mean_h >= 170:
        return "red"
    else:
        return "blue-red"


def _empty_result():
    return {
        "has_lacunae": False,
        "lacunae_score": 0.0,
        "lacunae_count": 0,
        "lacunae_area_ratio": 0.0,
        "dominant_color": "none",
    }