"""
蓝灰色卵圆巢检测
医学依据：蓝灰色卵圆巢(blue-grey ovoid nests)是基底细胞癌(bcc)的特征结构(34%)，
         对应真皮内基底样细胞巢团，现为离散的蓝灰色椭圆形团块。
BIP方法：HSV/Lab空间筛选蓝灰色区域 → 连通域分析 →
         椭圆形态筛选（长短轴比<3，面积适中）→ 与蓝白幕区分（离散vs弥漫）。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测蓝灰色卵圆巢结构。

    返回:
        nest_score: float (0-1)
        nest_count: int, 检测到的卵圆巢数量
        nest_area_ratio: float, 卵圆巢总面积占病灶面积比
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return {"nest_score": 0.0, "nest_count": 0, "nest_area_ratio": 0.0}

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 蓝灰色范围（OpenCV H: 0-180）
    # H∈[100,130] 对应蓝色到蓝灰色
    # S∈[15,100] 低到中等饱和度（灰调）
    # V∈[50,140] 中等偏低明度
    blue_gray_mask = (
        (h >= 100) & (h <= 130) &
        (s >= 15) & (s <= 100) &
        (v >= 50) & (v <= 140) &
        mask_bool
    )

    blue_gray_uint8 = blue_gray_mask.astype(np.uint8) * 255

    # 形态学闭运算填补小间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_gray_uint8 = cv2.morphologyEx(blue_gray_uint8, cv2.MORPH_CLOSE, kernel)

    # 连通域分析
    contours, _ = cv2.findContours(blue_gray_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nest_count = 0
    nest_total_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 面积筛选：卵圆巢是中等大小的离散团块
        if area < 50 or area > lesion_area * 0.25:
            continue

        # 椭圆拟合检查形态
        if len(cnt) < 5:
            continue

        ellipse = cv2.fitEllipse(cnt)
        (ex, ey), (ma, MA), angle = ellipse

        # 长短轴比<3（近椭圆形）
        aspect_ratio = MA / max(ma, 1e-6)
        if aspect_ratio > 3.0:
            continue

        # 凸度检查（卵圆巢边界较规则）
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexity = area / max(hull_area, 1e-6)
        if convexity < 0.6:
            continue

        nest_count += 1
        nest_total_area += area

    nest_area_ratio = nest_total_area / lesion_area

    # 评分
    count_score = min(1.0, nest_count / 5.0)
    area_score = min(1.0, nest_area_ratio / 0.2)
    nest_score = count_score * 0.6 + area_score * 0.4

    return {
        "nest_score": float(np.clip(nest_score, 0, 1)),
        "nest_count": int(nest_count),
        "nest_area_ratio": float(nest_area_ratio),
    }