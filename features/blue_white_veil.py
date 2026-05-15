"""
蓝白幕检测
医学依据：蓝白幕(Blue-White Veil)对应真皮层致密黑色素细胞+表皮纤维化，
         是七点检查法的主要标准之一(2分)，高度提示黑色素瘤。
BIP方法：在HSV空间筛选蓝白色调区域（H∈[200°,250°]对应OpenCV的[100,125]，
         中等饱和度，中等偏低明度），计算面积占比和分布模式。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测蓝白幕结构。

    返回:
        has_blue_white_veil: bool
        bwv_score: float (0-1)
        bwv_area_ratio: float, 蓝白区域占病灶面积比
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 蓝白幕颜色范围（OpenCV H: 0-180）
    # H∈[100,125] 对应色相200°-250°
    # S∈[25,130] 中等饱和度（不是纯蓝也不是纯白）
    # V∈[75,180] 中等偏低明度
    bwv_mask = (
        (h >= 100) & (h <= 125) &
        (s >= 25) & (s <= 130) &
        (v >= 75) & (v <= 180) &
        mask_bool
    )

    bwv_area = np.sum(bwv_mask)
    bwv_area_ratio = bwv_area / lesion_area

    # 排除蓝痣（蓝色均匀覆盖全病灶的情况）
    # 蓝白幕应是局部的，不应覆盖>80%
    is_focal = bwv_area_ratio < 0.80

    # 判定：面积>10%且为局灶性分布
    has_bwv = bwv_area_ratio > 0.10 and is_focal

    # 评分：面积比归一化，局灶性加权
    bwv_score = min(1.0, bwv_area_ratio / 0.4) * (1.0 if is_focal else 0.3)

    return {
        "has_blue_white_veil": bool(has_bwv),
        "bwv_score": float(np.clip(bwv_score, 0, 1)),
        "bwv_area_ratio": float(bwv_area_ratio),
    }


def _empty_result():
    return {
        "has_blue_white_veil": False,
        "bwv_score": 0.0,
        "bwv_area_ratio": 0.0,
    }