"""
颜色种类与分布分析（重构版）
改进点：
1) 以病灶周围皮肤作为个体化基线，降低设备/肤色偏差；
2) 使用“有效颜色”定义（占比+最小连通片），抑制噪声引起的假多色；
3) 多色阈值更保守，减少nv被误判为mel。
"""

import cv2
import numpy as np


# 6色中心（OpenCV Lab）
# 这些中心值是经验初始化，后续会做个体化平移校正。
COLOR_CENTERS = {
    "white":       np.array([210.0, 128.0, 128.0], dtype=np.float64),
    "red":         np.array([120.0, 170.0, 165.0], dtype=np.float64),
    "light_brown": np.array([160.0, 140.0, 155.0], dtype=np.float64),
    "dark_brown":  np.array([85.0, 138.0, 150.0], dtype=np.float64),
    "blue_gray":   np.array([95.0, 122.0, 108.0], dtype=np.float64),
    "black":       np.array([38.0, 128.0, 128.0], dtype=np.float64),
}

# 占比阈值：只有超过该比例才记为有效颜色
COLOR_PRESENCE_THRESH = 0.07


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

    img_smooth = cv2.GaussianBlur(image_rgb, (3, 3), 0)
    lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB).astype(np.float64)

    # 个体化校正：以病灶周围皮肤均值作为偏移参考
    skin_mean = _estimate_surrounding_skin_mean(lab, mask_bool)
    ref_skin = np.array([190.0, 128.0, 135.0], dtype=np.float64)
    offset = skin_mean - ref_skin

    centers = {k: v + offset for k, v in COLOR_CENTERS.items()}

    lesion_pixels = lab[mask_bool]
    labels = _assign_colors(lesion_pixels, centers)

    color_ratios = {}
    for cname in COLOR_CENTERS.keys():
        ratio = float(np.sum(labels == cname)) / max(lesion_area, 1)
        color_ratios[cname] = ratio

    # 连通片校验：去掉由零散噪声导致的“假颜色”
    effective = _effective_colors_from_components(labels, color_ratios, mask_bool, lesion_area)

    color_count = max(1, len(effective))
    probs = np.array([color_ratios[c] for c in COLOR_CENTERS.keys()], dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size > 0:
        probs = probs / np.sum(probs)
        color_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    else:
        color_entropy = 0.0

    color_score = min(1.0, (color_count - 1) / 5.0)

    return {
        "color_count": int(color_count),
        "color_score": float(color_score),
        "color_entropy": float(color_entropy),
        "color_ratios": color_ratios,
    }


def _assign_colors(lesion_pixels: np.ndarray, centers: dict) -> np.ndarray:
    """按最近中心分配颜色标签。"""
    names = list(centers.keys())
    center_mat = np.vstack([centers[n] for n in names])

    # 给予ab通道更高权重，降低光照对颜色分类的影响
    w = np.array([0.8, 1.1, 1.1], dtype=np.float64)
    x = lesion_pixels[:, None, :] * w[None, None, :]
    c = center_mat[None, :, :] * w[None, None, :]
    d2 = np.sum((x - c) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)

    return np.array([names[i] for i in idx])


def _effective_colors_from_components(labels: np.ndarray, ratios: dict, mask_bool: np.ndarray, lesion_area: int) -> list:
    """
    有效颜色判定:
    1) 占比超过阈值；
    2) 该颜色至少存在一个达到最小面积的连通片。
    """
    h, w = mask_bool.shape
    label_img = np.full((h, w), "", dtype=object)
    label_img[mask_bool] = labels

    min_component_area = max(12, int(0.008 * lesion_area))
    effective = []

    for cname, ratio in ratios.items():
        if ratio < COLOR_PRESENCE_THRESH:
            continue

        binary = (label_img == cname).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        has_valid_component = False
        for i in range(1, num):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a >= min_component_area:
                has_valid_component = True
                break

        if has_valid_component:
            effective.append(cname)

    return effective


def _estimate_surrounding_skin_mean(lab: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """
    估计病灶周围皮肤Lab均值，作为个体化白平衡基线。
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dil = cv2.dilate(mask_u8, k, iterations=1)
    ring = (dil > 0) & (~mask_bool)

    if np.sum(ring) < 200:
        h, w = mask_bool.shape
        border = np.zeros((h, w), dtype=bool)
        bw = max(8, int(min(h, w) * 0.04))
        border[:bw, :] = True
        border[-bw:, :] = True
        border[:, :bw] = True
        border[:, -bw:] = True
        ring = border & (~mask_bool)

    if np.sum(ring) == 0:
        return np.array([190.0, 128.0, 135.0], dtype=np.float64)

    vals = lab[ring]
    return np.median(vals, axis=0).astype(np.float64)


def _empty_result():
    return {
        "color_count": 0,
        "color_score": 0.0,
        "color_entropy": 0.0,
        "color_ratios": {},
    }