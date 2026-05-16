"""
病灶分割模块（重构版）
核心改进：
1) 移除黑边干扰，降低伪病灶风险；
2) 多通道候选分割竞争，避免单阈值失效；
3) 失败时不再“全图回退”，改为保守中心掩膜并标记低质量。
"""

import cv2
import numpy as np


def segment_lesion(image_rgb: np.ndarray, morph_radius: int = 15) -> dict:
    """
    分割皮肤病灶区域。

    参数:
        image_rgb: 预处理后的RGB图像 (H, W, 3), uint8
        morph_radius: 形态学精修的结构元素半径

    返回:
        dict:
            mask: 二值掩膜 (H, W), uint8, 255=病灶区域
            contour: 最大轮廓点集, np.ndarray or None
            bbox: 边界框 (x, y, w, h)
            area: 病灶面积（像素数）
            quality_score: 分割质量分 (0-1)
            segmentation_failed: bool
    """
    h, w = image_rgb.shape[:2]
    work_img = _remove_dark_frame(image_rgb)

    lab = cv2.cvtColor(work_img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(work_img, cv2.COLOR_RGB2HSV)
    l_channel = lab[:, :, 0]
    v_channel = hsv[:, :, 2]

    blur_l = cv2.GaussianBlur(l_channel, (5, 5), 0)
    blur_v = cv2.GaussianBlur(v_channel, (5, 5), 0)

    _, bin_l = cv2.threshold(blur_l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin_v = cv2.threshold(blur_v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(5, morph_radius)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    cand_masks = []
    for b in (bin_l, bin_v):
        m = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = _fill_holes(m)
        m = _largest_component(m)
        cand_masks.append(m)

    best_mask, quality = _select_best_mask(cand_masks, (h, w))

    segmentation_failed = quality < 0.18
    if segmentation_failed:
        best_mask = _conservative_center_mask(work_img)
        quality = max(quality, 0.10)

    contour, bbox, area = _contour_bbox_area(best_mask)

    return {
        "mask": best_mask,
        "contour": contour,
        "bbox": bbox,
        "area": int(area),
        "quality_score": float(np.clip(quality, 0.0, 1.0)),
        "segmentation_failed": bool(segmentation_failed),
    }


def _remove_dark_frame(image_rgb: np.ndarray) -> np.ndarray:
    """简单黑边抑制：将近黑边框替换为邻近中值，减少阈值分割误导。"""
    img = image_rgb.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    border = np.zeros_like(gray, dtype=np.uint8)
    bw = max(6, int(min(h, w) * 0.03))
    border[:bw, :] = 1
    border[-bw:, :] = 1
    border[:, :bw] = 1
    border[:, -bw:] = 1

    dark_border = (gray < 20) & (border > 0)
    if np.sum(dark_border) < 0.25 * np.sum(border):
        return img

    inner = gray[bw:h - bw, bw:w - bw]
    if inner.size == 0:
        return img

    fill_val = int(np.median(inner))
    for c in range(3):
        channel = img[:, :, c]
        channel[dark_border] = fill_val
        img[:, :, c] = channel
    return img


def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """保留最大连通域。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return binary_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    out[labels == largest_idx] = 255
    return out


def _select_best_mask(cand_masks: list, shape_hw: tuple) -> tuple:
    """从候选掩膜中选质量最高者。"""
    best = None
    best_q = -1.0
    for m in cand_masks:
        q = _mask_quality(m, shape_hw)
        if q > best_q:
            best_q = q
            best = m
    if best is None:
        h, w = shape_hw
        best = np.zeros((h, w), dtype=np.uint8)
        best_q = 0.0
    return best, best_q


def _mask_quality(mask: np.ndarray, shape_hw: tuple) -> float:
    """基于面积比例、触边率、形状紧致度综合评分。"""
    h, w = shape_hw
    area = float(np.sum(mask > 0))
    total = float(h * w)
    if area <= 1:
        return 0.0

    area_ratio = area / max(total, 1.0)
    if area_ratio < 0.01 or area_ratio > 0.92:
        area_score = 0.0
    elif area_ratio < 0.03:
        area_score = 0.4
    elif area_ratio <= 0.65:
        area_score = 1.0
    else:
        area_score = 0.5

    edge = np.zeros_like(mask, dtype=np.uint8)
    bw = max(2, int(min(h, w) * 0.02))
    edge[:bw, :] = 1
    edge[-bw:, :] = 1
    edge[:, :bw] = 1
    edge[:, -bw:] = 1
    touch_ratio = np.sum((mask > 0) & (edge > 0)) / max(np.sum(mask > 0), 1)
    edge_score = 1.0 - min(1.0, touch_ratio / 0.35)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        compact_score = 0.0
    else:
        c = max(contours, key=cv2.contourArea)
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if p <= 1e-6:
            compact_score = 0.0
        else:
            circularity = 4.0 * np.pi * a / (p * p + 1e-10)
            compact_score = float(np.clip(circularity, 0.0, 1.0))

    return 0.50 * area_score + 0.35 * edge_score + 0.15 * compact_score


def _conservative_center_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    分割失败时的保守掩膜:
    取中心椭圆区域中偏暗像素，避免全图伪特征污染。
    """
    h, w = image_rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = h * 0.32, w * 0.32
    center_ellipse = (((yy - cy) / max(ry, 1.0)) ** 2 + ((xx - cx) / max(rx, 1.0)) ** 2) <= 1.0

    l_channel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
    t = np.percentile(l_channel, 55)
    dark_region = l_channel <= t

    mask = (center_ellipse & dark_region).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = _fill_holes(mask)
    mask = _largest_component(mask)

    if np.sum(mask > 0) < 0.01 * h * w:
        mask = center_ellipse.astype(np.uint8) * 255
    return mask


def _contour_bbox_area(mask: np.ndarray) -> tuple:
    """提取最大轮廓和几何信息。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape
        return None, (0, 0, w, h), int(np.sum(mask > 0))

    largest = max(contours, key=cv2.contourArea)
    bbox = cv2.boundingRect(largest)
    area = cv2.contourArea(largest)
    return largest, bbox, int(area)


def _fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    填充二值掩膜中的内部孔洞。
    利用 floodFill 从边界填充背景，取反后与原图合并。
    """
    h, w = binary_mask.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    im_floodfill = binary_mask.copy()
    cv2.floodFill(im_floodfill, flood_fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(im_floodfill)
    filled = binary_mask | holes
    return filled