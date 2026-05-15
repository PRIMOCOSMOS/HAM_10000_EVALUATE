"""
条纹/假足检测
医学依据：条纹(streaks)和假足(pseudopods)是病灶边缘放射状生长的黑色素细胞巢，
         局灶性/不对称分布高度提示黑色素瘤（七点检查法次要标准，1分）。
         均匀分布于全周则提示Spitz痣（良性）。
BIP方法：对病灶边界做极坐标展开 → 在展开图中检测垂直于边界方向的短线段 →
         评估分布的不对称性。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测病灶边缘的放射状条纹/假足结构。

    返回:
        has_streaks: bool
        streak_score: float (0-1)
        streak_asymmetry: float (0-1), 条纹分布不对称性（1=极不对称）
        streak_count: int, 检测到的条纹段数
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    # 获取病灶轮廓和质心
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_result()

    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return _empty_result()

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # 转灰度并增强边缘区域的径向结构
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # 创建边界带掩膜（病灶边缘内外各10px的环形区域）
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    outer = cv2.dilate(mask, kernel_dilate)
    inner = cv2.erode(mask, kernel_erode)
    border_band = ((outer > 0) & ~(inner > 0)).astype(np.uint8)

    if np.sum(border_band) < 50:
        return _empty_result()

    # --- 极坐标展开边界带 ---
    polar_img, polar_mask = _polar_unwrap(gray, border_band, cx, cy)

    if polar_img is None:
        return _empty_result()

    # --- 在极坐标图中检测径向线段（垂直方向的边缘） ---
    # Sobel在水平方向（检测垂直结构，即径向条纹）
    sobel_x = cv2.Sobel(polar_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x[polar_mask == 0] = 0

    # 自适应阈值
    valid_vals = sobel_x[polar_mask > 0]
    if len(valid_vals) == 0:
        return _empty_result()

    thresh = np.mean(valid_vals) + 1.5 * np.std(valid_vals)
    streak_binary = (sobel_x > thresh).astype(np.uint8)

    # 按角度方向（水平轴）统计条纹强度分布
    num_sectors = 36  # 每10°一个扇区
    cols = polar_img.shape[1]
    sector_width = max(cols // num_sectors, 1)

    sector_scores = []
    for i in range(num_sectors):
        c_start = i * sector_width
        c_end = min((i + 1) * sector_width, cols)
        sector_region = streak_binary[:, c_start:c_end]
        mask_region = polar_mask[:, c_start:c_end]
        if np.sum(mask_region) > 0:
            sector_scores.append(np.sum(sector_region) / np.sum(mask_region))
        else:
            sector_scores.append(0.0)

    sector_scores = np.array(sector_scores)

    # 条纹计数：连续高分扇区视为一段条纹
    streak_thresh = np.mean(sector_scores) + np.std(sector_scores)
    is_streak_sector = sector_scores > streak_thresh
    streak_count = _count_runs(is_streak_sector)

    # 分布不对称性：上下半圆条纹强度差异
    half = num_sectors // 2
    upper_sum = np.sum(sector_scores[:half])
    lower_sum = np.sum(sector_scores[half:])
    total = upper_sum + lower_sum
    if total > 0:
        streak_asymmetry = abs(upper_sum - lower_sum) / total
    else:
        streak_asymmetry = 0.0

    # 综合评分
    count_norm = min(1.0, streak_count / 5.0)
    streak_score = count_norm * 0.5 + streak_asymmetry * 0.5

    has_streaks = streak_count >= 2 and streak_asymmetry > 0.3

    return {
        "has_streaks": bool(has_streaks),
        "streak_score": float(np.clip(streak_score, 0, 1)),
        "streak_asymmetry": float(streak_asymmetry),
        "streak_count": int(streak_count),
    }


def _polar_unwrap(gray: np.ndarray, band_mask: np.ndarray,
                  cx: int, cy: int, num_angles: int = 360, num_radii: int = 50):
    """将环形边界带展开为极坐标矩形图像"""
    h, w = gray.shape
    # 确定半径范围
    ys, xs = np.where(band_mask > 0)
    if len(xs) == 0:
        return None, None

    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r_min, r_max = np.min(dists), np.max(dists)
    if r_max - r_min < 5:
        return None, None

    radii = np.linspace(r_min, r_max, num_radii)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    polar_img = np.zeros((num_radii, num_angles), dtype=np.float64)
    polar_mask = np.zeros((num_radii, num_angles), dtype=np.uint8)

    for ri, r in enumerate(radii):
        for ai, a in enumerate(angles):
            x = int(cx + r * np.cos(a))
            y = int(cy + r * np.sin(a))
            if 0 <= x < w and 0 <= y < h:
                polar_img[ri, ai] = gray[y, x]
                polar_mask[ri, ai] = band_mask[y, x]

    return polar_img, polar_mask


def _count_runs(bool_array: np.ndarray) -> int:
    """统计布尔数组中连续True段的数量（环形，首尾相连）"""
    if len(bool_array) == 0:
        return 0
    # 环形处理：拼接首尾
    extended = np.concatenate([bool_array, [bool_array[0]]])
    transitions = np.diff(extended.astype(int))
    # 上升沿数量即为段数
    return int(np.sum(transitions == 1))


def _empty_result():
    return {
        "has_streaks": False,
        "streak_score": 0.0,
        "streak_asymmetry": 0.0,
        "streak_count": 0,
    }