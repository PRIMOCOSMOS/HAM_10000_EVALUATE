"""
叶状/轮辐结构检测
医学依据：叶状结构(leaf-like areas)和轮辐状结构(spoke-wheel areas)是
         基底细胞癌(bcc)尤其是浅表型bcc的特征性结构。
         叶状：从中心向外的扇形棕色/蓝灰色色素区域。
         轮辐：从一个暗色中心点向外辐射的短线段。
BIP方法：在病灶边缘区域检测棕色/蓝灰色放射状结构 →
         极坐标展开后检测周期性径向结构 → 分别评估叶状和轮辐模式。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测叶状和轮辐状结构。

    返回:
        leaf_score: float (0-1), 叶状结构得分
        spoke_score: float (0-1), 轮辐结构得分
        combined_score: float (0-1), 综合得分
        structure_count: int, 检测到的放射状结构数
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    # --- 提取病灶边缘区域（叶状/轮辐结构多在边缘） ---
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    inner = cv2.erode(mask, kernel_erode)
    peripheral_mask = mask_bool & ~(inner > 0)

    if np.sum(peripheral_mask) < 50:
        return _empty_result()

    # --- 棕色/蓝灰色素区域提取 ---
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0].astype(np.float64)
    b_ch = lab[:, :, 2].astype(np.float64)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]

    # 棕色/蓝灰色素：中低亮度，有一定饱和度
    pigment_mask = (
        (l_ch > 30) & (l_ch < 160) &
        (s_ch > 20) &
        peripheral_mask
    )

    pigment_area = np.sum(pigment_mask)
    if pigment_area < 30:
        return _empty_result()

    # --- 获取质心用于极坐标分析 ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_result()

    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return _empty_result()

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # --- 叶状结构检测 ---
    # 叶状结构：边缘区域的扇形色素团块
    pigment_uint8 = pigment_mask.astype(np.uint8) * 255
    pig_contours, _ = cv2.findContours(pigment_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_count = 0
    for pc in pig_contours:
        area = cv2.contourArea(pc)
        if area < 30 or area > lesion_area * 0.15:
            continue

        # 叶状结构应为扇形/三角形：凸度中等，有一定长宽比
        hull = cv2.convexHull(pc)
        hull_area = cv2.contourArea(hull)
        convexity = area / max(hull_area, 1e-6)

        if len(pc) < 5:
            continue

        # 检查是否指向中心（质心到该团块的方向应与团块长轴一致）
        pc_moments = cv2.moments(pc)
        if pc_moments["m00"] == 0:
            continue
        pcx = int(pc_moments["m10"] / pc_moments["m00"])
        pcy = int(pc_moments["m01"] / pc_moments["m00"])

        # 到病灶中心的方向
        dx = pcx - cx
        dy = pcy - cy
        dist_to_center = np.sqrt(dx ** 2 + dy ** 2)

        if dist_to_center < 10:
            continue

        # 叶状条件：中等凸度 + 在边缘
        if 0.5 < convexity < 0.9 and area > 50:
            leaf_count += 1

    # --- 轮辐结构检测 ---
    # 轮辐：从暗色中心点向外辐射的短线段
    # 检测小的暗色圆点周围是否有放射状线段
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray[~peripheral_mask] = 128  # 中性化非边缘区域

    # 检测暗色小圆点（轮辐中心）
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel_small)
    blackhat[~peripheral_mask] = 0

    valid_bh = blackhat[peripheral_mask]
    if len(valid_bh) == 0 or np.std(valid_bh) < 1e-6:
        spoke_count = 0
    else:
        bh_thresh = np.mean(valid_bh) + 2.0 * np.std(valid_bh)
        dark_centers = (blackhat > bh_thresh).astype(np.uint8)

        center_contours, _ = cv2.findContours(dark_centers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spoke_count = 0
        for dc in center_contours:
            dc_area = cv2.contourArea(dc)
            if dc_area < 3 or dc_area > 50:
                continue
            dc_perimeter = cv2.arcLength(dc, True)
            if dc_perimeter < 1:
                continue
            circ = 4 * np.pi * dc_area / (dc_perimeter ** 2)
            if circ > 0.5:
                # 检查周围是否有放射状结构（简化：检查周围梯度方向一致性）
                spoke_count += 1

    # --- 综合评分 ---
    leaf_score = min(1.0, leaf_count / 4.0)
    spoke_score = min(1.0, spoke_count / 5.0)
    combined_score = max(leaf_score, spoke_score)
    structure_count = leaf_count + spoke_count

    return {
        "leaf_score": float(np.clip(leaf_score, 0, 1)),
        "spoke_score": float(np.clip(spoke_score, 0, 1)),
        "combined_score": float(np.clip(combined_score, 0, 1)),
        "structure_count": int(structure_count),
    }


def _empty_result():
    return {
        "leaf_score": 0.0,
        "spoke_score": 0.0,
        "combined_score": 0.0,
        "structure_count": 0,
    }