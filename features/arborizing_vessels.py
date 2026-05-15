"""
树枝状血管检测
医学依据：树枝状血管(arborizing vessels)是基底细胞癌(bcc)最具特异性的结构(59%)，
         粗大的分支状血管，逐级变细，分支角度60-90°。
BIP方法：红色通道Frangi血管增强 → 骨架化 → 分支点检测 →
         分支角度和层次结构分析。
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import frangi


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测树枝状血管结构。

    返回:
        has_arborizing: bool
        arborizing_score: float (0-1)
        branching_factor: float, 分支因子（分支点数/骨架长度）
        mean_branch_angle: float, 平均分支角度（度）
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    # 提取红色通道并增强血管
    red = image_rgb[:, :, 0].astype(np.float64)
    green = image_rgb[:, :, 1].astype(np.float64)

    # 血管增强：红-绿差异
    vessel_img = np.clip(red - green * 0.8, 0, 255)
    vessel_img[~mask_bool] = 0

    # Frangi血管增强滤波
    vessel_img_norm = vessel_img / max(np.max(vessel_img), 1e-6)
    frangi_result = frangi(
        vessel_img_norm,
        sigmas=range(1, 5),
        black_ridges=False
    )
    frangi_result[~mask_bool] = 0

    # 阈值化
    valid = frangi_result[mask_bool]
    if np.max(valid) < 1e-8:
        return _empty_result()

    thresh = np.mean(valid) + 2.0 * np.std(valid)
    vessel_binary = ((frangi_result > thresh) & mask_bool).astype(np.uint8)

    # 形态学闭运算连接断裂血管
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_binary = cv2.morphologyEx(vessel_binary, cv2.MORPH_CLOSE, kernel)

    vessel_area = np.sum(vessel_binary)
    if vessel_area < 30:
        return _empty_result()

    # --- 骨架化和分支分析 ---
    skeleton = skeletonize(vessel_binary > 0).astype(np.uint8)
    skeleton_length = int(np.sum(skeleton))

    if skeleton_length < 20:
        return _empty_result()

    # 检测分支点（邻域连通数>=3）
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skeleton, -1, neighbor_kernel)
    branch_points = (skeleton > 0) & (neighbor_count >= 3)
    num_branches = int(np.sum(branch_points))

    # 分支因子：分支点数 / 骨架总长度 * 1000
    branching_factor = num_branches / skeleton_length * 1000.0

    # --- 分支角度估计 ---
    mean_branch_angle = _estimate_branch_angles(skeleton, branch_points)

    # --- 血管粗细的层次性（逐级变细） ---
    # 用距离变换评估血管宽度分布
    dist = cv2.distanceTransform(vessel_binary, cv2.DIST_L2, 5)
    vessel_widths = dist[skeleton > 0]
    if len(vessel_widths) > 10:
        width_range = np.max(vessel_widths) - np.min(vessel_widths)
        hierarchy_score = min(1.0, width_range / 3.0)
    else:
        hierarchy_score = 0.0

    # --- 综合评分 ---
    # 树枝状血管 = 分支因子适中 + 角度在60-90° + 有层次性
    branch_score = min(1.0, branching_factor / 50.0)
    angle_score = 0.0
    if 45 < mean_branch_angle < 110:
        angle_score = 1.0 - abs(mean_branch_angle - 75) / 75.0
        angle_score = max(0.0, angle_score)

    arborizing_score = (
        branch_score * 0.4 +
        angle_score * 0.3 +
        hierarchy_score * 0.3
    )

    has_arborizing = arborizing_score > 0.35 and num_branches >= 3

    return {
        "has_arborizing": bool(has_arborizing),
        "arborizing_score": float(np.clip(arborizing_score, 0, 1)),
        "branching_factor": float(branching_factor),
        "mean_branch_angle": float(mean_branch_angle),
    }


def _estimate_branch_angles(skeleton: np.ndarray, branch_points: np.ndarray) -> float:
    """
    估计分支点处的平均分支角度。
    在每个分支点周围取小邻域，分析从该点出发的骨架方向。
    """
    bp_ys, bp_xs = np.where(branch_points)
    if len(bp_ys) == 0:
        return 0.0

    angles = []
    radius = 5
    h, w = skeleton.shape

    for by, bx in zip(bp_ys[:20], bp_xs[:20]):  # 最多分析20个分支点
        # 取分支点周围的小窗口
        y1 = max(0, by - radius)
        y2 = min(h, by + radius + 1)
        x1 = max(0, bx - radius)
        x2 = min(w, bx + radius + 1)

        patch = skeleton[y1:y2, x1:x2].copy()
        cy, cx = by - y1, bx - x1

        # 找到从分支点出发的各分支方向
        # 在圆周上采样骨架像素的角度
        branch_dirs = []
        for angle_deg in range(0, 360, 10):
            rad = np.radians(angle_deg)
            for r in range(2, radius + 1):
                py = int(cy + r * np.sin(rad))
                px = int(cx + r * np.cos(rad))
                if 0 <= py < patch.shape[0] and 0 <= px < patch.shape[1]:
                    if patch[py, px] > 0:
                        branch_dirs.append(angle_deg)
                        break

        # 计算相邻分支方向之间的角度差
        if len(branch_dirs) >= 2:
            branch_dirs_sorted = sorted(branch_dirs)
            for i in range(len(branch_dirs_sorted) - 1):
                diff = branch_dirs_sorted[i + 1] - branch_dirs_sorted[i]
                if 30 < diff < 180:
                    angles.append(diff)

    if len(angles) == 0:
        return 0.0

    return float(np.mean(angles))


def _empty_result():
    return {
        "has_arborizing": False,
        "arborizing_score": 0.0,
        "branching_factor": 0.0,
        "mean_branch_angle": 0.0,
    }