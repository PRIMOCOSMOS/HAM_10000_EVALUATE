"""
色素网络检测与规则性评估（优化版）
核心改进：
  1. 降低Gabor响应阈值（mean + 0.5σ → 更宽松）
  2. 增加基于Gabor响应强度的直接评分（不完全依赖拓扑分析）
  3. 增加纹理能量比作为辅助判据
  4. 对典型nv的淡色网络更敏感
"""

import cv2
import numpy as np
from skimage.filters import gabor
from skimage.morphology import skeletonize


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测色素网络并评估其规则性。

    返回:
        has_network: bool
        network_score: float (0-1)
        regularity_score: float (0-1)
        coverage: float
        gabor_strength: float, Gabor响应强度（归一化）
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)

    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    # --- 步骤1：Top-Hat变换增强网格线条 ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(l_channel.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel)
    tophat = tophat.astype(np.float64)
    tophat[~mask_bool] = 0

    # --- 步骤2：Gabor滤波器组 ---
    orientations = np.linspace(0, np.pi, 6, endpoint=False)
    frequencies = [0.08, 0.12, 0.16, 0.20]  # 扩展频率范围覆盖更多网络间距

    gabor_max = np.zeros_like(l_channel)
    for freq in frequencies:
        for theta in orientations:
            filt_real, _ = gabor(tophat, frequency=freq, theta=theta)
            gabor_max = np.maximum(gabor_max, np.abs(filt_real))

    gabor_max[~mask_bool] = 0

    # --- 步骤3：Gabor响应强度评分（新增，更鲁棒） ---
    values = gabor_max[mask_bool]
    gabor_mean = np.mean(values)
    gabor_std = np.std(values)
    # 归一化强度：相对于病灶内的响应分布
    # 典型有网络的病变，gabor_mean会显著高于无网络的
    gabor_strength = min(1.0, gabor_mean / 8.0)  # 经验归一化因子

    # --- 步骤4：二值化（降低阈值） ---
    # 改用 mean + 0.5σ（原来是 mean + 1σ，太严格）
    thresh = np.mean(values) + 0.5 * np.std(values)
    network_binary = ((gabor_max > thresh) & mask_bool).astype(np.uint8)

    # 网络覆盖率
    network_pixels = np.sum(network_binary)
    coverage = network_pixels / lesion_area

    # --- 步骤5：骨架化和拓扑分析（辅助） ---
    skeleton = skeletonize(network_binary > 0).astype(np.uint8)
    cross_points = _count_crossing_points(skeleton)
    cross_density = cross_points / max(lesion_area, 1) * 10000

    # --- 步骤6：综合网络存在性评分（重新设计权重） ---
    # 三个维度：Gabor强度(主) + 覆盖率(主) + 拓扑交叉点(辅)
    strength_component = min(1.0, gabor_strength / 0.5) * 0.4
    coverage_component = min(1.0, coverage / 0.20) * 0.4
    topo_component = min(1.0, cross_density / 3.0) * 0.2

    network_score = strength_component + coverage_component + topo_component
    network_score = float(np.clip(network_score, 0, 1))

    # 判定阈值降低：score > 0.25 即认为有网络
    has_network = network_score > 0.25

    # --- 步骤7：规则性评估 ---
    regularity_score = _assess_regularity(skeleton, network_binary, mask_bool)

    return {
        "has_network": bool(has_network),
        "network_score": network_score,
        "regularity_score": float(np.clip(regularity_score, 0, 1)),
        "coverage": float(coverage),
        "gabor_strength": float(gabor_strength),
    }


def _count_crossing_points(skeleton: np.ndarray) -> int:
    """统计骨架中的交叉点数量"""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    crossings = (skeleton > 0) & (neighbor_count >= 3)
    return int(np.sum(crossings))


def _assess_regularity(skeleton: np.ndarray, network_binary: np.ndarray,
                       mask_bool: np.ndarray) -> float:
    """评估网络规则性"""
    if np.sum(skeleton) < 30:
        # 网络少但存在时，给中等规则性分数（典型nv可能网络淡但规则）
        return 0.6

    dist = cv2.distanceTransform(network_binary, cv2.DIST_L2, 5)
    thickness_values = dist[skeleton > 0]

    if len(thickness_values) < 10:
        return 0.6

    thickness_cv = np.std(thickness_values) / max(np.mean(thickness_values), 1e-6)

    h, w = mask_bool.shape
    cy, cx = h // 2, w // 2
    quadrants = [
        skeleton[:cy, :cx], skeleton[:cy, cx:],
        skeleton[cy:, :cx], skeleton[cy:, cx:]
    ]
    densities = [np.sum(q) / max(q.size, 1) for q in quadrants]
    mean_density = np.mean(densities)
    density_cv = np.std(densities) / max(mean_density, 1e-6) if mean_density > 0 else 0

    regularity = 1.0 - min(1.0, (thickness_cv * 0.5 + density_cv * 0.5))
    return max(0.0, regularity)


def _empty_result() -> dict:
    return {
        "has_network": False,
        "network_score": 0.0,
        "regularity_score": 0.0,
        "coverage": 0.0,
        "gabor_strength": 0.0,
    }