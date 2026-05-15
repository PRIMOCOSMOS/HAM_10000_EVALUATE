"""
不规则点/球检测
医学依据：不规则分布的点(dots)和球(globules)对应不规则分布的黑色素细胞巢，
         大小不一+分布不规则提示黑色素瘤（七点检查法次要标准，1分）。
         大小均匀+分布规则提示良性痣。
BIP方法：LoG(Laplacian of Gaussian)多尺度斑点检测 →
         评估检测到的斑点大小变异性和空间分布均匀性。
"""

import cv2
import numpy as np
from scipy.ndimage import label


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测病灶内的点/球结构并评估其规则性。

    返回:
        has_irregular_dots: bool
        irregularity_score: float (0-1), 越高越不规则
        size_cv: float, 斑点大小的变异系数
        distribution_entropy: float, 空间分布熵
        dot_count: int
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return _empty_result()

    # 转灰度
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray[~mask_bool] = 0

    # --- 多尺度LoG斑点检测 ---
    scales = [2, 3, 4, 6, 8]  # sigma值，对应不同大小的点/球
    log_max = np.zeros_like(gray)
    scale_map = np.zeros_like(gray)

    for sigma in scales:
        # LoG近似：DoG (Difference of Gaussians)
        k = int(sigma * 6) | 1
        g1 = cv2.GaussianBlur(gray, (k, k), sigma)
        g2 = cv2.GaussianBlur(gray, (k, k), sigma * 1.6)
        dog = g1 - g2

        # 取绝对值（检测暗色和亮色斑点）
        dog_abs = np.abs(dog) * (sigma ** 2)  # 尺度归一化

        # 更新最大响应
        update = dog_abs > log_max
        log_max[update] = dog_abs[update]
        scale_map[update] = sigma

    log_max[~mask_bool] = 0

    # --- 阈值化提取斑点 ---
    valid_vals = log_max[mask_bool]
    if len(valid_vals) == 0 or np.std(valid_vals) < 1e-6:
        return _empty_result()

    thresh = np.mean(valid_vals) + 1.5 * np.std(valid_vals)
    blob_binary = (log_max > thresh).astype(np.uint8)
    blob_binary[~mask_bool] = 0

    # 连通域分析
    labeled, num_features = label(blob_binary)
    if num_features < 3:
        return _empty_result()

    # 提取每个斑点的属性
    sizes = []
    centroids = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        area = np.sum(component)
        # 过滤过小噪点和过大区域
        if area < 5 or area > lesion_area * 0.1:
            continue
        sizes.append(area)
        ys, xs = np.where(component)
        centroids.append((np.mean(xs), np.mean(ys)))

    dot_count = len(sizes)
    if dot_count < 3:
        return _empty_result()

    sizes = np.array(sizes, dtype=np.float64)
    centroids = np.array(centroids)

    # --- 大小变异系数 ---
    size_cv = float(np.std(sizes) / max(np.mean(sizes), 1e-6))

    # --- 空间分布均匀性（分象限统计） ---
    distribution_entropy = _spatial_entropy(centroids, mask_bool)

    # --- 综合不规则性评分 ---
    # size_cv高 + 分布熵低（集中）= 不规则
    size_irregularity = min(1.0, size_cv / 1.5)
    # 分布熵低意味着集中在某些区域（不均匀）
    max_entropy = np.log2(4)  # 4象限均匀分布的最大熵
    distribution_irregularity = 1.0 - min(1.0, distribution_entropy / max_entropy)

    irregularity_score = size_irregularity * 0.6 + distribution_irregularity * 0.4
    has_irregular = irregularity_score > 0.4

    return {
        "has_irregular_dots": bool(has_irregular),
        "irregularity_score": float(np.clip(irregularity_score, 0, 1)),
        "size_cv": float(size_cv),
        "distribution_entropy": float(distribution_entropy),
        "dot_count": int(dot_count),
    }


def _spatial_entropy(centroids: np.ndarray, mask_bool: np.ndarray) -> float:
    """计算斑点质心在4象限中的分布熵"""
    if len(centroids) == 0:
        return 0.0

    h, w = mask_bool.shape
    mid_x, mid_y = w / 2.0, h / 2.0

    # 统计4象限中的斑点数
    counts = np.zeros(4)
    for (x, y) in centroids:
        qi = int(x >= mid_x) + 2 * int(y >= mid_y)
        counts[qi] += 1

    # 计算熵
    total = np.sum(counts)
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _empty_result():
    return {
        "has_irregular_dots": False,
        "irregularity_score": 0.0,
        "size_cv": 0.0,
        "distribution_entropy": 0.0,
        "dot_count": 0,
    }