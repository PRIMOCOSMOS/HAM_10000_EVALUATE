"""
脑回样裂隙检测
医学依据：脑回样裂隙(fissures and ridges)是脂溢性角化病(bkl)的特征结构，
         表皮乳头瘤样增生形成的沟回结构，类似大脑表面的沟和回。
BIP方法：多方向Gabor滤波检测平行曲线状结构 → 曲率分析区分脑回样（高曲率平行）
         与色素网络（低曲率交叉网格）。
"""

import cv2
import numpy as np
from skimage.filters import gabor


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    检测脑回样裂隙结构。

    返回:
        cerebriform_score: float (0-1)
        ridge_density: float, 沟回结构密度
        mean_curvature: float, 平均曲率（高=脑回样）
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return {"cerebriform_score": 0.0, "ridge_density": 0.0, "mean_curvature": 0.0}

    # 转灰度
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray[~mask_bool] = 0

    # --- Gabor滤波检测线状结构 ---
    orientations = np.linspace(0, np.pi, 8, endpoint=False)
    # 脑回样结构频率较低（间距较宽），用低频Gabor
    frequency = 0.08

    gabor_responses = []
    for theta in orientations:
        filt_real, _ = gabor(gray, frequency=frequency, theta=theta)
        gabor_responses.append(np.abs(filt_real))

    gabor_responses = np.array(gabor_responses)

    # 取每像素的最大方向响应和对应方向
    max_response = np.max(gabor_responses, axis=0)
    max_direction = np.argmax(gabor_responses, axis=0)

    max_response[~mask_bool] = 0

    # --- 提取沟回区域 ---
    valid = max_response[mask_bool]
    if np.std(valid) < 1e-6:
        return {"cerebriform_score": 0.0, "ridge_density": 0.0, "mean_curvature": 0.0}

    thresh = np.mean(valid) + 1.0 * np.std(valid)
    ridge_binary = ((max_response > thresh) & mask_bool).astype(np.uint8)

    ridge_density = np.sum(ridge_binary) / lesion_area

    # --- 曲率分析 ---
    # 通过方向场的局部变化估计曲率
    # 方向变化大 = 曲线弯曲 = 脑回样
    direction_float = max_direction.astype(np.float64)
    direction_float[~mask_bool] = 0

    # 方向梯度（用Sobel近似）
    grad_x = cv2.Sobel(direction_float, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(direction_float, cv2.CV_64F, 0, 1, ksize=5)
    curvature_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
    curvature_map[~mask_bool] = 0

    # 仅在沟回区域内计算平均曲率
    ridge_region = ridge_binary > 0
    if np.sum(ridge_region) > 0:
        mean_curvature = float(np.mean(curvature_map[ridge_region]))
    else:
        mean_curvature = 0.0

    # --- 平行性检测 ---
    # 在沟回区域内，相邻像素方向一致性高 = 平行结构
    # 用方向场的局部方差评估
    kernel_var = np.ones((7, 7)) / 49.0
    direction_local_mean = cv2.filter2D(direction_float, -1, kernel_var)
    direction_sq_mean = cv2.filter2D(direction_float ** 2, -1, kernel_var)
    direction_var = direction_sq_mean - direction_local_mean ** 2
    direction_var[~mask_bool] = 0

    if np.sum(ridge_region) > 0:
        parallelism = 1.0 - min(1.0, np.mean(direction_var[ridge_region]) / 3.0)
    else:
        parallelism = 0.0

    # --- 综合评分 ---
    # 脑回样 = 高密度 + 高曲率 + 高平行性
    cerebriform_score = (
        min(1.0, ridge_density / 0.3) * 0.4 +
        min(1.0, mean_curvature / 2.0) * 0.3 +
        parallelism * 0.3
    )

    return {
        "cerebriform_score": float(np.clip(cerebriform_score, 0, 1)),
        "ridge_density": float(ridge_density),
        "mean_curvature": float(mean_curvature),
    }