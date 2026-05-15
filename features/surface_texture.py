"""
表面鳞屑/粗糙度检测
医学依据：光化性角化病(akiec)表面有白色/黄色鳞屑覆盖，表面粗糙；
         脂溢性角化病(bkl)表面可光滑或有蜡样质感。
BIP方法：小波分解计算高频子带能量评估表面粗糙度 →
         高亮低饱和度片状区域检测鳞屑覆盖。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    评估病灶表面纹理粗糙度和鳞屑覆盖。

    返回:
        roughness_score: float (0-1), 表面粗糙度
        scale_area_ratio: float, 鳞屑区域占病灶面积比
        high_freq_energy: float, 高频能量比
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return {"roughness_score": 0.0, "scale_area_ratio": 0.0, "high_freq_energy": 0.0}

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # --- 高频能量评估（简化小波：用高斯金字塔近似） ---
    # 原图能量
    lesion_gray = gray.copy()
    lesion_gray[~mask_bool] = 0
    total_energy = np.sum(lesion_gray[mask_bool] ** 2)

    # 低频分量（高斯模糊）
    low_freq = cv2.GaussianBlur(lesion_gray, (15, 15), 3.0)
    # 高频分量 = 原图 - 低频
    high_freq = lesion_gray - low_freq
    high_freq[~mask_bool] = 0

    high_freq_energy_val = np.sum(high_freq[mask_bool] ** 2)
    if total_energy > 0:
        high_freq_ratio = high_freq_energy_val / total_energy
    else:
        high_freq_ratio = 0.0

    # --- 鳞屑区域检测 ---
    # 鳞屑：高亮度、低饱和度、有纹理（高频能量高）的片状区域
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    l_channel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float64)
    s_channel = hsv[:, :, 1].astype(np.float64) / 255.0

    lesion_l_mean = np.mean(l_channel[mask_bool])

    # 鳞屑候选：亮度高于均值、饱和度低、在病灶内
    scale_candidate = (
        (l_channel > lesion_l_mean + 0.5 * np.std(l_channel[mask_bool])) &
        (s_channel < 0.2) &
        mask_bool
    )

    # 鳞屑应有纹理（排除光滑白色区域如瘢痕）
    # 用局部标准差评估纹理
    local_std = _local_std(gray, ksize=7)
    textured = local_std > 5.0

    scale_mask = scale_candidate & textured
    scale_area_ratio = np.sum(scale_mask) / lesion_area

    # --- 综合粗糙度评分 ---
    roughness_score = (
        min(1.0, high_freq_ratio / 0.15) * 0.5 +
        min(1.0, scale_area_ratio / 0.3) * 0.5
    )

    return {
        "roughness_score": float(np.clip(roughness_score, 0, 1)),
        "scale_area_ratio": float(scale_area_ratio),
        "high_freq_energy": float(high_freq_ratio),
    }


def _local_std(image: np.ndarray, ksize: int = 7) -> np.ndarray:
    """计算局部标准差图"""
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    local_mean = cv2.filter2D(image, -1, kernel)
    local_sq_mean = cv2.filter2D(image ** 2, -1, kernel)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)
    return np.sqrt(local_var)