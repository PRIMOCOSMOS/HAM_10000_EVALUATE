## preprocessing/illumination_correction.py

"""
光照均匀化模块
原理：用大核高斯模糊估计低频光照场（渐晕效应），
     原图除以光照场后乘以均值恢复亮度，消除不均匀光照。
"""

import cv2
import numpy as np


def correct_illumination(image_rgb: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """
    基于低频光照场估计的除法校正。

    参数:
        image_rgb: 输入RGB图像 (H, W, 3), uint8
        sigma: 高斯模糊的标准差，越大估计的光照场越平滑
               建议设为图像短边的 1/6 ~ 1/4

    返回:
        光照校正后的RGB图像, uint8
    """
    img = image_rgb.astype(np.float64)

    # 高斯核尺寸需为奇数，取 6*sigma 保证截断误差足够小
    ksize = int(np.ceil(sigma * 6)) | 1  # 确保奇数

    # 逐通道估计光照场并做除法校正
    result = np.zeros_like(img)
    for c in range(3):
        channel = img[:, :, c]

        # 低频光照场估计
        illumination = cv2.GaussianBlur(channel, (ksize, ksize), sigma)

        # 除法校正：I_corrected = I / L * mean(L)
        # 乘以均值保持整体亮度不变
        mean_illum = np.mean(illumination)
        # 避免除零
        corrected = channel / np.maximum(illumination, 1e-6) * mean_illum
        result[:, :, c] = corrected

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result