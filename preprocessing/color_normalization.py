"""
颜色归一化模块
原理：Shades-of-Gray 方法估计光源色温，对图像做白平衡校正，
     消除不同采集设备带来的色偏差异。
"""

import numpy as np


def normalize_color(image_rgb: np.ndarray, norm_p: int = 6, target_intensity: float = 128.0) -> np.ndarray:
    """
    Shades-of-Gray 颜色归一化。

    当 norm_p=1 等价于 Gray-World，norm_p→∞ 等价于 Max-RGB，
    norm_p=6 是文献推荐的折中值（Finlayson & Trezzi, 2004）。

    参数:
        image_rgb: 输入RGB图像 (H, W, 3), uint8
        norm_p: Minkowski范数的阶数
        target_intensity: 归一化后的目标平均亮度

    返回:
        颜色归一化后的RGB图像, uint8
    """
    img = image_rgb.astype(np.float64)

    # 对每个通道计算 p-范数均值作为光源估计
    # illuminant_c = (mean(I_c^p))^(1/p)
    illuminant = np.zeros(3)
    for c in range(3):
        channel = img[:, :, c]
        illuminant[c] = np.power(np.mean(np.power(channel, norm_p)), 1.0 / norm_p)

    # 避免除零
    illuminant = np.maximum(illuminant, 1e-6)

    # 计算缩放因子：将估计光源映射到目标亮度
    scale = target_intensity / illuminant

    # 逐通道缩放
    result = img * scale[np.newaxis, np.newaxis, :]

    # 裁剪到有效范围
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result