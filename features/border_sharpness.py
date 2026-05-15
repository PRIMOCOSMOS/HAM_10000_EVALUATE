"""
边界锐利度评估
医学依据：脂溢性角化病(bkl)通常边界非常清晰锐利，像"贴在皮肤上"；
         光化性角化病(akiec)边界模糊渐变，与周围皮肤过渡不清。
BIP方法：在病灶边界处计算梯度幅值的平均值和锐利度指标。
"""

import cv2
import numpy as np


def compute(image_rgb: np.ndarray, mask: np.ndarray, **kwargs) -> dict:
    """
    评估病灶边界的锐利程度。

    返回:
        sharpness_score: float (0-1), 越高越锐利
        mean_gradient: float, 边界处平均梯度幅值
        gradient_consistency: float, 梯度一致性（标准差的倒数归一化）
    """
    mask_bool = mask > 0
    lesion_area = np.sum(mask_bool)
    if lesion_area == 0:
        return {"sharpness_score": 0.0, "mean_gradient": 0.0, "gradient_consistency": 0.0}

    # 转灰度
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # 计算梯度幅值（Sobel）
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 创建边界带掩膜（边界内外各3px）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    border_band = ((dilated > 0) & ~(eroded > 0))

    border_pixels = np.sum(border_band)
    if border_pixels < 10:
        return {"sharpness_score": 0.0, "mean_gradient": 0.0, "gradient_consistency": 0.0}

    # 边界处梯度统计
    border_gradients = gradient_mag[border_band]
    mean_gradient = float(np.mean(border_gradients))
    std_gradient = float(np.std(border_gradients))

    # 梯度一致性：CV的倒数归一化（一致性高=锐利边界）
    cv_gradient = std_gradient / max(mean_gradient, 1e-6)
    gradient_consistency = 1.0 / (1.0 + cv_gradient)

    # 综合锐利度评分
    # 归一化mean_gradient（典型范围0-100）
    grad_norm = min(1.0, mean_gradient / 60.0)
    sharpness_score = grad_norm * 0.7 + gradient_consistency * 0.3

    return {
        "sharpness_score": float(np.clip(sharpness_score, 0, 1)),
        "mean_gradient": float(mean_gradient),
        "gradient_consistency": float(gradient_consistency),
    }