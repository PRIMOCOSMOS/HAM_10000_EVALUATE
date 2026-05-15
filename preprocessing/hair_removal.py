"""
去毛发模块
原理：黑帽变换提取暗色细线结构（毛发），形态学闭运算生成毛发掩膜，
     再用 OpenCV inpainting 修复被毛发遮挡的区域。
"""

import cv2
import numpy as np


def remove_hair(image_rgb: np.ndarray, kernel_size: int = 17, threshold: int = 10) -> np.ndarray:
    """
    去除皮肤镜图像中的毛发伪影。

    参数:
        image_rgb: 输入RGB图像 (H, W, 3), uint8
        kernel_size: 黑帽变换结构元素尺寸，需为奇数，越大可检测越粗的毛发
        threshold: 二值化阈值，用于从黑帽结果中分离毛发区域

    返回:
        去毛发后的RGB图像, uint8
    """
    # 转灰度用于形态学操作
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 黑帽变换：提取比周围暗的细长结构（毛发）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 二值化得到毛发掩膜
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # 轻微膨胀确保毛发边缘完全覆盖
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)

    # 用 Telea inpainting 修复毛发区域
    result = cv2.inpaint(image_rgb, hair_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)

    return result