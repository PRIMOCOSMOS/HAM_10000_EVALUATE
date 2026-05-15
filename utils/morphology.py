"""
通用形态学操作封装
对OpenCV形态学操作做简洁封装，统一接口风格。
"""

import cv2
import numpy as np


def tophat(image: np.ndarray, radius: int = 9, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """
    白帽变换：提取比周围亮的小结构。

    参数:
        image: 单通道图像, uint8或float
        radius: 结构元素半径
        shape: 结构元素形状 (cv2.MORPH_ELLIPSE / MORPH_RECT)

    返回:
        白帽变换结果, 与输入同类型
    """
    kernel = cv2.getStructuringElement(shape, (radius * 2 + 1, radius * 2 + 1))
    img_uint8 = _ensure_uint8(image)
    return cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel)


def blackhat(image: np.ndarray, radius: int = 9, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """
    黑帽变换：提取比周围暗的小结构。

    参数:
        image: 单通道图像
        radius: 结构元素半径
        shape: 结构元素形状

    返回:
        黑帽变换结果
    """
    kernel = cv2.getStructuringElement(shape, (radius * 2 + 1, radius * 2 + 1))
    img_uint8 = _ensure_uint8(image)
    return cv2.morphologyEx(img_uint8, cv2.MORPH_BLACKHAT, kernel)


def open_close(binary: np.ndarray, radius: int = 7, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """
    先开运算去噪点，再闭运算填小孔。

    参数:
        binary: 二值图像, uint8
        radius: 结构元素半径
        shape: 结构元素形状

    返回:
        精修后的二值图像
    """
    kernel = cv2.getStructuringElement(shape, (radius * 2 + 1, radius * 2 + 1))
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    填充二值掩膜中的内部孔洞。

    参数:
        binary_mask: 二值图像 (H, W), uint8, 255=前景

    返回:
        填充孔洞后的二值图像
    """
    h, w = binary_mask.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    im_flood = binary_mask.copy()
    cv2.floodFill(im_flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(im_flood)
    return binary_mask | holes


def create_border_band(mask: np.ndarray, width: int = 7) -> np.ndarray:
    """
    创建病灶边界带掩膜（边界内外各width像素的环形区域）。

    参数:
        mask: 病灶掩膜, uint8
        width: 边界带宽度（单侧像素数）

    返回:
        边界带二值掩膜, uint8
    """
    ksize = width * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    border = ((dilated > 0) & ~(eroded > 0)).astype(np.uint8) * 255
    return border


def create_radial_zones(mask: np.ndarray, ratio: float = 0.5) -> tuple:
    """
    用距离变换将病灶分为中央区和周围区。

    参数:
        mask: 病灶掩膜, uint8
        ratio: 中央区距离阈值比例 (0-1)

    返回:
        (central_mask, peripheral_mask): 两个bool数组
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = np.max(dist)
    if max_dist < 1:
        mask_bool = mask > 0
        return mask_bool, np.zeros_like(mask_bool)

    central = dist > (max_dist * ratio)
    peripheral = (mask > 0) & ~central
    return central, peripheral


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """确保图像为uint8格式"""
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)