"""
病灶分割模块
原理：在Lab的L通道上用Otsu自适应阈值做初始分割，
     再通过形态学开闭运算精修边界、填充孔洞，
     最终输出二值掩膜和边界轮廓。
"""

import cv2
import numpy as np


def segment_lesion(image_rgb: np.ndarray, morph_radius: int = 15) -> dict:
    """
    分割皮肤病灶区域。

    参数:
        image_rgb: 预处理后的RGB图像 (H, W, 3), uint8
        morph_radius: 形态学精修的结构元素半径

    返回:
        dict:
            mask: 二值掩膜 (H, W), uint8, 255=病灶区域
            contour: 最大轮廓点集, np.ndarray
            bbox: 边界框 (x, y, w, h)
            area: 病灶面积（像素数）
    """
    # 转Lab空间，取L通道（对色素病变对比度好）
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)

    # Otsu自适应阈值（病灶通常比周围皮肤暗）
    thresh_val, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学精修：先开运算去小噪点，再闭运算填小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_radius, morph_radius))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 填充内部孔洞（flood fill 方法）
    mask_filled = _fill_holes(mask)

    # 提取轮廓，保留最大连通域（即病灶主体）
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 分割失败时返回全图作为病灶（降级处理）
        h, w = image_rgb.shape[:2]
        return {
            "mask": np.ones((h, w), dtype=np.uint8) * 255,
            "contour": None,
            "bbox": (0, 0, w, h),
            "area": h * w,
        }

    # 取面积最大的轮廓
    largest = max(contours, key=cv2.contourArea)

    # 生成最终掩膜（仅保留最大连通域）
    final_mask = np.zeros_like(mask_filled)
    cv2.drawContours(final_mask, [largest], -1, 255, thickness=cv2.FILLED)

    bbox = cv2.boundingRect(largest)
    area = cv2.contourArea(largest)

    return {
        "mask": final_mask,
        "contour": largest,
        "bbox": bbox,
        "area": int(area),
    }


def _fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    填充二值掩膜中的内部孔洞。
    利用 floodFill 从边界填充背景，取反后与原图合并。
    """
    h, w = binary_mask.shape
    # 创建比原图大2像素的画布（floodFill要求）
    flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # 复制原图用于flood fill
    im_floodfill = binary_mask.copy()

    # 从左上角(0,0)开始填充背景
    cv2.floodFill(im_floodfill, flood_fill_mask, (0, 0), 255)

    # 取反得到孔洞区域
    holes = cv2.bitwise_not(im_floodfill)

    # 原掩膜 + 孔洞 = 填充后的掩膜
    filled = binary_mask | holes

    return filled