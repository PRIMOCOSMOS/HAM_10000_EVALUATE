"""
极坐标展开/逆变换工具
用于条纹/假足检测、叶状/轮辐结构检测等需要分析径向结构的场景。
"""

import cv2
import numpy as np


def polar_unwrap(image: np.ndarray, mask: np.ndarray, center: tuple = None,
                 num_angles: int = 360, num_radii: int = 50,
                 r_min: float = None, r_max: float = None) -> dict:
    """
    将图像的环形区域展开为极坐标矩形。

    参数:
        image: 输入图像（灰度或彩色）
        mask: 感兴趣区域掩膜, uint8
        center: 极坐标中心 (cx, cy)，默认为掩膜质心
        num_angles: 角度采样数
        num_radii: 半径采样数
        r_min: 最小半径，默认自动计算
        r_max: 最大半径，默认自动计算

    返回:
        dict:
            polar_image: 展开后的图像 (num_radii, num_angles)或(num_radii, num_angles, C)
            polar_mask: 展开后的掩膜 (num_radii, num_angles)
            center: 使用的中心坐标
            r_min: 使用的最小半径
            r_max: 使用的最大半径
            angles: 角度数组 (num_angles,)
            radii: 半径数组 (num_radii,)
    """
    mask_bool = mask > 0

    # 确定中心
    if center is None:
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return _empty_polar(image, num_angles, num_radii)
        center = (int(np.mean(xs)), int(np.mean(ys)))

    cx, cy = center

    # 确定半径范围
    if r_min is None or r_max is None:
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return _empty_polar(image, num_angles, num_radii)
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        if r_min is None:
            r_min = float(np.min(dists))
        if r_max is None:
            r_max = float(np.max(dists))

    if r_max - r_min < 3:
        return _empty_polar(image, num_angles, num_radii)

    radii = np.linspace(r_min, r_max, num_radii)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    h, w = mask.shape[:2]
    is_color = image.ndim == 3

    if is_color:
        polar_image = np.zeros((num_radii, num_angles, image.shape[2]), dtype=image.dtype)
    else:
        polar_image = np.zeros((num_radii, num_angles), dtype=np.float64)

    polar_mask = np.zeros((num_radii, num_angles), dtype=np.uint8)

    # 构建采样坐标网格（向量化加速）
    r_grid, a_grid = np.meshgrid(radii, angles, indexing="ij")
    x_grid = cx + r_grid * np.cos(a_grid)
    y_grid = cy + r_grid * np.sin(a_grid)

    # 整数坐标（最近邻采样）
    xi = np.clip(np.round(x_grid).astype(int), 0, w - 1)
    yi = np.clip(np.round(y_grid).astype(int), 0, h - 1)

    # 采样
    if is_color:
        polar_image = image[yi, xi]
    else:
        img_float = image.astype(np.float64) if image.dtype != np.float64 else image
        polar_image = img_float[yi, xi]

    polar_mask = mask[yi, xi]

    return {
        "polar_image": polar_image,
        "polar_mask": polar_mask,
        "center": center,
        "r_min": r_min,
        "r_max": r_max,
        "angles": angles,
        "radii": radii,
    }


def polar_rewrap(polar_image: np.ndarray, center: tuple, r_min: float, r_max: float,
                 output_shape: tuple) -> np.ndarray:
    """
    将极坐标图像逆变换回笛卡尔坐标。

    参数:
        polar_image: 极坐标图像 (num_radii, num_angles)
        center: 中心坐标 (cx, cy)
        r_min: 最小半径
        r_max: 最大半径
        output_shape: 输出图像尺寸 (H, W)

    返回:
        笛卡尔坐标图像
    """
    h, w = output_shape
    cx, cy = center
    num_radii, num_angles = polar_image.shape[:2]

    output = np.zeros(output_shape, dtype=polar_image.dtype)

    # 对输出图像每个像素计算其极坐标
    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    theta[theta < 0] += 2 * np.pi

    # 映射到极坐标图像的索引
    ri = ((r - r_min) / (r_max - r_min) * (num_radii - 1)).astype(int)
    ai = (theta / (2 * np.pi) * num_angles).astype(int) % num_angles

    # 有效范围
    valid = (ri >= 0) & (ri < num_radii)
    output[valid] = polar_image[ri[valid], ai[valid]]

    return output


def angular_profile(polar_image: np.ndarray, polar_mask: np.ndarray,
                    num_sectors: int = 36) -> np.ndarray:
    """
    计算极坐标图像的角度方向强度分布。

    参数:
        polar_image: 极坐标图像 (num_radii, num_angles)
        polar_mask: 极坐标掩膜
        num_sectors: 扇区数量

    返回:
        各扇区的平均强度数组 (num_sectors,)
    """
    num_angles = polar_image.shape[1]
    sector_width = max(num_angles // num_sectors, 1)

    profile = np.zeros(num_sectors)
    for i in range(num_sectors):
        c_start = i * sector_width
        c_end = min((i + 1) * sector_width, num_angles)
        sector_data = polar_image[:, c_start:c_end]
        sector_mask = polar_mask[:, c_start:c_end]
        if np.sum(sector_mask) > 0:
            profile[i] = np.mean(sector_data[sector_mask > 0])

    return profile


def _empty_polar(image, num_angles, num_radii):
    """返回空的极坐标结果"""
    if image.ndim == 3:
        polar_image = np.zeros((num_radii, num_angles, image.shape[2]), dtype=image.dtype)
    else:
        polar_image = np.zeros((num_radii, num_angles), dtype=np.float64)

    return {
        "polar_image": polar_image,
        "polar_mask": np.zeros((num_radii, num_angles), dtype=np.uint8),
        "center": (0, 0),
        "r_min": 0.0,
        "r_max": 0.0,
        "angles": np.zeros(num_angles),
        "radii": np.zeros(num_radii),
    }