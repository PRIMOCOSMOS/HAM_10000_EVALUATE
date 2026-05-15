"""
Gabor滤波器组构建与应用
用于色素网络检测、脑回样裂隙检测等需要方向性纹理分析的场景。
"""

import numpy as np
from skimage.filters import gabor


def build_gabor_bank(frequencies: list = None, num_orientations: int = 6) -> list:
    """
    构建Gabor滤波器参数组。

    参数:
        frequencies: 频率列表，对应不同纹理间距
        num_orientations: 方向数量

    返回:
        list of dict: 每个元素包含 {'frequency': float, 'theta': float}
    """
    if frequencies is None:
        frequencies = [0.1, 0.15, 0.2]

    orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)

    bank = []
    for freq in frequencies:
        for theta in orientations:
            bank.append({"frequency": freq, "theta": float(theta)})

    return bank


def apply_gabor_bank(image: np.ndarray, bank: list = None,
                     frequencies: list = None, num_orientations: int = 6) -> dict:
    """
    对图像应用Gabor滤波器组，返回最大响应图和方向图。

    参数:
        image: 单通道图像, float64
        bank: 预构建的滤波器参数组（优先使用）
        frequencies: 若bank为None则用此参数构建
        num_orientations: 若bank为None则用此参数构建

    返回:
        dict:
            max_response: 每像素的最大Gabor响应幅值
            direction_map: 每像素最大响应对应的方向索引
            mean_response: 所有滤波器响应的均值图
    """
    if bank is None:
        bank = build_gabor_bank(frequencies, num_orientations)

    h, w = image.shape
    max_response = np.zeros((h, w), dtype=np.float64)
    direction_map = np.zeros((h, w), dtype=np.int32)
    sum_response = np.zeros((h, w), dtype=np.float64)

    for idx, params in enumerate(bank):
        filt_real, _ = gabor(image, frequency=params["frequency"], theta=params["theta"])
        response = np.abs(filt_real)
        sum_response += response

        update = response > max_response
        max_response[update] = response[update]
        direction_map[update] = idx

    mean_response = sum_response / max(len(bank), 1)

    return {
        "max_response": max_response,
        "direction_map": direction_map,
        "mean_response": mean_response,
    }


def gabor_energy(image: np.ndarray, frequency: float, theta: float) -> np.ndarray:
    """
    计算单个Gabor滤波器的能量响应（实部²+虚部²的平方根）。

    参数:
        image: 单通道图像, float64
        frequency: Gabor频率
        theta: Gabor方向（弧度）

    返回:
        能量响应图
    """
    filt_real, filt_imag = gabor(image, frequency=frequency, theta=theta)
    return np.sqrt(filt_real ** 2 + filt_imag ** 2)