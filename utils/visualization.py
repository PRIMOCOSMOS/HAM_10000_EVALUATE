"""
特征可视化工具
将检测到的特征叠加标注到原图上，生成诊断报告可视化。
"""

import cv2
import numpy as np


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 255, 0), alpha: float = 0.3) -> np.ndarray:
    """
    将掩膜以半透明颜色叠加到原图上。

    参数:
        image_rgb: 原始RGB图像
        mask: 二值掩膜, uint8
        color: 叠加颜色 (R, G, B)
        alpha: 透明度 (0=完全透明, 1=完全不透明)

    返回:
        叠加后的RGB图像
    """
    overlay = image_rgb.copy()
    mask_bool = mask > 0

    color_layer = np.zeros_like(image_rgb)
    color_layer[mask_bool] = color

    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * color_layer[mask_bool]
    ).astype(np.uint8)

    return overlay


def overlay_contour(image_rgb: np.ndarray, mask: np.ndarray,
                    color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    在原图上绘制掩膜轮廓线。

    参数:
        image_rgb: 原始RGB图像
        mask: 二值掩膜
        color: 轮廓颜色 (R, G, B)
        thickness: 线宽

    返回:
        绘制轮廓后的图像
    """
    result = image_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV drawContours使用BGR，这里输入是RGB所以颜色需反转
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def overlay_features(image_rgb: np.ndarray, feature_masks: dict) -> np.ndarray:
    """
    将多个特征掩膜以不同颜色叠加到原图。

    参数:
        image_rgb: 原始RGB图像
        feature_masks: dict, {特征名: (mask, color)}
            mask: 二值掩膜
            color: (R, G, B) 颜色

    返回:
        多特征叠加后的图像
    """
    result = image_rgb.copy()
    for name, (feat_mask, color) in feature_masks.items():
        if feat_mask is not None and np.sum(feat_mask) > 0:
            result = overlay_mask(result, feat_mask, color, alpha=0.4)
    return result


def draw_report(image_rgb: np.ndarray, diagnosis: str, scores: dict,
                path: list = None, canvas_width: int = 400) -> np.ndarray:
    """
    生成诊断报告可视化图像（原图 + 右侧文字面板）。

    参数:
        image_rgb: 原始RGB图像
        diagnosis: 诊断结果字符串
        scores: 各特征得分字典
        path: 诊断路径列表 (可选)
        canvas_width: 右侧文字面板宽度

    返回:
        拼接后的可视化图像
    """
    h, w = image_rgb.shape[:2]

    # 创建右侧文字面板（白色背景）
    panel = np.ones((h, canvas_width, 3), dtype=np.uint8) * 255

    # 绘制文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    line_height = 22

    # 标题
    cv2.putText(panel, f"Diagnosis: {diagnosis}", (10, y_offset),
                font, 0.6, (0, 0, 200), 2)
    y_offset += line_height * 2

    # 诊断路径
    if path:
        cv2.putText(panel, "Decision Path:", (10, y_offset),
                    font, 0.45, (0, 0, 0), 1)
        y_offset += line_height
        for step in path:
            cv2.putText(panel, f"  {step}", (10, y_offset),
                        font, 0.35, (80, 80, 80), 1)
            y_offset += line_height

    y_offset += line_height

    # 特征得分
    cv2.putText(panel, "Feature Scores:", (10, y_offset),
                font, 0.45, (0, 0, 0), 1)
    y_offset += line_height

    for name, value in scores.items():
        if isinstance(value, float):
            text = f"  {name}: {value:.3f}"
        elif isinstance(value, bool):
            text = f"  {name}: {'Y' if value else 'N'}"
        else:
            text = f"  {name}: {value}"

        # 截断过长文字
        text = text[:45]
        cv2.putText(panel, text, (10, y_offset),
                    font, 0.33, (50, 50, 50), 1)
        y_offset += line_height

        if y_offset > h - 20:
            break

    # 拼接原图和面板
    result = np.hstack([image_rgb, panel])
    return result


def draw_score_bar(panel: np.ndarray, x: int, y: int,
                   score: float, width: int = 150, height: int = 12,
                   color: tuple = (0, 150, 0)) -> None:
    """
    在面板上绘制得分条形图（原地修改）。

    参数:
        panel: 目标图像
        x, y: 左上角坐标
        score: 得分 (0-1)
        width: 条形总宽度
        height: 条形高度
        color: 填充颜色
    """
    # 背景框
    cv2.rectangle(panel, (x, y), (x + width, y + height), (200, 200, 200), 1)
    # 填充
    fill_width = int(width * np.clip(score, 0, 1))
    if fill_width > 0:
        cv2.rectangle(panel, (x, y), (x + fill_width, y + height), color, -1)