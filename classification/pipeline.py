"""
完整分类流水线
串联预处理 → 特征提取 → 三层决策，输出最终诊断和完整报告。
"""

import numpy as np

from preprocessing import remove_hair, normalize_color, correct_illumination, segment_lesion
from features import (
    compute_pigment_network, compute_lacunae, compute_central_white_patch,
    compute_symmetry, compute_color_variegation, compute_blue_white_veil,
    compute_streaks, compute_dots_globules, compute_regression,
    compute_vascular_pattern, compute_milia_cysts, compute_comedo_openings,
    compute_fissures_ridges, compute_border_sharpness, compute_strawberry_pattern,
    compute_surface_texture, compute_arborizing_vessels, compute_ovoid_nests,
    compute_leaf_spoke_structures,
)
from classification.layer1_origin import (
    classify_origin, ORIGIN_MELANOCYTIC, ORIGIN_VASCULAR, ORIGIN_FIBROUS
)
from classification.layer2_melanocytic import classify_melanocytic
from classification.layer2_keratinocytic import classify_keratinocytic


def run_pipeline(image_rgb: np.ndarray, config: dict = None) -> dict:
    """
    执行完整的HAM10000七分类流水线。

    参数:
        image_rgb: 原始RGB图像 (H, W, 3), uint8
        config: 可选配置字典，覆盖默认阈值/参数

    返回:
        dict:
            diagnosis: str, 最终诊断 ('nv'/'mel'/'bkl'/'akiec'/'bcc'/'vasc'/'df')
            confidence: float (0-1)
            origin: str, 组织来源
            decision_path: list[str], 决策路径
            all_features: dict, 所有特征的量化结果
            layer1_result: dict, 第1层决策详情
            layer2_result: dict, 第2层决策详情（如适用）
    """
    if config is None:
        config = {}

    decision_path = []

    # ========== 第0层：预处理 ==========
    img = remove_hair(image_rgb)
    img = normalize_color(img)
    img = correct_illumination(img)
    seg_result = segment_lesion(img)
    mask = seg_result["mask"]

    decision_path.append(f"[L0] 预处理完成, 病灶面积={seg_result['area']}px")

    # ========== 第1层特征提取 ==========
    feat_network = compute_pigment_network(img, mask)
    feat_lacunae = compute_lacunae(img, mask)
    feat_cwp = compute_central_white_patch(img, mask)

    l1_features = {
        "pigment_network": feat_network,
        "lacunae": feat_lacunae,
        "central_white_patch": feat_cwp,
    }

    # ========== 第1层决策 ==========
    l1_result = classify_origin(l1_features, thresholds=config.get("layer1_thresholds"))
    origin = l1_result["origin"]
    decision_path.append(
        f"[L1] 组织来源={origin} (confidence={l1_result['confidence']:.2f}): "
        f"{l1_result['reasoning']}"
    )

    # ========== 第2层特征提取与决策 ==========
    l2_result = None
    all_features = dict(l1_features)

    if origin == ORIGIN_VASCULAR:
        diagnosis = "vasc"
        confidence = l1_result["confidence"]
        decision_path.append("[L2] 血管来源 → vasc (无需组内细分)")

    elif origin == ORIGIN_FIBROUS:
        diagnosis = "df"
        confidence = l1_result["confidence"]
        decision_path.append("[L2] 纤维来源 → df (无需组内细分)")

    elif origin == ORIGIN_MELANOCYTIC:
        feat_sym = compute_symmetry(img, mask)
        feat_color = compute_color_variegation(img, mask)
        feat_bwv = compute_blue_white_veil(img, mask)
        feat_streaks = compute_streaks(img, mask)
        feat_dots = compute_dots_globules(img, mask)
        feat_reg = compute_regression(img, mask)
        feat_vasc = compute_vascular_pattern(img, mask)

        l2_features = {
            "pigment_network": feat_network,
            "symmetry": feat_sym,
            "color_variegation": feat_color,
            "blue_white_veil": feat_bwv,
            "streaks": feat_streaks,
            "dots_globules": feat_dots,
            "regression": feat_reg,
            "vascular_pattern": feat_vasc,
        }
        all_features.update(l2_features)

        l2_result = classify_melanocytic(
            l2_features, thresholds=config.get("layer2_mel_thresholds")
        )
        diagnosis = l2_result["diagnosis"]
        confidence = l2_result["confidence"]
        decision_path.append(f"[L2] 黑色素细胞组: {l2_result['reasoning']}")

    else:
        # 角质形成细胞组
        feat_milia = compute_milia_cysts(img, mask)
        feat_comedo = compute_comedo_openings(img, mask)
        feat_fissures = compute_fissures_ridges(img, mask)
        feat_border = compute_border_sharpness(img, mask)
        feat_strawberry = compute_strawberry_pattern(img, mask)
        feat_texture = compute_surface_texture(img, mask)
        feat_vasc = compute_vascular_pattern(img, mask)
        feat_arbor = compute_arborizing_vessels(img, mask)
        feat_nests = compute_ovoid_nests(img, mask)
        feat_leaf = compute_leaf_spoke_structures(img, mask)
        feat_bwv = compute_blue_white_veil(img, mask)

        l2_features = {
            "milia_cysts": feat_milia,
            "comedo_openings": feat_comedo,
            "fissures_ridges": feat_fissures,
            "border_sharpness": feat_border,
            "strawberry_pattern": feat_strawberry,
            "surface_texture": feat_texture,
            "vascular_pattern": feat_vasc,
            "arborizing_vessels": feat_arbor,
            "ovoid_nests": feat_nests,
            "leaf_spoke_structures": feat_leaf,
            "blue_white_veil": feat_bwv,
        }
        all_features.update(l2_features)

        l2_result = classify_keratinocytic(
            l2_features, weights=config.get("layer2_kerat_weights")
        )
        diagnosis = l2_result["diagnosis"]
        confidence = l2_result["confidence"]
        decision_path.append(f"[L2] 角质形成细胞组: {l2_result['reasoning']}")

    return {
        "diagnosis": diagnosis,
        "confidence": float(confidence),
        "origin": origin,
        "decision_path": decision_path,
        "all_features": all_features,
        "layer1_result": l1_result,
        "layer2_result": l2_result,
    }