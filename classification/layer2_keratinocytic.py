"""
第2层决策：角质形成细胞组内细分 (bkl vs akiec vs bcc)
基于各类特异性结构的加权评分：
  - bkl: 粟粒样囊肿 + 粉刺样开口 + 脑回样裂隙 + 边界锐利
  - akiec: 草莓样模式 + 表面鳞屑/粗糙 + 点状血管
  - bcc: 树枝状血管 + 蓝灰卵圆巢 + 叶状/轮辐结构

取最高分者为诊断结果。
"""

import numpy as np


def classify_keratinocytic(features: dict, weights: dict = None) -> dict:
    """
    在角质形成细胞组内区分 bkl, akiec, bcc。

    参数:
        features: dict, 包含以下键:
            - milia_cysts: dict
            - comedo_openings: dict
            - fissures_ridges: dict
            - border_sharpness: dict
            - strawberry_pattern: dict
            - surface_texture: dict
            - vascular_pattern: dict
            - arborizing_vessels: dict
            - ovoid_nests: dict
            - leaf_spoke_structures: dict
            - blue_white_veil: dict (辅助bcc判定)

    返回:
        dict:
            diagnosis: str, 'bkl' / 'akiec' / 'bcc'
            confidence: float (0-1)
            scores: dict, 各类得分明细
            reasoning: str
    """
    # 默认权重（基于文献中各结构的阳性预测值）
    w = {
        # bkl权重
        "bkl_milia": 0.25,
        "bkl_comedo": 0.25,
        "bkl_fissures": 0.25,
        "bkl_border": 0.25,
        # akiec权重
        "akiec_strawberry": 0.35,
        "akiec_roughness": 0.35,
        "akiec_dotted_vessels": 0.30,
        # bcc权重
        "bcc_arborizing": 0.35,
        "bcc_ovoid_nests": 0.25,
        "bcc_leaf_spoke": 0.20,
        "bcc_ulceration": 0.10,
        "bcc_blue_structures": 0.10,
    }
    if weights:
        w.update(weights)

    # 提取各模块结果
    milia = features.get("milia_cysts", {})
    comedo = features.get("comedo_openings", {})
    fissures = features.get("fissures_ridges", {})
    border = features.get("border_sharpness", {})
    strawberry = features.get("strawberry_pattern", {})
    texture = features.get("surface_texture", {})
    vasc = features.get("vascular_pattern", {})
    arbor = features.get("arborizing_vessels", {})
    nests = features.get("ovoid_nests", {})
    leaf_spoke = features.get("leaf_spoke_structures", {})
    bwv = features.get("blue_white_veil", {})

    # --- bkl 得分 ---
    bkl_score = (
        w["bkl_milia"] * milia.get("milia_score", 0.0) +
        w["bkl_comedo"] * comedo.get("comedo_score", 0.0) +
        w["bkl_fissures"] * fissures.get("cerebriform_score", 0.0) +
        w["bkl_border"] * border.get("sharpness_score", 0.0)
    )

    # --- akiec 得分 ---
    # 点状血管对akiec的贡献
    dotted_vessel_score = 0.0
    if vasc.get("morphology_type") == "dotted":
        dotted_vessel_score = vasc.get("dotted_ratio", 0.0)

    akiec_score = (
        w["akiec_strawberry"] * strawberry.get("strawberry_score", 0.0) +
        w["akiec_roughness"] * texture.get("roughness_score", 0.0) +
        w["akiec_dotted_vessels"] * dotted_vessel_score
    )

    # --- bcc 得分 ---
    # 蓝色结构对bcc的辅助贡献（蓝灰卵圆巢已单独计算，这里用bwv补充）
    blue_struct_score = min(1.0, nests.get("nest_score", 0.0) + bwv.get("bwv_score", 0.0) * 0.5)

    bcc_score = (
        w["bcc_arborizing"] * arbor.get("arborizing_score", 0.0) +
        w["bcc_ovoid_nests"] * nests.get("nest_score", 0.0) +
        w["bcc_leaf_spoke"] * leaf_spoke.get("combined_score", 0.0) +
        w["bcc_blue_structures"] * blue_struct_score
    )
    # 溃疡贡献（如果有的话，从surface_texture中推断）
    # 简化处理：高粗糙度+低亮度区域可能提示溃疡，但这里暂不加入

    scores = {
        "bkl": float(bkl_score),
        "akiec": float(akiec_score),
        "bcc": float(bcc_score),
    }

    # --- 取最高分 ---
    diagnosis = max(scores, key=scores.get)
    max_score = scores[diagnosis]

    # 置信度：最高分与次高分的差距
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] > 0:
        margin = sorted_scores[0] - sorted_scores[1]
        confidence = min(1.0, 0.5 + margin / sorted_scores[0] * 0.5)
    else:
        confidence = 0.33  # 所有得分为0时均等不确定

    # --- 构建推理说明 ---
    detail_parts = []
    if diagnosis == "bkl":
        if milia.get("milia_count", 0) > 0:
            detail_parts.append(f"粟粒样囊肿x{milia['milia_count']}")
        if comedo.get("comedo_count", 0) > 0:
            detail_parts.append(f"粉刺样开口x{comedo['comedo_count']}")
        if fissures.get("cerebriform_score", 0) > 0.3:
            detail_parts.append("脑回样裂隙")
        if border.get("sharpness_score", 0) > 0.5:
            detail_parts.append("边界锐利")
    elif diagnosis == "akiec":
        if strawberry.get("strawberry_score", 0) > 0.3:
            detail_parts.append("草莓样模式")
        if texture.get("roughness_score", 0) > 0.3:
            detail_parts.append("表面粗糙/鳞屑")
        if dotted_vessel_score > 0.3:
            detail_parts.append("点状血管")
    elif diagnosis == "bcc":
        if arbor.get("has_arborizing", False):
            detail_parts.append("树枝状血管")
        if nests.get("nest_count", 0) > 0:
            detail_parts.append(f"蓝灰卵圆巢x{nests['nest_count']}")
        if leaf_spoke.get("combined_score", 0) > 0.3:
            detail_parts.append("叶状/轮辐结构")

    reasoning = (
        f"评分: bkl={bkl_score:.3f}, akiec={akiec_score:.3f}, bcc={bcc_score:.3f} → {diagnosis}; "
        f"依据: {', '.join(detail_parts) if detail_parts else '综合评分'}"
    )

    return {
        "diagnosis": diagnosis,
        "confidence": float(np.clip(confidence, 0, 1)),
        "scores": scores,
        "reasoning": reasoning,
    }