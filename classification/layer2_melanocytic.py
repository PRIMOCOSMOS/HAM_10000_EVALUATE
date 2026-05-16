"""
第2层决策：黑色素细胞组内细分 (nv vs mel)
重构点：
1) 主标准门控：降低“噪声特征累加”误伤nv；
2) 加入GLCM纹理异质性作为边界修正，而非粗暴替代7-point；
3) 保持接口不变。
"""

import numpy as np


def classify_melanocytic(features: dict, thresholds: dict = None) -> dict:
    t = {
        "symmetry_thresh": 1.25,
        "color_count_thresh": 5,
        "few_color_thresh": 3,
        "bwv_score_thresh": 0.36,
        "bwv_area_ratio_thresh": 0.03,
        "streak_score_thresh": 0.36,
        "streak_asym_thresh": 0.35,
        "dots_irregularity_thresh": 0.45,
        "dots_entropy_thresh": 0.35,
        "regression_score_thresh": 0.34,
        "regression_area_ratio_thresh": 0.04,
        "polymorphism_thresh": 0.50,
        "network_regularity_thresh": 0.52,
        "network_typical_thresh": 0.66,
        "network_coverage_typical_thresh": 0.10,
        "mel_total_thresh": 4,
        "texture_irregularity_thresh": 0.55,
    }
    if thresholds:
        t.update(thresholds)

    net = features.get("pigment_network", {})
    sym = features.get("symmetry", {})
    color = features.get("color_variegation", {})
    bwv = features.get("blue_white_veil", {})
    streaks = features.get("streaks", {})
    dots = features.get("dots_globules", {})
    reg = features.get("regression", {})
    vasc = features.get("vascular_pattern", {})
    glcm = features.get("glcm_texture", {})

    criteria = {}
    mel_score = 0
    major_count = 0
    nv_support = 0.0

    # 主标准1: 非典型网络(2)
    network_regularity = float(net.get("regularity_score", 1.0))
    has_network = bool(net.get("has_network", False))
    coverage = float(net.get("coverage", 0.0))
    atypical_network = has_network and (network_regularity < t["network_regularity_thresh"]) and (coverage >= 0.06)
    criteria["atypical_network"] = {
        "present": atypical_network,
        "score": 2 if atypical_network else 0,
        "detail": f"regularity={network_regularity:.3f}, coverage={coverage:.3f}",
    }
    if atypical_network:
        mel_score += 2
        major_count += 1

    # 主标准2: 蓝白幕(2)
    bwv_val = float(bwv.get("bwv_score", 0.0))
    bwv_area = float(bwv.get("bwv_area_ratio", 0.0))
    has_bwv = (bwv_val >= t["bwv_score_thresh"]) and (bwv_area >= t["bwv_area_ratio_thresh"])
    criteria["blue_white_veil"] = {
        "present": has_bwv,
        "score": 2 if has_bwv else 0,
        "detail": f"bwv={bwv_val:.3f}, area={bwv_area:.3f}",
    }
    if has_bwv:
        mel_score += 2
        major_count += 1

    # 主标准3: 非典型血管(2)
    poly_score = float(vasc.get("polymorphism_score", 0.0))
    vessel_type = str(vasc.get("morphology_type", "none")).lower()
    has_atypical_vessels = (vessel_type in ("dotted", "linear", "mixed")) and (poly_score >= t["polymorphism_thresh"])
    criteria["atypical_vascular"] = {
        "present": has_atypical_vessels,
        "score": 2 if has_atypical_vessels else 0,
        "detail": f"type={vessel_type}, polymorphism={poly_score:.3f}",
    }
    if has_atypical_vessels:
        mel_score += 2
        major_count += 1

    # 次标准1: 不规则条纹(1)
    streak_val = float(streaks.get("streak_score", 0.0))
    streak_asym = float(streaks.get("streak_asymmetry", 0.0))
    has_streaks = (streak_val >= t["streak_score_thresh"]) and (streak_asym >= t["streak_asym_thresh"])
    criteria["irregular_streaks"] = {
        "present": has_streaks,
        "score": 1 if has_streaks else 0,
        "detail": f"streak={streak_val:.3f}, asym={streak_asym:.3f}",
    }
    if has_streaks:
        mel_score += 1

    # 次标准2: 不规则点球(1)
    dots_irreg = float(dots.get("irregularity_score", 0.0))
    dots_entropy = float(dots.get("distribution_entropy", 0.0))
    has_irreg_dots = (dots_irreg >= t["dots_irregularity_thresh"]) and (dots_entropy >= t["dots_entropy_thresh"])
    criteria["irregular_dots_globules"] = {
        "present": has_irreg_dots,
        "score": 1 if has_irreg_dots else 0,
        "detail": f"irreg={dots_irreg:.3f}, entropy={dots_entropy:.3f}",
    }
    if has_irreg_dots:
        mel_score += 1

    # 次标准3: 回归结构(1)
    reg_val = float(reg.get("regression_score", 0.0))
    scar_area = float(reg.get("scar_area_ratio", 0.0))
    has_regression = (reg_val >= t["regression_score_thresh"]) and (scar_area >= t["regression_area_ratio_thresh"])
    criteria["regression_structures"] = {
        "present": has_regression,
        "score": 1 if has_regression else 0,
        "detail": f"reg={reg_val:.3f}, scar={scar_area:.3f}",
    }
    if has_regression:
        mel_score += 1

    # 次标准4: 多色(1)
    color_count = int(color.get("color_count", 1))
    has_multicolor = color_count >= t["color_count_thresh"]
    criteria["multicolor"] = {
        "present": has_multicolor,
        "score": 1 if has_multicolor else 0,
        "detail": f"color_count={color_count}",
    }
    if has_multicolor:
        mel_score += 1

    # GLCM边界修正：异质纹理
    texture_irregularity = float(glcm.get("texture_irregularity", 0.0))
    has_texture_irregular = texture_irregularity >= t["texture_irregularity_thresh"]
    criteria["texture_irregularity"] = {
        "present": has_texture_irregular,
        "score": 0,
        "detail": f"texture_irregularity={texture_irregularity:.3f}",
    }

    # nv负向证据
    typical_network = has_network and (network_regularity >= t["network_typical_thresh"]) and (coverage >= t["network_coverage_typical_thresh"])
    symmetry_score = float(sym.get("symmetry_score", 0.0))
    symmetric = symmetry_score < t["symmetry_thresh"]
    few_colors = color_count <= t["few_color_thresh"]
    no_major = major_count == 0

    criteria["nv_typical_network"] = {"present": typical_network, "score": 0, "detail": "typical network"}
    criteria["nv_symmetric"] = {"present": symmetric, "score": 0, "detail": f"symmetry={symmetry_score:.3f}"}
    criteria["nv_few_colors"] = {"present": few_colors, "score": 0, "detail": f"color_count={color_count}"}

    if typical_network:
        nv_support += 1.5
    if symmetric:
        nv_support += 1.0
    if few_colors:
        nv_support += 1.0
    if no_major:
        nv_support += 0.8

    diagnosis = "nv"

    # 规则主判定
    if (mel_score >= t["mel_total_thresh"] and major_count >= 1) or mel_score >= 5:
        diagnosis = "mel"

    # GLCM在边界区辅助上调
    if diagnosis == "nv" and mel_score == t["mel_total_thresh"] - 1 and major_count >= 1 and has_texture_irregular:
        diagnosis = "mel"

    # nv证据强时回落
    if diagnosis == "mel" and mel_score == t["mel_total_thresh"] and nv_support >= 2.5:
        diagnosis = "nv"

    if diagnosis == "mel":
        confidence = 0.55 + 0.07 * mel_score + 0.08 * major_count - 0.05 * nv_support + 0.05 * texture_irregularity
        confidence = np.clip(confidence, 0.30, 0.98)
    else:
        margin = max(0.0, t["mel_total_thresh"] - mel_score)
        confidence = 0.52 + 0.08 * margin + 0.09 * nv_support - 0.04 * texture_irregularity
        confidence = np.clip(confidence, 0.35, 0.99)

    pos = [k for k, v in criteria.items() if v["present"] and k in (
        "atypical_network", "blue_white_veil", "atypical_vascular",
        "irregular_streaks", "irregular_dots_globules", "regression_structures", "multicolor"
    )]
    nv_pos = [k for k, v in criteria.items() if v["present"] and k.startswith("nv_")]

    reasoning = (
        f"mel_score={mel_score}, major={major_count}, texture={texture_irregularity:.2f}, "
        f"nv_support={nv_support:.2f} -> {diagnosis}; "
        f"mel_positive={pos if pos else ['none']}; nv_positive={nv_pos if nv_pos else ['none']}"
    )

    return {
        "diagnosis": diagnosis,
        "mel_score": int(mel_score),
        "confidence": float(np.clip(confidence, 0, 1)),
        "criteria_detail": criteria,
        "reasoning": reasoning,
    }