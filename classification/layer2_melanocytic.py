"""
第2层决策：黑色素细胞组内细分 (nv vs mel)
基于七点检查法(7-Point Checklist)的加权评分逻辑：
  - 主要标准(2分): 非典型网络/蓝白幕/非典型血管
  - 次要标准(1分): 不规则条纹/不规则点球/回归结构/多色
  - 总分≥3 → 黑色素瘤(mel)
  - 总分<3 → 黑色素细胞痣(nv)

同时参考Menzies方法的阴性特征（对称性+颜色均匀度）作为辅助。
"""

import numpy as np


def classify_melanocytic(features: dict, thresholds: dict = None) -> dict:
    """
    在黑色素细胞组内区分 nv 和 mel。

    参数:
        features: dict, 包含以下键:
            - pigment_network: dict
            - symmetry: dict
            - color_variegation: dict
            - blue_white_veil: dict
            - streaks: dict
            - dots_globules: dict
            - regression: dict
            - vascular_pattern: dict

    返回:
        dict:
            diagnosis: str, 'nv' 或 'mel'
            mel_score: int, 七点法总分 (0-10)
            confidence: float (0-1)
            criteria_detail: dict, 各标准的判定明细
            reasoning: str
    """
    t = {
        "symmetry_thresh": 1.5,
        "color_count_thresh": 4,
        "bwv_score_thresh": 0.3,
        "streak_score_thresh": 0.3,
        "dots_irregularity_thresh": 0.4,
        "regression_score_thresh": 0.3,
        "polymorphism_thresh": 0.4,
        "network_regularity_thresh": 0.5,
        "mel_total_thresh": 3,
    }
    if thresholds:
        t.update(thresholds)

    # 提取各模块结果
    net = features.get("pigment_network", {})
    sym = features.get("symmetry", {})
    color = features.get("color_variegation", {})
    bwv = features.get("blue_white_veil", {})
    streaks = features.get("streaks", {})
    dots = features.get("dots_globules", {})
    reg = features.get("regression", {})
    vasc = features.get("vascular_pattern", {})

    # --- 七点检查法评分 ---
    criteria = {}
    mel_score = 0

    # 主要标准1: 非典型色素网络 (2分)
    # 网络存在但规则性低 → 非典型
    network_regularity = net.get("regularity_score", 1.0)
    has_network = net.get("has_network", False)
    atypical_network = has_network and network_regularity < t["network_regularity_thresh"]
    criteria["atypical_network"] = {
        "present": atypical_network,
        "score": 2 if atypical_network else 0,
        "detail": f"regularity={network_regularity:.3f} (thresh<{t['network_regularity_thresh']})",
    }
    if atypical_network:
        mel_score += 2

    # 主要标准2: 蓝白幕 (2分)
    bwv_val = bwv.get("bwv_score", 0.0)
    has_bwv = bwv_val >= t["bwv_score_thresh"]
    criteria["blue_white_veil"] = {
        "present": has_bwv,
        "score": 2 if has_bwv else 0,
        "detail": f"bwv_score={bwv_val:.3f} (thresh>={t['bwv_score_thresh']})",
    }
    if has_bwv:
        mel_score += 2

    # 主要标准3: 非典型血管模式 (2分)
    poly_score = vasc.get("polymorphism_score", 0.0)
    has_atypical_vessels = poly_score >= t["polymorphism_thresh"]
    criteria["atypical_vascular"] = {
        "present": has_atypical_vessels,
        "score": 2 if has_atypical_vessels else 0,
        "detail": f"polymorphism={poly_score:.3f} (thresh>={t['polymorphism_thresh']})",
    }
    if has_atypical_vessels:
        mel_score += 2

    # 次要标准1: 不规则条纹 (1分)
    streak_val = streaks.get("streak_score", 0.0)
    has_streaks = streak_val >= t["streak_score_thresh"]
    criteria["irregular_streaks"] = {
        "present": has_streaks,
        "score": 1 if has_streaks else 0,
        "detail": f"streak_score={streak_val:.3f} (thresh>={t['streak_score_thresh']})",
    }
    if has_streaks:
        mel_score += 1

    # 次要标准2: 不规则点/球 (1分)
    dots_irreg = dots.get("irregularity_score", 0.0)
    has_irreg_dots = dots_irreg >= t["dots_irregularity_thresh"]
    criteria["irregular_dots_globules"] = {
        "present": has_irreg_dots,
        "score": 1 if has_irreg_dots else 0,
        "detail": f"irregularity={dots_irreg:.3f} (thresh>={t['dots_irregularity_thresh']})",
    }
    if has_irreg_dots:
        mel_score += 1

    # 次要标准3: 回归结构 (1分)
    reg_val = reg.get("regression_score", 0.0)
    has_regression = reg_val >= t["regression_score_thresh"]
    criteria["regression_structures"] = {
        "present": has_regression,
        "score": 1 if has_regression else 0,
        "detail": f"regression_score={reg_val:.3f} (thresh>={t['regression_score_thresh']})",
    }
    if has_regression:
        mel_score += 1

    # 次要标准4: 多色 (1分) — 颜色≥4种
    color_count = color.get("color_count", 1)
    has_multicolor = color_count >= t["color_count_thresh"]
    criteria["multicolor"] = {
        "present": has_multicolor,
        "score": 1 if has_multicolor else 0,
        "detail": f"color_count={color_count} (thresh>={t['color_count_thresh']})",
    }
    if has_multicolor:
        mel_score += 1

    # --- 辅助参考: Menzies阴性特征 ---
    symmetry_score = sym.get("symmetry_score", 0.0)
    is_asymmetric = symmetry_score >= t["symmetry_thresh"]

    # --- 最终判定 ---
    diagnosis = "mel" if mel_score >= t["mel_total_thresh"] else "nv"

    # 置信度计算
    if diagnosis == "mel":
        # 分数越高越确信
        confidence = min(1.0, mel_score / 7.0)
    else:
        # 分数越低越确信是nv
        confidence = 1.0 - mel_score / t["mel_total_thresh"]
        confidence = max(0.3, confidence)

    # 对称性作为辅助修正
    if diagnosis == "nv" and is_asymmetric and mel_score == t["mel_total_thresh"] - 1:
        # 边界情况：差1分但不对称，降低nv置信度
        confidence *= 0.7

    reasoning_parts = []
    for name, info in criteria.items():
        if info["present"]:
            reasoning_parts.append(f"{name}(+{info['score']})")

    reasoning = (
        f"七点法总分={mel_score}/{t['mel_total_thresh']} → {diagnosis}; "
        f"阳性标准: {', '.join(reasoning_parts) if reasoning_parts else '无'}; "
        f"对称性={symmetry_score:.2f}"
    )

    return {
        "diagnosis": diagnosis,
        "mel_score": int(mel_score),
        "confidence": float(np.clip(confidence, 0, 1)),
        "criteria_detail": criteria,
        "reasoning": reasoning,
    }