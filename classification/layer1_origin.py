"""
第1层决策：组织来源分组（重构版）
目标：减少nv被错误分流到keratinocytic，并保持vasc/df的特异性。
"""

import numpy as np


ORIGIN_VASCULAR = "vascular"
ORIGIN_FIBROUS = "fibrous"
ORIGIN_MELANOCYTIC = "melanocytic"
ORIGIN_KERATINOCYTIC = "keratinocytic"


def classify_origin(features: dict, thresholds: dict = None) -> dict:
    """
    根据第1层特征判定病变的组织来源。
    """
    t = {
        # 兼容旧阈值字段
        "lacunae_score_thresh": 0.48,
        "lacunae_count_thresh": 4,
        "df_pattern_score_thresh": 0.50,
        "df_gradient_thresh": 0.38,
        "network_score_thresh": 0.24,
        "network_coverage_thresh": 0.07,
        "network_score_soft_thresh": 0.14,

        # 新逻辑阈值
        "vascular_evidence_thresh": 0.58,
        "fibrous_evidence_thresh": 0.56,
        "melanocytic_evidence_thresh": 0.38,
        "origin_margin": 0.08,
    }
    if thresholds:
        t.update(thresholds)

    net = features.get("pigment_network", {})
    lac = features.get("lacunae", {})
    cwp = features.get("central_white_patch", {})

    vasc_score = float(lac.get("lacunae_score", 0.0))
    vasc_count = int(lac.get("lacunae_count", 0))
    vasc_color_type = str(lac.get("color_type", "unknown"))

    df_score = float(cwp.get("df_pattern_score", 0.0))
    df_gradient = float(cwp.get("radial_gradient", 0.0))
    has_central_white = bool(cwp.get("has_central_white_patch", False))
    has_periph_net = bool(cwp.get("has_peripheral_network", False))

    mel_origin_score = float(net.get("network_score", 0.0))
    net_coverage = float(net.get("coverage", 0.0))
    gabor_strength = float(net.get("gabor_strength", 0.0))
    has_network = bool(net.get("has_network", False))

    # 证据计算
    vasc_count_norm = np.clip((vasc_count - 1) / 5.0, 0.0, 1.0)
    vasc_red_bonus = 0.1 if vasc_color_type in ("red", "red_lacunae", "hemorrhagic") else 0.0
    vascular_evidence = (
        0.65 * vasc_score +
        0.20 * vasc_count_norm +
        0.15 * vasc_red_bonus
    )

    fibrous_evidence = (
        0.45 * df_score +
        0.15 * np.clip(df_gradient / 0.5, 0.0, 1.0) +
        0.20 * (1.0 if has_central_white else 0.0) +
        0.20 * (1.0 if has_periph_net else 0.0)
    )

    melanocytic_evidence = (
        0.45 * mel_origin_score +
        0.20 * np.clip(net_coverage / 0.2, 0.0, 1.0) +
        0.20 * np.clip(gabor_strength / 0.5, 0.0, 1.0) +
        0.15 * (1.0 if has_network else 0.0)
    )

    scores = {
        "vascular": float(np.clip(vascular_evidence, 0.0, 1.0)),
        "fibrous": float(np.clip(fibrous_evidence, 0.0, 1.0)),
        "melanocytic": float(np.clip(melanocytic_evidence, 0.0, 1.0)),
        "keratinocytic": 0.0,
    }

    margin = t["origin_margin"]

    # 1) vasc: 高特异，要求强证据 + 足够领先
    if (
        vascular_evidence >= t["vascular_evidence_thresh"] and
        vascular_evidence >= fibrous_evidence + margin and
        vascular_evidence >= melanocytic_evidence + margin and
        vasc_score >= t["lacunae_score_thresh"] and
        vasc_count >= t["lacunae_count_thresh"]
    ):
        conf = np.clip(0.55 + 0.45 * vascular_evidence, 0.0, 1.0)
        return {
            "origin": ORIGIN_VASCULAR,
            "confidence": float(conf),
            "scores": scores,
            "reasoning": f"血管证据强: lacunae_score={vasc_score:.3f}, count={vasc_count}",
        }

    # 2) df: 维持高特异条件
    if (
        fibrous_evidence >= t["fibrous_evidence_thresh"] and
        fibrous_evidence >= vascular_evidence + margin and
        fibrous_evidence >= melanocytic_evidence + margin and
        df_score >= t["df_pattern_score_thresh"] and
        has_central_white and has_periph_net and
        df_gradient >= t["df_gradient_thresh"]
    ):
        conf = np.clip(0.55 + 0.45 * fibrous_evidence, 0.0, 1.0)
        return {
            "origin": ORIGIN_FIBROUS,
            "confidence": float(conf),
            "scores": scores,
            "reasoning": (
                f"纤维证据强: df_score={df_score:.3f}, "
                f"central_white={int(has_central_white)}, peripheral_net={int(has_periph_net)}"
            ),
        }

    # 3) melanocytic：只要有中等网络证据，优先于keratinocytic（防止nv误分流）
    if (
        melanocytic_evidence >= t["melanocytic_evidence_thresh"] or
        (
            mel_origin_score >= t["network_score_soft_thresh"] and
            gabor_strength > 0.15 and
            max(vascular_evidence, fibrous_evidence) < 0.45
        )
    ):
        conf = np.clip(0.45 + 0.45 * melanocytic_evidence, 0.0, 1.0)
        return {
            "origin": ORIGIN_MELANOCYTIC,
            "confidence": float(conf),
            "scores": scores,
            "reasoning": (
                f"黑色素来源证据: network_score={mel_origin_score:.3f}, "
                f"coverage={net_coverage:.3f}, gabor={gabor_strength:.3f}"
            ),
        }

    # 4) 歧义区处理：若无明显vasc/df且存在轻度网络线索，仍归melanocytic低置信
    if melanocytic_evidence > 0.20 and max(vascular_evidence, fibrous_evidence) < 0.35:
        conf = np.clip(0.35 + 0.30 * melanocytic_evidence, 0.0, 1.0)
        return {
            "origin": ORIGIN_MELANOCYTIC,
            "confidence": float(conf),
            "scores": scores,
            "reasoning": "歧义区倾向黑色素来源（弱网络证据存在）",
        }

    # 5) 默认角质形成细胞组
    max_other = max(vascular_evidence, fibrous_evidence, melanocytic_evidence)
    ker_conf = np.clip(0.35 + 0.65 * (1.0 - max_other), 0.0, 1.0)
    scores["keratinocytic"] = float(ker_conf)

    return {
        "origin": ORIGIN_KERATINOCYTIC,
        "confidence": float(ker_conf),
        "scores": scores,
        "reasoning": f"其余来源证据不足，归入角质组 (max_other={max_other:.3f})",
    }