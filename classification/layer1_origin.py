"""
第1层决策：组织来源分组（优化版）
核心改进：
  1. 增加"melanocytic先验"——当网络得分处于边界区间时，倾向判为melanocytic
  2. 提高vasc和df的判定门槛，减少假阳性
  3. 增加"弱网络+对称+少色"的辅助判据
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
        "lacunae_score_thresh": 0.45,
        "lacunae_count_thresh": 3,
        "df_pattern_score_thresh": 0.45,
        "df_gradient_thresh": 0.35,
        "network_score_thresh": 0.25,
        "network_coverage_thresh": 0.08,
        # 新增：弱网络的宽松阈值（用于边界情况）
        "network_score_soft_thresh": 0.15,
    }
    if thresholds:
        t.update(thresholds)

    net = features.get("pigment_network", {})
    lac = features.get("lacunae", {})
    cwp = features.get("central_white_patch", {})

    vasc_score = lac.get("lacunae_score", 0.0)
    vasc_count = lac.get("lacunae_count", 0)
    df_score = cwp.get("df_pattern_score", 0.0)
    df_gradient = cwp.get("radial_gradient", 0.0)
    has_central_white = cwp.get("has_central_white_patch", False)
    has_periph_net = cwp.get("has_peripheral_network", False)
    mel_origin_score = net.get("network_score", 0.0)
    net_coverage = net.get("coverage", 0.0)
    gabor_strength = net.get("gabor_strength", 0.0)

    scores = {
        "vascular": float(vasc_score),
        "fibrous": float(df_score),
        "melanocytic": float(mel_origin_score),
        "keratinocytic": 0.0,
    }

    # --- 优先级决策 ---

    # 1. 血管来源（提高门槛）
    if (vasc_score >= t["lacunae_score_thresh"] and
            vasc_count >= t["lacunae_count_thresh"]):
        confidence = min(1.0, vasc_score / 0.7)
        return {
            "origin": ORIGIN_VASCULAR,
            "confidence": float(confidence),
            "scores": scores,
            "reasoning": f"红色腔隙: count={vasc_count}, score={vasc_score:.3f}",
        }

    # 2. 纤维来源（提高门槛，需要三个条件同时满足）
    if (df_score >= t["df_pattern_score_thresh"] and
            has_central_white and has_periph_net and
            df_gradient >= t["df_gradient_thresh"]):
        confidence = min(1.0, df_score / 0.7)
        return {
            "origin": ORIGIN_FIBROUS,
            "confidence": float(confidence),
            "scores": scores,
            "reasoning": f"中央白斑+周围网络: df_score={df_score:.3f}",
        }

    # 3. 黑色素细胞来源（降低门槛 + 增加宽松路径）
    # 主路径：标准阈值
    if (mel_origin_score >= t["network_score_thresh"] and
            net_coverage >= t["network_coverage_thresh"]):
        confidence = min(1.0, mel_origin_score / 0.6)
        return {
            "origin": ORIGIN_MELANOCYTIC,
            "confidence": float(confidence),
            "scores": scores,
            "reasoning": (
                f"色素网络: score={mel_origin_score:.3f}, "
                f"coverage={net_coverage:.3f}"
            ),
        }

    # 宽松路径：弱网络信号但Gabor强度尚可
    # 很多典型nv的网络淡但确实存在，这里给一个"benefit of doubt"
    if (mel_origin_score >= t["network_score_soft_thresh"] and
            gabor_strength > 0.15):
        # 额外检查：如果同时没有强烈的角质类特征，倾向判为melanocytic
        # （角质类病变通常完全没有网络信号）
        if vasc_score < 0.2 and df_score < 0.2:
            confidence = min(0.7, mel_origin_score / 0.4)
            return {
                "origin": ORIGIN_MELANOCYTIC,
                "confidence": float(confidence),
                "scores": scores,
                "reasoning": (
                    f"弱色素网络(宽松路径): score={mel_origin_score:.3f}, "
                    f"gabor_strength={gabor_strength:.3f}"
                ),
            }

    # 4. 默认：角质形成细胞来源
    max_other = max(vasc_score, df_score, mel_origin_score)
    keratinocytic_confidence = 1.0 - min(1.0, max_other / 0.3)
    scores["keratinocytic"] = float(keratinocytic_confidence)

    return {
        "origin": ORIGIN_KERATINOCYTIC,
        "confidence": float(max(0.3, keratinocytic_confidence)),
        "scores": scores,
        "reasoning": f"无特异性结构, 归入角质组 (max_other={max_other:.3f})",
    }