"""
诊断报告生成模块
为单张图像生成可解释的诊断报告，为批量评估生成汇总报告。
"""

import numpy as np
from datetime import datetime


def generate_single_report(pipeline_result: dict, image_id: str = "") -> str:
    """
    为单张图像生成文本格式的诊断报告。

    参数:
        pipeline_result: run_pipeline的输出
        image_id: 图像标识符

    返回:
        格式化的诊断报告字符串
    """
    diag = pipeline_result["diagnosis"]
    conf = pipeline_result["confidence"]
    origin = pipeline_result["origin"]
    path = pipeline_result["decision_path"]
    features = pipeline_result["all_features"]

    lines = []
    lines.append("┌" + "─" * 58 + "┐")
    lines.append(f"│ 诊断报告 {image_id:<47} │")
    lines.append("├" + "─" * 58 + "┤")
    lines.append(f"│ 诊断结果: {_diagnosis_name(diag):<45} │")
    lines.append(f"│ 置信度:   {conf:.3f}{'':>44} │")
    lines.append(f"│ 组织来源: {origin:<45} │")
    lines.append("├" + "─" * 58 + "┤")
    lines.append("│ 决策路径:                                                  │")

    for step in path:
        # 截断过长行
        step_display = step[:56]
        lines.append(f"│   {step_display:<55} │")

    lines.append("├" + "─" * 58 + "┤")
    lines.append("│ 关键特征得分:                                              │")

    # 提取关键得分展示
    key_scores = _extract_key_scores(features, origin)
    for name, value in key_scores.items():
        if isinstance(value, bool):
            val_str = "是" if value else "否"
        elif isinstance(value, float):
            val_str = f"{value:.3f}"
        else:
            val_str = str(value)
        line = f"│   {name:<30} {val_str:<25} │"
        lines.append(line)

    lines.append("└" + "─" * 58 + "┘")

    return "\n".join(lines)


def generate_batch_summary(results: list, y_true: list = None) -> str:
    """
    为批量评估生成汇总报告。

    参数:
        results: list of run_pipeline输出
        y_true: 真实标签列表（可选，用于计算准确率）

    返回:
        汇总报告字符串
    """
    n = len(results)
    if n == 0:
        return "无结果"

    # 统计诊断分布
    diag_counts = {}
    origin_counts = {}
    confidences = []

    for r in results:
        diag = r["diagnosis"]
        origin = r["origin"]
        diag_counts[diag] = diag_counts.get(diag, 0) + 1
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        confidences.append(r["confidence"])

    lines = []
    lines.append("=" * 50)
    lines.append(f"批量评估汇总 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    lines.append(f"总样本数: {n}")
    lines.append("=" * 50)

    # 诊断分布
    lines.append("诊断分布:")
    for cls in ["nv", "mel", "bkl", "akiec", "bcc", "vasc", "df"]:
        count = diag_counts.get(cls, 0)
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  {cls:<6} {count:>5} ({pct:>5.1f}%) {bar}")

    # 组织来源分布
    lines.append("组织来源分布:")
    for origin in ["melanocytic", "keratinocytic", "vascular", "fibrous"]:
        count = origin_counts.get(origin, 0)
        pct = count / n * 100
        lines.append(f"  {origin:<15} {count:>5} ({pct:>5.1f}%)")

    # 置信度统计
    conf_arr = np.array(confidences)
    lines.append(f"置信度统计:")
    lines.append(f"  均值: {np.mean(conf_arr):.3f}")
    lines.append(f"  中位数: {np.median(conf_arr):.3f}")
    lines.append(f"  最小值: {np.min(conf_arr):.3f}")
    lines.append(f"  最大值: {np.max(conf_arr):.3f}")
    lines.append(f"  低置信(<0.5)样本数: {np.sum(conf_arr < 0.5)}")

    # 如果有真实标签，计算准确率
    if y_true is not None and len(y_true) == n:
        y_pred = [r["diagnosis"] for r in results]
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / n
        lines.append(f"与真实标签对比:")
        lines.append(f"  总体准确率: {accuracy:.4f} ({correct}/{n})")

        # 每类准确率
        lines.append("  各类准确率:")
        for cls in ["nv", "mel", "bkl", "akiec", "bcc", "vasc", "df"]:
            cls_indices = [i for i, t in enumerate(y_true) if t == cls]
            if len(cls_indices) == 0:
                continue
            cls_correct = sum(1 for i in cls_indices if y_pred[i] == cls)
            cls_acc = cls_correct / len(cls_indices)
            lines.append(f"    {cls:<6} {cls_acc:.4f} ({cls_correct}/{len(cls_indices)})")

    lines.append("=" * 50)
    return "\n".join(lines)


def _diagnosis_name(code: str) -> str:
    """诊断代码转中文名称"""
    names = {
        "nv": "黑色素细胞痣 (Melanocytic Nevus)",
        "mel": "黑色素瘤 (Melanoma)",
        "bkl": "良性角化病样病变 (Benign Keratosis)",
        "akiec": "光化性角化病 (Actinic Keratosis)",
        "bcc": "基底细胞癌 (Basal Cell Carcinoma)",
        "vasc": "血管性病变 (Vascular Lesion)",
        "df": "皮肤纤维瘤 (Dermatofibroma)",
    }
    return names.get(code, code)


def _extract_key_scores(features: dict, origin: str) -> dict:
    """根据组织来源提取最相关的关键得分用于报告展示"""
    key_scores = {}

    # 第1层通用特征
    net = features.get("pigment_network", {})
    lac = features.get("lacunae", {})
    cwp = features.get("central_white_patch", {})

    key_scores["色素网络得分"] = net.get("network_score", 0.0)
    key_scores["腔隙得分"] = lac.get("lacunae_score", 0.0)
    key_scores["DF模式得分"] = cwp.get("df_pattern_score", 0.0)

    # 第2层按组展示
    if origin == "melanocytic":
        sym = features.get("symmetry", {})
        color = features.get("color_variegation", {})
        bwv = features.get("blue_white_veil", {})
        streaks = features.get("streaks", {})
        dots = features.get("dots_globules", {})
        reg = features.get("regression", {})

        key_scores["对称性评分"] = sym.get("symmetry_score", 0.0)
        key_scores["颜色种类"] = color.get("color_count", 0)
        key_scores["蓝白幕得分"] = bwv.get("bwv_score", 0.0)
        key_scores["条纹得分"] = streaks.get("streak_score", 0.0)
        key_scores["点球不规则性"] = dots.get("irregularity_score", 0.0)
        key_scores["回归结构得分"] = reg.get("regression_score", 0.0)
        key_scores["网络规则性"] = net.get("regularity_score", 0.0)

    elif origin == "keratinocytic":
        milia = features.get("milia_cysts", {})
        comedo = features.get("comedo_openings", {})
        fiss = features.get("fissures_ridges", {})
        border = features.get("border_sharpness", {})
        straw = features.get("strawberry_pattern", {})
        tex = features.get("surface_texture", {})
        arbor = features.get("arborizing_vessels", {})
        nests = features.get("ovoid_nests", {})

        key_scores["粟粒样囊肿数"] = milia.get("milia_count", 0)
        key_scores["粉刺样开口数"] = comedo.get("comedo_count", 0)
        key_scores["脑回样得分"] = fiss.get("cerebriform_score", 0.0)
        key_scores["边界锐利度"] = border.get("sharpness_score", 0.0)
        key_scores["草莓样得分"] = straw.get("strawberry_score", 0.0)
        key_scores["粗糙度得分"] = tex.get("roughness_score", 0.0)
        key_scores["树枝状血管得分"] = arbor.get("arborizing_score", 0.0)
        key_scores["卵圆巢数"] = nests.get("nest_count", 0)

    elif origin == "vascular":
        key_scores["腔隙数量"] = lac.get("lacunae_count", 0)
        key_scores["腔隙面积比"] = lac.get("lacunae_area_ratio", 0.0)
        key_scores["主要颜色"] = lac.get("dominant_color", "none")

    elif origin == "fibrous":
        key_scores["中央白斑"] = cwp.get("has_central_white_patch", False)
        key_scores["周围网络"] = cwp.get("has_peripheral_network", False)
        key_scores["径向梯度"] = cwp.get("radial_gradient", 0.0)

    return key_scores