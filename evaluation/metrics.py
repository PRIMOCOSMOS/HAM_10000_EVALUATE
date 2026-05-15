"""
分类评估指标模块
计算准确率、召回率、精确率、F1、混淆矩阵等，支持七分类和分组评估。
"""

import numpy as np
from collections import Counter


# HAM10000 七分类标签
ALL_CLASSES = ["nv", "mel", "bkl", "akiec", "bcc", "vasc", "df"]

# 按组织来源分组
ORIGIN_GROUPS = {
    "melanocytic": ["nv", "mel"],
    "keratinocytic": ["bkl", "akiec", "bcc"],
    "vascular": ["vasc"],
    "fibrous": ["df"],
}


def compute_metrics(y_true: list, y_pred: list, classes: list = None) -> dict:
    """
    计算完整的分类评估指标。

    参数:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        classes: 类别列表，默认使用ALL_CLASSES

    返回:
        dict:
            accuracy: float, 总体准确率
            per_class: dict, 每类的precision/recall/f1/support
            macro_avg: dict, 宏平均precision/recall/f1
            weighted_avg: dict, 加权平均precision/recall/f1
            confusion_matrix: np.ndarray, 混淆矩阵
            origin_accuracy: dict, 按组织来源分组的准确率
    """
    if classes is None:
        classes = ALL_CLASSES

    n = len(y_true)
    if n == 0:
        return _empty_metrics(classes)

    # --- 总体准确率 ---
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    # --- 混淆矩阵 ---
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        ti = class_to_idx.get(t)
        pi = class_to_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1

    # --- 每类指标 ---
    per_class = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    # --- 宏平均 ---
    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    macro_avg = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
    }

    # --- 加权平均 ---
    total_support = sum(v["support"] for v in per_class.values())
    if total_support > 0:
        weighted_precision = sum(
            v["precision"] * v["support"] for v in per_class.values()
        ) / total_support
        weighted_recall = sum(
            v["recall"] * v["support"] for v in per_class.values()
        ) / total_support
        weighted_f1 = sum(
            v["f1"] * v["support"] for v in per_class.values()
        ) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    weighted_avg = {
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1": float(weighted_f1),
    }

    # --- 按组织来源分组的准确率 ---
    origin_accuracy = _compute_origin_accuracy(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "per_class": per_class,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
        "confusion_matrix": cm,
        "origin_accuracy": origin_accuracy,
    }


def print_classification_report(metrics: dict, classes: list = None) -> str:
    """
    生成格式化的分类报告字符串。

    参数:
        metrics: compute_metrics的输出
        classes: 类别列表

    返回:
        格式化报告字符串
    """
    if classes is None:
        classes = ALL_CLASSES

    lines = []
    lines.append("=" * 65)
    lines.append("HAM10000 BIP Classification Report")
    lines.append("=" * 65)
    lines.append(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 65)

    per_class = metrics.get("per_class", {})
    for cls in classes:
        info = per_class.get(cls, {})
        lines.append(
            f"{cls:<10} {info.get('precision', 0):.4f}     "
            f"{info.get('recall', 0):.4f}     "
            f"{info.get('f1', 0):.4f}     "
            f"{info.get('support', 0):>5}"
        )

    lines.append("-" * 65)
    macro = metrics.get("macro_avg", {})
    weighted = metrics.get("weighted_avg", {})
    lines.append(
        f"{'macro':<10} {macro.get('precision', 0):.4f}     "
        f"{macro.get('recall', 0):.4f}     "
        f"{macro.get('f1', 0):.4f}"
    )
    lines.append(
        f"{'weighted':<10} {weighted.get('precision', 0):.4f}     "
        f"{weighted.get('recall', 0):.4f}     "
        f"{weighted.get('f1', 0):.4f}"
    )
    lines.append("-" * 65)
    lines.append(f"Overall Accuracy: {metrics.get('accuracy', 0):.4f}")

    # 组织来源分组准确率
    origin_acc = metrics.get("origin_accuracy", {})
    if origin_acc:
        lines.append("")
        lines.append("Origin Group Accuracy:")
        for group, acc in origin_acc.items():
            lines.append(f"  {group:<15} {acc:.4f}")

    lines.append("=" * 65)

    report = "\n".join(lines)
    return report


def _compute_origin_accuracy(y_true: list, y_pred: list) -> dict:
    """计算按组织来源分组的准确率（第1层决策的评估）"""
    def get_origin(label):
        for origin, members in ORIGIN_GROUPS.items():
            if label in members:
                return origin
        return "unknown"

    origin_true = [get_origin(t) for t in y_true]
    origin_pred = [get_origin(p) for p in y_pred]

    origin_accuracy = {}
    for group in ORIGIN_GROUPS:
        indices = [i for i, ot in enumerate(origin_true) if ot == group]
        if len(indices) == 0:
            origin_accuracy[group] = 0.0
            continue
        correct = sum(1 for i in indices if origin_true[i] == origin_pred[i])
        origin_accuracy[group] = correct / len(indices)

    return origin_accuracy


def _empty_metrics(classes):
    return {
        "accuracy": 0.0,
        "per_class": {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for c in classes},
        "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "weighted_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "confusion_matrix": np.zeros((len(classes), len(classes)), dtype=np.int32),
        "origin_accuracy": {},
    }