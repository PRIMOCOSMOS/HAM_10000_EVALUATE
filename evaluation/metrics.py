"""
分类评估指标模块
计算准确率、召回率、精确率、F1、混淆矩阵等，支持七分类和分组评估。
新增：nv安全性指标，约束“nv尽量不误诊”的优化方向。
"""

import numpy as np


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
    """
    if classes is None:
        classes = ALL_CLASSES

    n = len(y_true)
    if n == 0:
        return _empty_metrics(classes)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        ti = class_to_idx.get(t)
        pi = class_to_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1

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

    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    macro_avg = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
    }

    total_support = sum(v["support"] for v in per_class.values())
    if total_support > 0:
        weighted_precision = sum(v["precision"] * v["support"] for v in per_class.values()) / total_support
        weighted_recall = sum(v["recall"] * v["support"] for v in per_class.values()) / total_support
        weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    weighted_avg = {
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1": float(weighted_f1),
    }

    origin_accuracy = _compute_origin_accuracy(y_true, y_pred)
    safety_metrics = _compute_nv_safety_metrics(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "per_class": per_class,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
        "confusion_matrix": cm,
        "origin_accuracy": origin_accuracy,
        "safety_metrics": safety_metrics,
    }


def print_classification_report(metrics: dict, classes: list = None) -> str:
    """
    生成格式化的分类报告字符串。
    """
    if classes is None:
        classes = ALL_CLASSES

    lines = []
    lines.append("=" * 70)
    lines.append("HAM10000 BIP Classification Report")
    lines.append("=" * 70)
    lines.append(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 70)

    per_class = metrics.get("per_class", {})
    for cls in classes:
        info = per_class.get(cls, {})
        lines.append(
            f"{cls:<10} {info.get('precision', 0):.4f}     "
            f"{info.get('recall', 0):.4f}     "
            f"{info.get('f1', 0):.4f}     "
            f"{info.get('support', 0):>5}"
        )

    lines.append("-" * 70)
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
    lines.append("-" * 70)
    lines.append(f"Overall Accuracy: {metrics.get('accuracy', 0):.4f}")

    origin_acc = metrics.get("origin_accuracy", {})
    if origin_acc:
        lines.append("")
        lines.append("Origin Group Accuracy:")
        for group, acc in origin_acc.items():
            lines.append(f"  {group:<15} {acc:.4f}")

    safety = metrics.get("safety_metrics", {})
    if safety:
        lines.append("")
        lines.append("NV Safety Metrics:")
        lines.append(f"  nv_misdiagnosis_rate   {safety.get('nv_misdiagnosis_rate', 0):.4f}")
        lines.append(f"  nv_to_mel_rate         {safety.get('nv_to_mel_rate', 0):.4f}")
        lines.append(f"  nv_to_keratin_rate     {safety.get('nv_to_keratin_rate', 0):.4f}")
        lines.append(f"  nv_to_other_rate       {safety.get('nv_to_other_rate', 0):.4f}")

    lines.append("=" * 70)
    return "\n".join(lines)


def _compute_origin_accuracy(y_true: list, y_pred: list) -> dict:
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


def _compute_nv_safety_metrics(y_true: list, y_pred: list) -> dict:
    nv_idx = [i for i, t in enumerate(y_true) if t == "nv"]
    n_nv = len(nv_idx)
    if n_nv == 0:
        return {
            "nv_misdiagnosis_rate": 0.0,
            "nv_to_mel_rate": 0.0,
            "nv_to_keratin_rate": 0.0,
            "nv_to_other_rate": 0.0,
        }

    wrong = [i for i in nv_idx if y_pred[i] != "nv"]
    nv_misdiagnosis_rate = len(wrong) / n_nv

    nv_to_mel = sum(1 for i in nv_idx if y_pred[i] == "mel") / n_nv
    nv_to_keratin = sum(1 for i in nv_idx if y_pred[i] in ("bkl", "akiec", "bcc")) / n_nv
    nv_to_other = sum(1 for i in nv_idx if y_pred[i] in ("vasc", "df")) / n_nv

    return {
        "nv_misdiagnosis_rate": float(nv_misdiagnosis_rate),
        "nv_to_mel_rate": float(nv_to_mel),
        "nv_to_keratin_rate": float(nv_to_keratin),
        "nv_to_other_rate": float(nv_to_other),
    }


def _empty_metrics(classes):
    return {
        "accuracy": 0.0,
        "per_class": {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for c in classes},
        "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "weighted_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "confusion_matrix": np.zeros((len(classes), len(classes)), dtype=np.int32),
        "origin_accuracy": {},
        "safety_metrics": {
            "nv_misdiagnosis_rate": 0.0,
            "nv_to_mel_rate": 0.0,
            "nv_to_keratin_rate": 0.0,
            "nv_to_other_rate": 0.0,
        },
    }