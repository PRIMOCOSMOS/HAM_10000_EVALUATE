"""
HAM10000 BIP分类系统 - 主入口
功能：
  1. 单图诊断模式：对单张图像执行完整pipeline并输出诊断报告
  2. 批量评估模式：对整个数据集执行分类并计算准确率指标
  3. 快速验证模式：随机抽样子集做快速正确率验证

用法：
  python main.py --mode single --image <image_id>
  python main.py --mode evaluate [--num_samples N]
  python main.py --mode validate --num_samples 100
"""

import os
import sys
import argparse
import time
import csv
import json
import numpy as np
from pathlib import Path

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from classification.pipeline import run_pipeline
from evaluation.metrics import compute_metrics, print_classification_report
from evaluation.report_generator import generate_single_report, generate_batch_summary


# ============================================================
# 数据加载工具
# ============================================================

def load_metadata(metadata_path: str = None) -> list:
    """
    加载HAM10000元数据CSV。

    返回:
        list of dict, 每个元素包含:
            image_id, lesion_id, dx, dx_type, age, sex, localization
    """
    if metadata_path is None:
        metadata_path = config.METADATA_PATH

    records = []
    # 尝试带/不带.csv后缀
    paths_to_try = [metadata_path, metadata_path + ".csv"]
    actual_path = None
    for p in paths_to_try:
        if os.path.isfile(p):
            actual_path = p
            break

    if actual_path is None:
        raise FileNotFoundError(
            f"找不到元数据文件，尝试路径: {paths_to_try}"
        )

    with open(actual_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "image_id": row["image_id"].strip(),
                "lesion_id": row["lesion_id"].strip(),
                "dx": row["dx"].strip(),
                "dx_type": row["dx_type"].strip(),
                "age": row.get("age", "").strip(),
                "sex": row.get("sex", "").strip(),
                "localization": row.get("localization", "").strip(),
            })

    print(f"[数据] 加载元数据: {len(records)} 条记录, 来源: {actual_path}")
    return records


def find_image_path(image_id: str, image_dirs: list = None) -> str:
    """
    在多个图像文件夹中查找指定image_id对应的图像文件。

    参数:
        image_id: 图像ID (如 'ISIC_0027419')
        image_dirs: 图像文件夹列表

    返回:
        图像文件的完整路径

    异常:
        FileNotFoundError: 找不到对应图像
    """
    if image_dirs is None:
        image_dirs = config.IMAGE_DIRS

    filename = f"{image_id}.jpg"

    for dir_path in image_dirs:
        full_path = os.path.join(dir_path, filename)
        if os.path.isfile(full_path):
            return full_path

    raise FileNotFoundError(
        f"找不到图像 {filename}，搜索目录: {image_dirs}"
    )


def load_image(image_path: str) -> np.ndarray:
    """加载图像为RGB格式的numpy数组"""
    import cv2
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise IOError(f"无法读取图像: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# ============================================================
# 运行模式
# ============================================================

def run_single(image_id: str):
    """单图诊断模式"""
    print(f"{'='*60}")
    print(f"单图诊断模式: {image_id}")
    print(f"{'='*60}")

    # 查找并加载图像
    image_path = find_image_path(image_id)
    print(f"[加载] {image_path}")
    image_rgb = load_image(image_path)
    print(f"[图像] 尺寸: {image_rgb.shape}")

    # 查找真实标签（如果有）
    metadata = load_metadata()
    true_label = None
    for record in metadata:
        if record["image_id"] == image_id:
            true_label = record["dx"]
            break

    # 执行pipeline
    t0 = time.time()
    result = run_pipeline(image_rgb, config=config.PIPELINE_CONFIG)
    elapsed = time.time() - t0

    # 输出报告
    report = generate_single_report(result, image_id=image_id)
    print(report)
    print(f"[耗时] {elapsed:.2f}s")

    if true_label:
        match = "✓ 正确" if result["diagnosis"] == true_label else "✗ 错误"
        print(f"[验证] 真实标签: {true_label}, 预测: {result['diagnosis']} → {match}")

    # 保存报告
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    report_path = os.path.join(config.REPORT_DIR, f"{image_id}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[保存] 报告已保存至: {report_path}")

    return result


def run_evaluate(num_samples: int = None, random_seed: int = 42):
    """
    批量评估模式：对数据集执行分类并计算完整指标。

    参数:
        num_samples: 评估样本数，None=全量
        random_seed: 随机种子（用于子集抽样）
    """
    print(f"{'='*60}")
    print(f"批量评估模式")
    print(f"{'='*60}")

    metadata = load_metadata()

    # 子集抽样
    if num_samples is not None and num_samples < len(metadata):
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(metadata), size=num_samples, replace=False)
        metadata = [metadata[i] for i in sorted(indices)]
        print(f"[抽样] 随机选取 {num_samples} 个样本 (seed={random_seed})")

    total = len(metadata)
    print(f"[评估] 共 {total} 个样本")

    y_true = []
    y_pred = []
    results = []
    errors = []

    t_start = time.time()

    for idx, record in enumerate(metadata):
        image_id = record["image_id"]
        true_label = record["dx"]

        try:
            image_path = find_image_path(image_id)
            image_rgb = load_image(image_path)
            result = run_pipeline(image_rgb, config=config.PIPELINE_CONFIG)

            y_true.append(true_label)
            y_pred.append(result["diagnosis"])
            results.append(result)

        except Exception as e:
            errors.append({"image_id": image_id, "error": str(e)})
            continue

        # 进度显示
        if (idx + 1) % 50 == 0 or idx == total - 1:
            elapsed = time.time() - t_start
            speed = (idx + 1) / elapsed
            correct_so_far = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            acc_so_far = correct_so_far / len(y_true) if y_true else 0
            eta = (total - idx - 1) / speed if speed > 0 else 0
            print(
                f"  [{idx+1}/{total}] "
                f"acc={acc_so_far:.4f} "
                f"speed={speed:.1f}img/s "
                f"ETA={eta:.0f}s "
                f"errors={len(errors)}"
            )

    total_time = time.time() - t_start

    # 计算指标
    metrics = compute_metrics(y_true, y_pred)
    report_str = print_classification_report(metrics)
    print(f"{report_str}")
    print(f"[完成] 总耗时: {total_time:.1f}s, 平均: {total_time/max(len(y_true),1):.2f}s/img")

    if errors:
        print(f"[警告] {len(errors)} 个样本处理失败")

    # 保存结果
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 保存指标JSON
    metrics_save = {
        "accuracy": metrics["accuracy"],
        "per_class": metrics["per_class"],
        "macro_avg": metrics["macro_avg"],
        "weighted_avg": metrics["weighted_avg"],
        "origin_accuracy": metrics["origin_accuracy"],
        "total_samples": len(y_true),
        "total_errors": len(errors),
        "total_time_seconds": total_time,
    }
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_save, f, indent=2, ensure_ascii=False)
    print(f"[保存] 指标已保存至: {metrics_path}")

    # 保存分类报告
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[保存] 报告已保存至: {report_path}")

    # 保存混淆矩阵
    if config.EVALUATION.get("save_confusion_matrix", True):
        cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.csv")
        _save_confusion_matrix(metrics["confusion_matrix"], cm_path)
        print(f"[保存] 混淆矩阵已保存至: {cm_path}")

    # 保存逐样本预测结果
    predictions_path = os.path.join(config.RESULTS_DIR, "predictions.csv")
    with open(predictions_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "true_label", "predicted_label", "correct", "confidence"])
        for i, record in enumerate(metadata[:len(y_true)]):
            writer.writerow([
                record["image_id"],
                y_true[i],
                y_pred[i],
                int(y_true[i] == y_pred[i]),
                f"{results[i]['confidence']:.4f}",
            ])
    print(f"[保存] 逐样本预测已保存至: {predictions_path}")

    # 批量汇总
    summary = generate_batch_summary(results, y_true)
    summary_path = os.path.join(config.RESULTS_DIR, "batch_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[保存] 批量汇总已保存至: {summary_path}")

    return metrics


def run_validate(num_samples: int = 100, random_seed: int = 42):
    """
    快速验证模式：随机抽样少量样本快速检查正确率。
    适合调参时快速迭代。
    """
    print(f"{'='*60}")
    print(f"快速验证模式 (n={num_samples})")
    print(f"{'='*60}")

    metrics = run_evaluate(num_samples=num_samples, random_seed=random_seed)

    # 输出简洁的验证结果
    print(f"{'─'*40}")
    print(f"快速验证结果:")
    print(f"  总体准确率: {metrics['accuracy']:.4f}")
    print(f"  宏平均F1:   {metrics['macro_avg']['f1']:.4f}")
    print(f"  加权F1:     {metrics['weighted_avg']['f1']:.4f}")
    print(f"{'─'*40}")

    return metrics


# ============================================================
# 辅助函数
# ============================================================

def _save_confusion_matrix(cm: np.ndarray, path: str):
    """保存混淆矩阵为CSV"""
    from evaluation.metrics import ALL_CLASSES
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # 表头
        writer.writerow(["true\\pred"] + ALL_CLASSES)
        # 数据行
        for i, cls in enumerate(ALL_CLASSES):
            writer.writerow([cls] + [int(cm[i, j]) for j in range(len(ALL_CLASSES))])


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="HAM10000 BIP分类系统 - 基于皮肤镜学先验的传统图像处理方法"
    )
    parser.add_argument(
        "--mode", type=str, default="validate",
        choices=["single", "evaluate", "validate"],
        help="运行模式: single=单图诊断, evaluate=全量评估, validate=快速验证"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="单图模式下的image_id (如 ISIC_0027419)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="评估/验证的样本数量 (默认: validate=100, evaluate=全量)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if args.mode == "single":
        if args.image is None:
            # 默认取第一张图做演示
            metadata = load_metadata()
            args.image = metadata[0]["image_id"]
            print(f"[提示] 未指定image_id，使用第一张: {args.image}")
        run_single(args.image)

    elif args.mode == "evaluate":
        num = args.num_samples or config.EVALUATION.get("num_samples")
        run_evaluate(num_samples=num, random_seed=args.seed)

    elif args.mode == "validate":
        num = args.num_samples or 100
        run_validate(num_samples=num, random_seed=args.seed)


if __name__ == "__main__":
    main()