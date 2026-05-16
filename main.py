"""
HAM10000 主入口
模式:
- single: 单图诊断
- validate: 快速验证（默认按config.VALIDATION["validate_num_samples"]）
- evaluate: 批量评估（默认按config.VALIDATION["evaluate_num_samples"]）
"""

import os
import sys
import csv
import json
import time
import argparse
import random
from copy import deepcopy

import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from classification.pipeline import run_pipeline  # noqa: E402
from evaluation.metrics import compute_metrics, print_classification_report  # noqa: E402
from evaluation.report_generator import generate_single_report, generate_batch_summary  # noqa: E402


def _find_existing_path(base_path: str) -> str:
    candidates = [base_path, base_path + ".csv"]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"File not found, tried: {candidates}")


def load_metadata():
    md_path = _find_existing_path(config.PATHS["metadata"])
    rows = []
    with open(md_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image_id": r["image_id"].strip(),
                "lesion_id": r["lesion_id"].strip(),
                "dx": r["dx"].strip(),
            })
    return rows, md_path


def find_image_path(image_id: str) -> str:
    fn = f"{image_id}.jpg"
    for d in config.PATHS["image_dirs"]:
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Image not found: {image_id}")


def load_image_rgb(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        raise IOError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_confusion_matrix(cm: np.ndarray, path: str):
    classes = ["nv", "mel", "bkl", "akiec", "bcc", "vasc", "df"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + classes)
        for i, cls in enumerate(classes):
            w.writerow([cls] + [int(cm[i, j]) for j in range(len(classes))])


def run_single(image_id: str):
    os.makedirs(config.PATHS["report_dir"], exist_ok=True)

    img_path = find_image_path(image_id)
    img = load_image_rgb(img_path)

    run_cfg = deepcopy(config.PIPELINE_CONFIG)
    result = run_pipeline(img, config=run_cfg)

    report = generate_single_report(result, image_id=image_id)
    print(report)

    report_path = os.path.join(config.PATHS["report_dir"], f"{image_id}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[save] {report_path}")

    return result


def run_evaluate(num_samples=None, seed=None):
    os.makedirs(config.PATHS["results_dir"], exist_ok=True)

    if seed is None:
        seed = int(config.VALIDATION["seed"])

    rows, md_path = load_metadata()
    print(f"[meta] source={md_path} total={len(rows)}")

    if num_samples is None:
        num_samples = config.VALIDATION["evaluate_num_samples"]

    if isinstance(num_samples, int) and num_samples > 0 and num_samples < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:num_samples]
        print(f"[eval] subset={len(rows)} seed={seed}")

    run_cfg = deepcopy(config.PIPELINE_CONFIG)
    results = []
    y_true = []
    y_pred = []
    errors = []

    t0 = time.time()
    total = len(rows)

    for i, r in enumerate(rows):
        image_id = r["image_id"]
        true_label = r["dx"]
        try:
            img = load_image_rgb(find_image_path(image_id))
            out = run_pipeline(img, config=run_cfg)

            results.append(out)
            y_true.append(true_label)
            y_pred.append(out["diagnosis"])

        except Exception as e:
            errors.append((image_id, str(e)))

        if (i + 1) % 20 == 0 or i < 5:
            elapsed = time.time() - t0
            speed = (i + 1) / max(elapsed, 1e-9)
            eta = (total - i - 1) / max(speed, 1e-9)
            print(f"[{i+1}/{total}] speed={speed:.2f} img/s ETA={eta:.0f}s errors={len(errors)}")

    metrics = compute_metrics(y_true, y_pred)
    report = print_classification_report(metrics)
    print(report)

    # 保存 metrics
    metrics_json = {
        "accuracy": metrics["accuracy"],
        "per_class": metrics["per_class"],
        "macro_avg": metrics["macro_avg"],
        "weighted_avg": metrics["weighted_avg"],
        "origin_accuracy": metrics.get("origin_accuracy", {}),
        "safety_metrics": metrics.get("safety_metrics", {}),
        "num_samples": len(y_true),
        "num_errors": len(errors),
    }

    metrics_path = os.path.join(config.PATHS["results_dir"], "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    print(f"[save] {metrics_path}")

    report_path = os.path.join(config.PATHS["results_dir"], "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[save] {report_path}")

    if config.VALIDATION.get("save_confusion_matrix", True):
        cm_path = os.path.join(config.PATHS["results_dir"], "confusion_matrix.csv")
        save_confusion_matrix(metrics["confusion_matrix"], cm_path)
        print(f"[save] {cm_path}")

    if config.VALIDATION.get("save_predictions", True):
        pred_path = os.path.join(config.PATHS["results_dir"], "predictions.csv")
        with open(pred_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "true", "pred", "correct", "confidence"])
            for i, r in enumerate(rows[:len(y_true)]):
                w.writerow([
                    r["image_id"], y_true[i], y_pred[i],
                    int(y_true[i] == y_pred[i]),
                    f"{results[i]['confidence']:.4f}",
                ])
        print(f"[save] {pred_path}")

    summary = generate_batch_summary(results, y_true)
    summary_path = os.path.join(config.PATHS["results_dir"], "batch_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[save] {summary_path}")

    if errors:
        err_path = os.path.join(config.PATHS["results_dir"], "errors.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            for image_id, msg in errors:
                f.write(f"{image_id}\t{msg}\n")
        print(f"[save] {err_path}")

    return metrics


def run_validate(num_samples=None, seed=None):
    if num_samples is None:
        num_samples = config.VALIDATION["validate_num_samples"]
    if seed is None:
        seed = int(config.VALIDATION["seed"])

    print(f"[validate] num_samples={num_samples} seed={seed}")
    return run_evaluate(num_samples=num_samples, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="HAM10000 BIP + GLCM/SVM")
    parser.add_argument("--mode", type=str, default="validate", choices=["single", "validate", "evaluate"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(config.PATHS["output_dir"], exist_ok=True)
    os.makedirs(config.PATHS["report_dir"], exist_ok=True)
    os.makedirs(config.PATHS["results_dir"], exist_ok=True)

    if args.mode == "single":
        if args.image is None:
            rows, _ = load_metadata()
            if not rows:
                raise RuntimeError("metadata is empty")
            args.image = rows[0]["image_id"]
        run_single(args.image)

    elif args.mode == "validate":
        run_validate(num_samples=args.num_samples, seed=args.seed)

    else:
        run_evaluate(num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()