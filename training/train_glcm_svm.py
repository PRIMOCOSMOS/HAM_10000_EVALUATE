"""
训练 GLCM + 手工特征融合 SVM（Early + Final，可配置）
优化点：
1) 并行特征提取（ProcessPoolExecutor）
2) 缓存特征（npz）
3) 断点续跑（jsonl checkpoint）
4) 参数变更命中控制（config签名）
"""

import os
import sys
import csv
import json
import time
import hashlib
from copy import deepcopy
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 注入项目根路径，保证 import config 可用
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from classification.pipeline import run_pipeline  # noqa: E402
from classification.feature_vector import flatten_features, FEATURE_NAMES  # noqa: E402

ALL_CLASSES = ["nv", "mel", "bkl", "akiec", "bcc", "vasc", "df"]


def _find_existing_path(base_path: str) -> str:
    candidates = [base_path, base_path + ".csv"]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"File not found, tried: {candidates}")


def load_metadata() -> tuple:
    md_path = _find_existing_path(config.PATHS["metadata"])
    rows = []
    with open(md_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dx = r["dx"].strip()
            if dx in ALL_CLASSES:
                rows.append({
                    "image_id": r["image_id"].strip(),
                    "lesion_id": r["lesion_id"].strip(),
                    "dx": dx,
                })
    return rows, md_path


def find_image_path(image_id: str) -> str:
    fn = f"{image_id}.jpg"
    for d in config.PATHS["image_dirs"]:
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Image not found: {image_id}")


def read_image_rgb(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        raise IOError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_feature_indices(names):
    idx_map = {n: i for i, n in enumerate(FEATURE_NAMES)}
    idx = [idx_map[n] for n in names if n in idx_map]
    if not idx:
        raise ValueError("No valid feature names found in ML_FEATURE_SETS['early'/'final'].")
    return idx


def rebalance_train_set(X, y, seed=42):
    rb = config.TRAINING["rebalance"]
    if not rb.get("enabled", True):
        return X, y

    rng = np.random.RandomState(seed)
    cnt = Counter(y)
    counts = np.array(sorted(cnt.values()))
    pct = rb.get("target_percentile", 75)
    target = int(np.percentile(counts, pct))
    target = max(target, int(rb.get("target_min", 80)))
    cap_ratio = float(rb.get("majority_cap_ratio", 2.0))

    x_out, y_out = [], []
    for c in ALL_CLASSES:
        idx = np.where(y == c)[0]
        n = len(idx)
        if n == 0:
            continue

        if n > int(cap_ratio * target):
            sel = rng.choice(idx, size=int(cap_ratio * target), replace=False)
        elif n < target:
            sel = rng.choice(idx, size=target, replace=True)
        else:
            sel = idx

        x_out.append(X[sel])
        y_out.extend([c] * len(sel))

    Xb = np.concatenate(x_out, axis=0)
    yb = np.array(y_out)
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]


def nv_safety_metrics(y_true, y_pred):
    idx = np.where(y_true == "nv")[0]
    if len(idx) == 0:
        return {
            "nv_misdiagnosis_rate": 0.0,
            "nv_to_mel_rate": 0.0,
            "nv_to_keratin_rate": 0.0,
            "nv_to_other_rate": 0.0,
        }

    nv_pred = y_pred[idx]
    return {
        "nv_misdiagnosis_rate": float(np.mean(nv_pred != "nv")),
        "nv_to_mel_rate": float(np.mean(nv_pred == "mel")),
        "nv_to_keratin_rate": float(np.mean(np.isin(nv_pred, ["bkl", "akiec", "bcc"]))),
        "nv_to_other_rate": float(np.mean(np.isin(nv_pred, ["vasc", "df"]))),
    }


def print_eval_block(title, y_true, y_pred):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(classification_report(y_true, y_pred, labels=ALL_CLASSES, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CLASSES)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"macro_f1   : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"weighted_f1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"nv_safety  : {nv_safety_metrics(y_true, y_pred)}")


def print_timing_summary(timing_acc, n_done, n_total, t_start):
    topk = int(config.TRAINING.get("timing_topk", 12))
    elapsed = time.perf_counter() - t_start
    speed = n_done / max(elapsed, 1e-9)
    eta = (n_total - n_done) / max(speed, 1e-9)

    print(
        f"\n[timing] processed={n_done}/{n_total} "
        f"elapsed={elapsed/60:.1f}min speed={speed:.3f} img/s ETA={eta/60:.1f}min",
        flush=True
    )

    if n_done == 0:
        return

    avg = {k: v / n_done for k, v in timing_acc.items()}
    items = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:topk]
    print("[timing] top slow modules (sec/img):", flush=True)
    for k, v in items:
        print(f"  - {k:<32} {v:.4f}", flush=True)


def save_val_predictions(path, image_ids, y_true, early_pack=None, final_pack=None):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)

        headers = ["image_id", "true"]
        if early_pack is not None:
            cls_e, _, _ = early_pack
            headers += ["pred_early"] + [f"early_prob_{c}" for c in cls_e]
        if final_pack is not None:
            cls_f, _, _ = final_pack
            headers += ["pred_final"] + [f"final_prob_{c}" for c in cls_f]
        w.writerow(headers)

        n = len(y_true)
        for i in range(n):
            row = [image_ids[i], y_true[i]]
            if early_pack is not None:
                cls_e, p_e, yhat_e = early_pack
                row += [yhat_e[i]] + [float(x) for x in p_e[i]]
            if final_pack is not None:
                cls_f, p_f, yhat_f = final_pack
                row += [yhat_f[i]] + [float(x) for x in p_f[i]]
            w.writerow(row)


# =========================
# 缓存与签名
# =========================
def _json_default(obj):
    try:
        return float(obj)
    except Exception:
        return str(obj)


def build_feature_signature(run_cfg: dict) -> str:
    """
    只对“影响特征提取结果”的配置做签名。
    参数改动后签名变化 -> 自动新缓存，不会误命中旧缓存。
    """
    payload = {
        "preprocessing": config.PREPROCESSING,
        "layer1": run_cfg.get("layer1_thresholds", {}),
        "layer2_mel": run_cfg.get("layer2_mel_thresholds", {}),
        "layer2_kerat": run_cfg.get("layer2_kerat_weights", {}),
        "compute_budget": run_cfg.get("compute_budget", {}),
        "feature_switches": run_cfg.get("feature_switches", {}),
        "feature_names": FEATURE_NAMES,
        "ml_feature_sets": config.ML_FEATURE_SETS,
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def cache_paths(signature: str):
    cache_dir = os.path.join(config.PATHS["artifact_dir"], "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f"features_{signature}.npz")
    ckpt_path = os.path.join(cache_dir, f"features_{signature}.jsonl")
    meta_path = os.path.join(cache_dir, f"features_{signature}_meta.json")
    return npz_path, ckpt_path, meta_path


def load_npz_cache(npz_path: str):
    if not os.path.isfile(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"],
        "groups": data["groups"],
        "image_ids": data["image_ids"],
    }


def save_npz_cache(npz_path: str, X, y, groups, image_ids):
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        groups=groups,
        image_ids=image_ids,
    )


def load_checkpoint_map(ckpt_path: str):
    """
    返回 image_id -> record
    record: {"image_id","lesion_id","dx","vec","timing","error"}
    """
    out = {}
    if not os.path.isfile(ckpt_path):
        return out
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                image_id = obj.get("image_id")
                if image_id:
                    out[image_id] = obj
            except Exception:
                continue
    return out


def append_checkpoint(ckpt_fp, record: dict):
    ckpt_fp.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# 并行 worker
# =========================
def extract_single_record(record: dict, run_cfg: dict):
    image_id = record["image_id"]
    lesion_id = record["lesion_id"]
    dx = record["dx"]

    try:
        path = find_image_path(image_id)
        img = read_image_rgb(path)
        res = run_pipeline(img, config=run_cfg)
        vec = flatten_features(res["all_features"]).tolist()
        timing = res.get("timing", {})
        return {
            "ok": True,
            "image_id": image_id,
            "lesion_id": lesion_id,
            "dx": dx,
            "vec": vec,
            "timing": timing,
        }
    except Exception as e:
        return {
            "ok": False,
            "image_id": image_id,
            "lesion_id": lesion_id,
            "dx": dx,
            "error": str(e),
            "timing": {},
        }


def extract_features_with_cache_and_resume(rows, run_cfg):
    tr_cfg = config.TRAINING
    cache_cfg = tr_cfg.get("feature_cache", {})
    par_cfg = tr_cfg.get("parallel_extract", {})

    use_cache = bool(cache_cfg.get("enabled", True))
    resume = bool(cache_cfg.get("resume", True))
    force_rebuild = bool(cache_cfg.get("force_rebuild", False))
    save_npz = bool(cache_cfg.get("save_npz", True))
    flush_every = int(cache_cfg.get("checkpoint_flush_every", 20))

    signature = build_feature_signature(run_cfg)
    npz_path, ckpt_path, meta_path = cache_paths(signature)

    # 1) 命中完整缓存
    if use_cache and (not force_rebuild):
        cached = load_npz_cache(npz_path)
        if cached is not None:
            print(f"[cache] hit npz: {npz_path}", flush=True)
            return cached["X"], cached["y"], cached["groups"], cached["image_ids"], signature

    # 2) 断点恢复
    done_map = {}
    if use_cache and resume and (not force_rebuild):
        done_map = load_checkpoint_map(ckpt_path)
        if done_map:
            print(f"[cache] resume from checkpoint: {len(done_map)} done", flush=True)

    # 按 metadata 顺序构建 pending
    pending_rows = [r for r in rows if r["image_id"] not in done_map]
    total = len(rows)
    print(f"[extract] total={total}, pending={len(pending_rows)}", flush=True)

    timing_acc = defaultdict(float)
    for rec in done_map.values():
        tmap = rec.get("timing", {})
        for k, v in tmap.items():
            try:
                timing_acc[k] += float(v)
            except Exception:
                pass

    t0 = time.perf_counter()

    # checkpoint 追加写
    ckpt_fp = None
    if use_cache:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        mode = "a" if (resume and (not force_rebuild)) else "w"
        ckpt_fp = open(ckpt_path, mode, encoding="utf-8")

    # 并行/串行提取
    par_enabled = bool(par_cfg.get("enabled", True))
    max_workers = int(par_cfg.get("max_workers", 4))
    submit_chunk = int(par_cfg.get("submit_chunk", 64))

    errors = 0
    processed_new = 0

    def _handle_result(ret):
        nonlocal errors, processed_new
        image_id = ret["image_id"]
        done_map[image_id] = ret
        processed_new += 1

        if not ret.get("ok", False):
            errors += 1
        else:
            for k, v in ret.get("timing", {}).items():
                timing_acc[k] += float(v)

        if ckpt_fp is not None:
            append_checkpoint(ckpt_fp, ret)
            if processed_new % flush_every == 0:
                ckpt_fp.flush()

        n_done = len(done_map)
        log_every = int(config.TRAINING.get("log_every", 10))
        if processed_new % log_every == 0 or processed_new <= 5:
            print_timing_summary(timing_acc, n_done, total, t0)
            print(f"[extract] errors={errors}", flush=True)

    if len(pending_rows) > 0:
        if par_enabled and max_workers > 1:
            print(f"[extract] parallel on, workers={max_workers}", flush=True)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                idx = 0
                n = len(pending_rows)

                while idx < n or futures:
                    while idx < n and len(futures) < submit_chunk:
                        r = pending_rows[idx]
                        futures.append(ex.submit(extract_single_record, r, run_cfg))
                        idx += 1

                    done_futs = []
                    for fut in as_completed(futures):
                        done_futs.append(fut)
                        ret = fut.result()
                        _handle_result(ret)
                        if len(done_futs) >= submit_chunk:
                            break

                    done_set = set(done_futs)
                    futures = [f for f in futures if f not in done_set]
        else:
            print("[extract] parallel off, serial mode", flush=True)
            for r in pending_rows:
                ret = extract_single_record(r, run_cfg)
                _handle_result(ret)

    if ckpt_fp is not None:
        ckpt_fp.flush()
        ckpt_fp.close()

    # 重建按 metadata 顺序的有效样本
    ordered = []
    for r in rows:
        rec = done_map.get(r["image_id"])
        if rec is None:
            continue
        if not rec.get("ok", False):
            continue
        ordered.append(rec)

    if len(ordered) < 50:
        raise RuntimeError(f"Too few usable samples after extraction: {len(ordered)}")

    X = np.array([np.array(rec["vec"], dtype=np.float64) for rec in ordered], dtype=np.float64)
    y = np.array([rec["dx"] for rec in ordered])
    groups = np.array([rec["lesion_id"] for rec in ordered])
    image_ids = np.array([rec["image_id"] for rec in ordered])

    # 保存完整npz缓存
    if use_cache and save_npz:
        save_npz_cache(npz_path, X, y, groups, image_ids)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "signature": signature,
                "total_rows": len(rows),
                "usable_rows": int(len(y)),
                "errors": int(errors),
                "npz_path": npz_path,
                "ckpt_path": ckpt_path,
            }, f, indent=2, ensure_ascii=False)
        print(f"[cache] saved npz: {npz_path}", flush=True)

    print_timing_summary(timing_acc, len(y), len(rows), t0)
    return X, y, groups, image_ids, signature


def train():
    seed = int(config.TRAINING["seed"])
    np.random.seed(seed)

    train_early = bool(config.TRAINING.get("train_early_model", True))
    train_final = bool(config.TRAINING.get("train_final_model", True))
    if not train_early and not train_final:
        raise ValueError("Both train_early_model and train_final_model are False.")

    os.makedirs(config.PATHS["artifact_dir"], exist_ok=True)
    os.makedirs(config.PATHS["results_dir"], exist_ok=True)

    rows, md_path = load_metadata()
    max_samples = config.TRAINING.get("max_samples", None)
    if isinstance(max_samples, int) and max_samples > 0:
        rows = rows[:max_samples]

    print(f"[meta] source={md_path}", flush=True)
    print(f"[meta] usable_records={len(rows)}", flush=True)

    run_cfg = deepcopy(config.PIPELINE_CONFIG)
    run_cfg["timing"] = {"enabled": bool(config.TRAINING.get("timing_enabled", True))}
    if config.TRAINING.get("disable_ml_fusion_during_feature_extract", True):
        run_cfg["ml_fusion"]["enabled"] = False

    # ===== 特征提取（并行 + 缓存 + 断点续跑）=====
    X, y, groups, image_ids, signature = extract_features_with_cache_and_resume(rows, run_cfg)
    print(f"[extract] done X={X.shape}, signature={signature}", flush=True)

    # ===== 以下训练逻辑保持原样 =====
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=float(config.TRAINING["test_size"]),
        random_state=seed
    )
    tr_idx, va_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]
    val_ids = image_ids[va_idx]

    print(f"[split] train={len(y_train)} val={len(y_val)}", flush=True)
    print(f"[split] train_dist={Counter(y_train)}", flush=True)
    print(f"[split] val_dist={Counter(y_val)}", flush=True)

    Xb, yb = rebalance_train_set(X_train, y_train, seed=seed)
    print(f"[rebalance] before={Counter(y_train)}", flush=True)
    print(f"[rebalance] after ={Counter(yb)}", flush=True)

    early_names = list(config.ML_FEATURE_SETS["early"])
    early_idx = get_feature_indices(early_names)

    final_names_cfg = config.ML_FEATURE_SETS["final"]
    if final_names_cfg == "ALL":
        final_names = list(FEATURE_NAMES)
        final_idx = list(range(len(FEATURE_NAMES)))
    else:
        final_names = list(final_names_cfg)
        final_idx = get_feature_indices(final_names)

    artifact = {
        "class_order": None,
        "early_model": None,
        "early_scaler": None,
        "early_feature_names": [],
        "final_model": None,
        "final_scaler": None,
        "final_feature_names": [],
        "train_meta": {},
    }

    early_pack = None
    final_pack = None

    if train_early:
        Xb_e = Xb[:, early_idx]
        Xv_e = X_val[:, early_idx]

        scaler_e = StandardScaler()
        Xb_e_s = scaler_e.fit_transform(Xb_e)
        Xv_e_s = scaler_e.transform(Xv_e)

        svm_early_cfg = config.TRAINING["svm_early"]
        clf_e = SVC(
            kernel=svm_early_cfg["kernel"],
            C=float(svm_early_cfg["C"]),
            gamma=svm_early_cfg["gamma"],
            class_weight=svm_early_cfg["class_weight"],
            probability=bool(svm_early_cfg["probability"]),
            random_state=seed,
        )

        print("[train] fitting early SVM...", flush=True)
        clf_e.fit(Xb_e_s, yb)

        p_e = clf_e.predict_proba(Xv_e_s)
        cls_e = list(clf_e.classes_)
        yhat_e = np.array([cls_e[np.argmax(p)] for p in p_e])

        print_eval_block("[EARLY SVM] Validation", y_val, yhat_e)
        early_pack = (cls_e, p_e, yhat_e)

        artifact["class_order"] = cls_e
        artifact["early_model"] = clf_e
        artifact["early_scaler"] = scaler_e
        artifact["early_feature_names"] = early_names

    if train_final:
        Xb_f = Xb[:, final_idx]
        Xv_f = X_val[:, final_idx]

        scaler_f = StandardScaler()
        Xb_f_s = scaler_f.fit_transform(Xb_f)
        Xv_f_s = scaler_f.transform(Xv_f)

        svm_final_cfg = config.TRAINING["svm_final"]
        clf_f = SVC(
            kernel=svm_final_cfg["kernel"],
            C=float(svm_final_cfg["C"]),
            gamma=svm_final_cfg["gamma"],
            class_weight=svm_final_cfg["class_weight"],
            probability=bool(svm_final_cfg["probability"]),
            random_state=seed,
        )

        print("[train] fitting final SVM...", flush=True)
        clf_f.fit(Xb_f_s, yb)

        p_f = clf_f.predict_proba(Xv_f_s)
        cls_f = list(clf_f.classes_)
        yhat_f = np.array([cls_f[np.argmax(p)] for p in p_f])

        print_eval_block("[FINAL SVM] Validation", y_val, yhat_f)
        final_pack = (cls_f, p_f, yhat_f)

        artifact["class_order"] = cls_f
        artifact["final_model"] = clf_f
        artifact["final_scaler"] = scaler_f
        artifact["final_feature_names"] = final_names

    artifact["train_meta"] = {
        "seed": seed,
        "metadata_source": md_path,
        "num_requested": len(rows),
        "num_usable": int(len(y)),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "train_distribution": dict(Counter(y_train)),
        "val_distribution": dict(Counter(y_val)),
        "rebalance_distribution": dict(Counter(yb)),
        "train_early_model": train_early,
        "train_final_model": train_final,
        "feature_signature": signature,
    }

    if artifact["class_order"] is None:
        artifact["class_order"] = ALL_CLASSES

    pkl_path = config.PATHS["svm_artifact"]
    joblib.dump(artifact, pkl_path)
    print(f"[save] artifact={pkl_path}", flush=True)

    meta_json = os.path.join(config.PATHS["artifact_dir"], "glcm_svm_e2e_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_order": artifact["class_order"],
                "early_feature_names": artifact["early_feature_names"],
                "final_feature_names": artifact["final_feature_names"],
                "train_meta": artifact["train_meta"],
            },
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"[save] meta={meta_json}", flush=True)

    if config.TRAINING.get("save_val_predictions", True):
        pred_csv = os.path.join(config.PATHS["results_dir"], "svm_val_predictions.csv")
        save_val_predictions(pred_csv, val_ids, y_val, early_pack=early_pack, final_pack=final_pack)
        print(f"[save] val_predictions={pred_csv}", flush=True)

    print("[done] training completed.", flush=True)


if __name__ == "__main__":
    # Windows 多进程安全入口
    train()