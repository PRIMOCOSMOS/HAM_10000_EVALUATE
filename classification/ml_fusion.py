import os
import numpy as np
from classification.feature_vector import flatten_features_by_names, FEATURE_NAMES

ALL_CLASSES = ["nv", "mel", "bkl", "akiec", "bcc", "vasc", "df"]

try:
    import joblib
except Exception:
    joblib = None


def load_artifact(path: str):
    if not path or not os.path.isfile(path) or joblib is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def predict_probs(model_pack: dict, all_features: dict, stage: str):
    if model_pack is None:
        return None

    model = model_pack.get(f"{stage}_model")
    scaler = model_pack.get(f"{stage}_scaler")
    feat_names = model_pack.get(f"{stage}_feature_names", FEATURE_NAMES)
    class_order = model_pack.get("class_order", ALL_CLASSES)

    if model is None:
        return None

    x = flatten_features_by_names(all_features, feat_names).reshape(1, -1)
    if scaler is not None:
        x = scaler.transform(x)

    p = model.predict_proba(x)[0]
    return {c: float(p[i]) for i, c in enumerate(class_order)}


def rule_to_probs(rule_diag: str, rule_conf: float):
    conf = float(np.clip(rule_conf, 0.0, 1.0))
    rest = (1.0 - conf) / (len(ALL_CLASSES) - 1)
    d = {c: rest for c in ALL_CLASSES}
    d[rule_diag] = conf
    return d


def fuse_probs(rule_probs: dict, early_probs: dict, final_probs: dict, cfg: dict):
    wr = float(cfg.get("rule_weight", 0.35))
    we = float(cfg.get("early_weight", 0.30))
    wf = float(cfg.get("final_weight", 0.35))
    s = wr + we + wf
    wr, we, wf = wr / s, we / s, wf / s

    out = {}
    for c in ALL_CLASSES:
        out[c] = (
            wr * rule_probs.get(c, 0.0) +
            we * (early_probs.get(c, 0.0) if early_probs else 0.0) +
            wf * (final_probs.get(c, 0.0) if final_probs else 0.0)
        )

    z = sum(out.values())
    if z > 0:
        for c in out:
            out[c] /= z

    top = max(out, key=out.get)
    top_p = out[top]

    nv_margin = float(cfg.get("nv_margin", 0.08))
    if top != "nv" and (top_p - out["nv"]) < nv_margin:
        top = "nv"
        top_p = out["nv"]

    return top, float(np.clip(top_p, 0.0, 1.0)), out