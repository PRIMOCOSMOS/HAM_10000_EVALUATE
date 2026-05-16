# classification/pipeline.py
import time
import numpy as np

from preprocessing import remove_hair, normalize_color, correct_illumination, segment_lesion
from classification.layer1_origin import classify_origin, ORIGIN_MELANOCYTIC, ORIGIN_VASCULAR, ORIGIN_FIBROUS
from classification.layer2_melanocytic import classify_melanocytic
from classification.layer2_keratinocytic import classify_keratinocytic
from classification.feature_registry import extract_feature_modules, EARLY_MODULES, MEL_EXTRA, KER_EXTRA
from classification.ml_fusion import load_artifact, predict_probs, rule_to_probs, fuse_probs


class _Timer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.cost = {}
        self._t0 = None
        self._name = None

    def start(self, name: str):
        if not self.enabled:
            return
        self._name = name
        self._t0 = time.perf_counter()

    def stop(self):
        if not self.enabled or self._name is None:
            return
        dt = time.perf_counter() - self._t0
        self.cost[self._name] = self.cost.get(self._name, 0.0) + dt
        self._name = None
        self._t0 = None

    def wrap(self, name: str, fn, *args, **kwargs):
        self.start(name)
        out = fn(*args, **kwargs)
        self.stop()
        return out


def run_pipeline(image_rgb: np.ndarray, config: dict = None) -> dict:
    if config is None:
        config = {}

    # 新增：是否记录耗时
    enable_timing = bool(config.get("timing", {}).get("enabled", False))
    timer = _Timer(enable_timing)

    decision_path = []
    feature_switches = config.get("feature_switches", {})
    budget = config.get("compute_budget", {})
    fusion_cfg = config.get("ml_fusion", {})

    img = timer.wrap("preprocess.remove_hair", remove_hair, image_rgb)
    img = timer.wrap("preprocess.normalize_color", normalize_color, img)
    img = timer.wrap("preprocess.correct_illumination", correct_illumination, img)

    seg_result = timer.wrap("preprocess.segment_lesion", segment_lesion, img)
    mask = seg_result["mask"]
    seg_quality = float(seg_result.get("quality_score", 1.0))
    decision_path.append(f"[L0] seg_quality={seg_quality:.2f}")

    # Stage-A early
    timer.start("feature.extract_early")
    all_features, extracted_a = extract_feature_modules(img, mask, EARLY_MODULES, feature_switches)
    timer.stop()

    l1_features = {
        "pigment_network": all_features.get("pigment_network", {}),
        "lacunae": all_features.get("lacunae", {}),
        "central_white_patch": all_features.get("central_white_patch", {}),
    }

    l1_result = timer.wrap("cls.layer1_origin", classify_origin, l1_features, config.get("layer1_thresholds"))
    origin = l1_result["origin"]

    artifact = timer.wrap("ml.load_artifact", load_artifact, fusion_cfg.get("artifact_path", ""))
    early_probs = None
    if fusion_cfg.get("use_early_model", True):
        early_probs = timer.wrap("ml.predict_early", predict_probs, artifact, all_features, "early")

    # Stage-B high-cost selective
    need_high = True
    if budget.get("extract_high_cost_when_uncertain_only", True):
        conf_ref = max(early_probs.values()) if early_probs else l1_result["confidence"]
        need_high = conf_ref < float(budget.get("uncertainty_threshold", 0.62))
    else:
        need_high = False

    extra_modules = []
    if origin == ORIGIN_MELANOCYTIC:
        extra_modules = MEL_EXTRA
    elif origin not in (ORIGIN_VASCULAR, ORIGIN_FIBROUS):
        extra_modules = KER_EXTRA

    if need_high:
        timer.start("feature.extract_high")
        all_features, extracted_b = extract_feature_modules(
            img, mask, extra_modules, feature_switches, existing=all_features
        )
        timer.stop()
    else:
        extracted_b = []

    # Stage-C rules
    l2_result = None
    if origin == ORIGIN_VASCULAR:
        rule_diag, rule_conf = "vasc", l1_result["confidence"]
    elif origin == ORIGIN_FIBROUS:
        rule_diag, rule_conf = "df", l1_result["confidence"]
    elif origin == ORIGIN_MELANOCYTIC:
        mel_features = {
            "pigment_network": all_features.get("pigment_network", {}),
            "symmetry": all_features.get("symmetry", {}),
            "color_variegation": all_features.get("color_variegation", {}),
            "blue_white_veil": all_features.get("blue_white_veil", {}),
            "streaks": all_features.get("streaks", {}),
            "dots_globules": all_features.get("dots_globules", {}),
            "regression": all_features.get("regression", {}),
            "vascular_pattern": all_features.get("vascular_pattern", {}),
            "glcm_texture": all_features.get("glcm_texture", {}),
        }
        l2_result = timer.wrap("cls.layer2_melanocytic", classify_melanocytic, mel_features, config.get("layer2_mel_thresholds"))
        rule_diag, rule_conf = l2_result["diagnosis"], l2_result["confidence"]
    else:
        ker_features = {
            "milia_cysts": all_features.get("milia_cysts", {}),
            "comedo_openings": all_features.get("comedo_openings", {}),
            "fissures_ridges": all_features.get("fissures_ridges", {}),
            "border_sharpness": all_features.get("border_sharpness", {}),
            "strawberry_pattern": all_features.get("strawberry_pattern", {}),
            "surface_texture": all_features.get("surface_texture", {}),
            "vascular_pattern": all_features.get("vascular_pattern", {}),
            "arborizing_vessels": all_features.get("arborizing_vessels", {}),
            "ovoid_nests": all_features.get("ovoid_nests", {}),
            "leaf_spoke_structures": all_features.get("leaf_spoke_structures", {}),
            "blue_white_veil": all_features.get("blue_white_veil", {}),
        }
        l2_result = timer.wrap("cls.layer2_keratinocytic", classify_keratinocytic, ker_features, config.get("layer2_kerat_weights"))
        rule_diag, rule_conf = l2_result["diagnosis"], l2_result["confidence"]

    # Stage-D fusion
    final_probs = None
    if fusion_cfg.get("use_final_model", True):
        final_probs = timer.wrap("ml.predict_final", predict_probs, artifact, all_features, "final")

    rule_probs = timer.wrap("ml.rule_to_probs", rule_to_probs, rule_diag, rule_conf)
    final_diag, final_conf, fused_probs = timer.wrap("ml.fuse_probs", fuse_probs, rule_probs, early_probs, final_probs, fusion_cfg)

    if seg_quality < 0.25:
        final_conf *= 0.78

    result = {
        "diagnosis": final_diag,
        "confidence": float(np.clip(final_conf, 0.0, 1.0)),
        "origin": origin,
        "decision_path": decision_path,
        "all_features": all_features,
        "layer1_result": l1_result,
        "layer2_result": l2_result,
        "class_probs": fused_probs,
    }

    if enable_timing:
        # 总耗时
        total = sum(timer.cost.values())
        timer.cost["pipeline.total"] = total
        result["timing"] = timer.cost

    return result