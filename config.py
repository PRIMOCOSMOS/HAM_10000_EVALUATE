"""
全局配置文件
将训练、验证、推理相关参数统一集中管理。
"""

import os

# ============================================================
# 路径
# ============================================================
DATASET_ROOT = r"D:\BIP_EXERCISE\LABX_PROJECT\HAM10000"

PATHS = {
    "image_dirs": [
        os.path.join(DATASET_ROOT, "HAM10000_images_part_1"),
        os.path.join(DATASET_ROOT, "HAM10000_images_part_2"),
    ],
    "metadata": os.path.join(DATASET_ROOT, "HAM10000_metadata"),  # 支持无 .csv 后缀
    "output_dir": os.path.join(DATASET_ROOT, "output"),
    "report_dir": os.path.join(DATASET_ROOT, "output", "reports"),
    "results_dir": os.path.join(DATASET_ROOT, "output", "results"),
    "artifact_dir": os.path.join(DATASET_ROOT, "artifacts"),
    "svm_artifact": os.path.join(DATASET_ROOT, "artifacts", "glcm_svm_e2e.pkl"),
}

# 兼容旧代码命名
IMAGE_DIRS = PATHS["image_dirs"]
METADATA_PATH = PATHS["metadata"]
OUTPUT_DIR = PATHS["output_dir"]
REPORT_DIR = PATHS["report_dir"]
RESULTS_DIR = PATHS["results_dir"]
ARTIFACT_DIR = PATHS["artifact_dir"]

# ============================================================
# 预处理参数
# ============================================================
PREPROCESSING = {
    "hair_removal_kernel_size": 17,
    "hair_removal_threshold": 10,
    "color_norm_p": 6,
    "color_norm_target_intensity": 128.0,
    "illumination_sigma": 50.0,
    "segmentation_morph_radius": 15,
}

# ============================================================
# 第1层阈值
# ============================================================
LAYER1_THRESHOLDS = {
    "lacunae_score_thresh": 0.48,
    "lacunae_count_thresh": 4,
    "df_pattern_score_thresh": 0.50,
    "df_gradient_thresh": 0.38,
    "network_score_thresh": 0.24,
    "network_coverage_thresh": 0.07,
    "network_score_soft_thresh": 0.14,
    "vascular_evidence_thresh": 0.58,
    "fibrous_evidence_thresh": 0.56,
    "melanocytic_evidence_thresh": 0.38,
    "origin_margin": 0.08,
}

# ============================================================
# 第2层（melanocytic）阈值
# ============================================================
LAYER2_MELANOCYTIC_THRESHOLDS = {
    "symmetry_thresh": 1.25,
    "color_count_thresh": 5,
    "few_color_thresh": 3,
    "bwv_score_thresh": 0.36,
    "bwv_area_ratio_thresh": 0.03,
    "streak_score_thresh": 0.36,
    "streak_asym_thresh": 0.35,
    "dots_irregularity_thresh": 0.45,
    "dots_entropy_thresh": 0.35,
    "regression_score_thresh": 0.34,
    "regression_area_ratio_thresh": 0.04,
    "polymorphism_thresh": 0.50,
    "network_regularity_thresh": 0.52,
    "network_typical_thresh": 0.66,
    "network_coverage_typical_thresh": 0.10,
    "mel_total_thresh": 4,
    "texture_irregularity_thresh": 0.55,
}

# ============================================================
# 第2层（keratinocytic）权重
# ============================================================
LAYER2_KERATINOCYTIC_WEIGHTS = {
    "bkl_milia": 0.25,
    "bkl_comedo": 0.25,
    "bkl_fissures": 0.25,
    "bkl_border": 0.25,
    "akiec_strawberry": 0.35,
    "akiec_roughness": 0.35,
    "akiec_dotted_vessels": 0.30,
    "bcc_arborizing": 0.35,
    "bcc_ovoid_nests": 0.25,
    "bcc_leaf_spoke": 0.20,
    "bcc_ulceration": 0.10,
    "bcc_blue_structures": 0.10,
}

# ============================================================
# SVM 融合参数（推理）
# ============================================================
ML_FUSION = {
    "enabled": True,
    "use_early_model": False,
    "use_final_model": True,
    "rule_weight": 0.0,
    "early_weight": 0.0,
    "final_weight": 1.0,
    "nv_margin": 0.08,
    "artifact_path": PATHS["svm_artifact"],
}

# ============================================================
# 计算预算
# ============================================================
COMPUTE_BUDGET = {
    "profile": "quality",  # fast | balanced | quality
    "extract_high_cost_when_uncertain_only": True,
    "uncertainty_threshold": 0.62,
}

# ============================================================
# 特征模块开关
# enabled: 是否提取
# use_in_svm: 提取后是否参与SVM向量（可用于规则解释但不入模）
# cost: 成本标签，便于后续做profile映射
# ============================================================
FEATURE_SWITCHES = {
    "pigment_network": {"enabled": True, "use_in_svm": True, "cost": "low"},
    "lacunae": {"enabled": True, "use_in_svm": True, "cost": "low"},
    "central_white_patch": {"enabled": True, "use_in_svm": True, "cost": "low"},
    "glcm_texture": {"enabled": True, "use_in_svm": True, "cost": "low"},

    "symmetry": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "color_variegation": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "blue_white_veil": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "vascular_pattern": {"enabled": True, "use_in_svm": True, "cost": "mid"},

    "streaks": {"enabled": True, "use_in_svm": True, "cost": "high"},
    "dots_globules": {"enabled": True, "use_in_svm": True, "cost": "high"},
    "regression": {"enabled": True, "use_in_svm": True, "cost": "high"},

    "milia_cysts": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "comedo_openings": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "fissures_ridges": {"enabled": True, "use_in_svm": True, "cost": "high"},
    "border_sharpness": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "strawberry_pattern": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "surface_texture": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "arborizing_vessels": {"enabled": True, "use_in_svm": True, "cost": "high"},
    "ovoid_nests": {"enabled": True, "use_in_svm": True, "cost": "mid"},
    "leaf_spoke_structures": {"enabled": True, "use_in_svm": True, "cost": "mid"},
}

# ============================================================
# Pipeline 运行配置
# ============================================================
PIPELINE_CONFIG = {
    "layer1_thresholds": LAYER1_THRESHOLDS,
    "layer2_mel_thresholds": LAYER2_MELANOCYTIC_THRESHOLDS,
    "layer2_kerat_weights": LAYER2_KERATINOCYTIC_WEIGHTS,
    "ml_fusion": ML_FUSION,
    "compute_budget": COMPUTE_BUDGET,
    "feature_switches": FEATURE_SWITCHES,
}

# ============================================================
# 训练参数（全集中）
# ============================================================
TRAINING = {
    "seed": 42,
    "max_samples": None,       # None=全量
    "test_size": 0.2,
    "log_every": 10,
    "save_val_predictions": True,

    "timing_enabled": True,
    "timing_topk": 12,

    # 训练阶段提特征时通常关闭融合，避免加载模型开销/循环依赖
    "disable_ml_fusion_during_feature_extract": True,

    # 训练开关
    "train_early_model": False,
    "train_final_model": True,  # 关键：只训练 final SVM

    # 不平衡处理
    "rebalance": {
        "enabled": True,
        "target_percentile": 75,
        "target_min": 80,
        "majority_cap_ratio": 2.0,
    },

    # Early SVM 参数
    "svm_early": {
        "kernel": "rbf",
        "C": 4.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
    },

    # Final SVM 参数（若 train_final_model=False 则不会使用）
    "svm_final": {
        "kernel": "rbf",
        "C": 5.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
    },

    # 并行优化
    "parallel_extract": {
        "enabled": True,
        "max_workers": 8,            # i5建议 4~6 先测
        "submit_chunk": 64,          # 一次提交任务数，防止future堆积
    },

    # 特征缓存配置
    "feature_cache": {
        "enabled": True,
        "resume": True,              # 从checkpoint续跑
        "force_rebuild": False,      # True则忽略旧缓存重提取
        "save_npz": True,            # 提取结束后落盘完整缓存
        "checkpoint_flush_every": 20 # 每N条flush一次
    }
}

# ============================================================
# SVM 输入特征集合（训练/推理统一）
# ============================================================
ML_FEATURE_SETS = {
    "early": [
        "net_score", "net_regularity", "net_coverage", "net_gabor",
        "lacunae_score", "lacunae_count", "df_score", "df_grad",
        "symmetry_score", "shape_symmetry", "color_symmetry",
        "color_count", "color_entropy", "color_score",
        "bwv_score", "bwv_area_ratio",
        "vascular_poly", "vascular_density", "vascular_dotted", "vascular_linear",
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
        "glcm_energy", "glcm_correlation", "glcm_asm", "glcm_entropy", "glcm_irregularity",
    ],
    "final": "ALL",  # 或者手动给列表
}

# ============================================================
# 验证/评估参数（全集中）
# ============================================================
VALIDATION = {
    "seed": 42,
    "validate_num_samples": 200,
    "evaluate_num_samples": None,  # None=全量
    "save_confusion_matrix": True,
    "save_predictions": True,
    "save_individual_reports": False,
}