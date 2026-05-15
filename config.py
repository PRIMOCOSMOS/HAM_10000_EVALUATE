"""
全局配置：数据路径、阈值参数、特征提取参数
所有可调参数集中管理，便于实验调优。
"""

import os

# ============================================================
# 数据路径配置
# ============================================================
DATASET_ROOT = r"D:\BIP_EXERCISE\LABX_PROJECT\HAM10000"

# 图像文件夹（两个part合并搜索）
IMAGE_DIRS = [
    os.path.join(DATASET_ROOT, "HAM10000_images_part_1"),
    os.path.join(DATASET_ROOT, "HAM10000_images_part_2"),
]

# 元数据CSV（无.csv后缀）
METADATA_PATH = os.path.join(DATASET_ROOT, "HAM10000_metadata")

# 输出目录
OUTPUT_DIR = os.path.join(DATASET_ROOT, "output")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

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
# 第1层阈值：组织来源判定（优化后）
# ============================================================
LAYER1_THRESHOLDS = {
    "lacunae_score_thresh": 0.45,       # 略提高，减少误判vasc
    "lacunae_count_thresh": 3,
    "df_pattern_score_thresh": 0.45,    # 略提高，减少误判df
    "df_gradient_thresh": 0.35,
    "network_score_thresh": 0.05,       # 大幅降低！原0.35→0.25，让更多nv通过
    "network_coverage_thresh": 0.05,    # 大幅降低！原0.15→0.08
}

# ============================================================
# 第2层阈值：黑色素细胞组 (nv vs mel)
# ============================================================
LAYER2_MELANOCYTIC_THRESHOLDS = {
    "symmetry_thresh": 1.5,
    "color_count_thresh": 4,
    "bwv_score_thresh": 0.3,
    "streak_score_thresh": 0.3,
    "dots_irregularity_thresh": 0.4,
    "regression_score_thresh": 0.3,
    "polymorphism_thresh": 0.4,
    "network_regularity_thresh": 0.5,
    "mel_total_thresh": 3,
}

# ============================================================
# 第2层权重：角质形成细胞组 (bkl vs akiec vs bcc)
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
# 特征提取参数
# ============================================================
FEATURE_PARAMS = {
    "gabor_frequencies": [0.1, 0.15, 0.2],
    "gabor_orientations": 6,
    "log_scales": [2, 3, 4, 6, 8],
    "frangi_scales": [1, 2, 3, 4],
    "polar_num_angles": 360,
    "polar_num_radii": 50,
}

# ============================================================
# 评估配置
# ============================================================
EVALUATION = {
    "num_samples": None,       # None=全量评估，设为整数可做子集快速验证
    "random_seed": 42,
    "save_individual_reports": False,  # 是否保存每张图的诊断报告
    "save_confusion_matrix": True,
}

# ============================================================
# 汇总pipeline配置字典（传入run_pipeline）
# ============================================================
PIPELINE_CONFIG = {
    "layer1_thresholds": LAYER1_THRESHOLDS,
    "layer2_mel_thresholds": LAYER2_MELANOCYTIC_THRESHOLDS,
    "layer2_kerat_weights": LAYER2_KERATINOCYTIC_WEIGHTS,
}