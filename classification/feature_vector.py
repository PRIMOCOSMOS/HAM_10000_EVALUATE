import numpy as np

# 固定全量向量名
FEATURE_NAMES = [
    "net_score", "net_regularity", "net_coverage", "net_gabor",
    "lacunae_score", "lacunae_count", "df_score", "df_grad",
    "symmetry_score", "shape_symmetry", "color_symmetry",
    "color_count", "color_entropy", "color_score",
    "bwv_score", "bwv_area_ratio",
    "streak_score", "streak_asymmetry", "streak_count",
    "dots_irregularity", "dots_size_cv", "dots_distribution_entropy", "dots_count",
    "regression_score", "scar_area_ratio", "peppering_score",
    "vascular_poly", "vascular_density", "vascular_dotted", "vascular_linear",
    "milia_score", "milia_count", "comedo_score", "comedo_count",
    "fissure_score", "border_sharpness",
    "strawberry_score", "roughness_score",
    "arborizing_score", "nest_score", "nest_count", "leaf_spoke_score",
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "glcm_asm", "glcm_entropy", "glcm_irregularity",
]

def flatten_features(all_features: dict) -> np.ndarray:
    net = all_features.get("pigment_network", {})
    lac = all_features.get("lacunae", {})
    cwp = all_features.get("central_white_patch", {})
    sym = all_features.get("symmetry", {})
    color = all_features.get("color_variegation", {})
    bwv = all_features.get("blue_white_veil", {})
    streaks = all_features.get("streaks", {})
    dots = all_features.get("dots_globules", {})
    reg = all_features.get("regression", {})
    vasc = all_features.get("vascular_pattern", {})
    milia = all_features.get("milia_cysts", {})
    comedo = all_features.get("comedo_openings", {})
    fiss = all_features.get("fissures_ridges", {})
    border = all_features.get("border_sharpness", {})
    straw = all_features.get("strawberry_pattern", {})
    surf = all_features.get("surface_texture", {})
    arbor = all_features.get("arborizing_vessels", {})
    nests = all_features.get("ovoid_nests", {})
    leaf = all_features.get("leaf_spoke_structures", {})
    glcm = all_features.get("glcm_texture", {})

    vec = np.array([
        float(net.get("network_score", 0.0)),
        float(net.get("regularity_score", 0.0)),
        float(net.get("coverage", 0.0)),
        float(net.get("gabor_strength", 0.0)),
        float(lac.get("lacunae_score", 0.0)),
        float(lac.get("lacunae_count", 0)),
        float(cwp.get("df_pattern_score", 0.0)),
        float(cwp.get("radial_gradient", 0.0)),
        float(sym.get("symmetry_score", 0.0)),
        float(sym.get("shape_symmetry", 0.0)),
        float(sym.get("color_symmetry", 0.0)),
        float(color.get("color_count", 0)),
        float(color.get("color_entropy", 0.0)),
        float(color.get("color_score", 0.0)),
        float(bwv.get("bwv_score", 0.0)),
        float(bwv.get("bwv_area_ratio", 0.0)),
        float(streaks.get("streak_score", 0.0)),
        float(streaks.get("streak_asymmetry", 0.0)),
        float(streaks.get("streak_count", 0)),
        float(dots.get("irregularity_score", 0.0)),
        float(dots.get("size_cv", 0.0)),
        float(dots.get("distribution_entropy", 0.0)),
        float(dots.get("dot_count", 0)),
        float(reg.get("regression_score", 0.0)),
        float(reg.get("scar_area_ratio", 0.0)),
        float(reg.get("peppering_score", 0.0)),
        float(vasc.get("polymorphism_score", 0.0)),
        float(vasc.get("vessel_density", 0.0)),
        float(vasc.get("dotted_ratio", 0.0)),
        float(vasc.get("linear_ratio", 0.0)),
        float(milia.get("milia_score", 0.0)),
        float(milia.get("milia_count", 0)),
        float(comedo.get("comedo_score", 0.0)),
        float(comedo.get("comedo_count", 0)),
        float(fiss.get("cerebriform_score", 0.0)),
        float(border.get("sharpness_score", 0.0)),
        float(straw.get("strawberry_score", 0.0)),
        float(surf.get("roughness_score", 0.0)),
        float(arbor.get("arborizing_score", 0.0)),
        float(nests.get("nest_score", 0.0)),
        float(nests.get("nest_count", 0)),
        float(leaf.get("combined_score", 0.0)),
        float(glcm.get("contrast", 0.0)),
        float(glcm.get("dissimilarity", 0.0)),
        float(glcm.get("homogeneity", 0.0)),
        float(glcm.get("energy", 0.0)),
        float(glcm.get("correlation", 0.0)),
        float(glcm.get("asm", 0.0)),
        float(glcm.get("entropy", 0.0)),
        float(glcm.get("texture_irregularity", 0.0)),
    ], dtype=np.float64)

    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def flatten_features_by_names(all_features: dict, names: list) -> np.ndarray:
    full = flatten_features(all_features)
    idx_map = {n: i for i, n in enumerate(FEATURE_NAMES)}
    out = []
    for n in names:
        i = idx_map.get(n, None)
        out.append(full[i] if i is not None else 0.0)
    return np.array(out, dtype=np.float64)