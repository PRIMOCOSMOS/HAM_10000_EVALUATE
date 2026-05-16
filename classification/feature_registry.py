from typing import Dict, Tuple, List

from features import (
    compute_pigment_network, compute_lacunae, compute_central_white_patch,
    compute_symmetry, compute_color_variegation, compute_blue_white_veil,
    compute_streaks, compute_dots_globules, compute_regression, compute_vascular_pattern,
    compute_milia_cysts, compute_comedo_openings, compute_fissures_ridges,
    compute_border_sharpness, compute_strawberry_pattern, compute_surface_texture,
    compute_arborizing_vessels, compute_ovoid_nests, compute_leaf_spoke_structures,
    compute_glcm_texture
)

EXTRACTORS = {
    "pigment_network": compute_pigment_network,
    "lacunae": compute_lacunae,
    "central_white_patch": compute_central_white_patch,
    "glcm_texture": compute_glcm_texture,
    "symmetry": compute_symmetry,
    "color_variegation": compute_color_variegation,
    "blue_white_veil": compute_blue_white_veil,
    "vascular_pattern": compute_vascular_pattern,
    "streaks": compute_streaks,
    "dots_globules": compute_dots_globules,
    "regression": compute_regression,
    "milia_cysts": compute_milia_cysts,
    "comedo_openings": compute_comedo_openings,
    "fissures_ridges": compute_fissures_ridges,
    "border_sharpness": compute_border_sharpness,
    "strawberry_pattern": compute_strawberry_pattern,
    "surface_texture": compute_surface_texture,
    "arborizing_vessels": compute_arborizing_vessels,
    "ovoid_nests": compute_ovoid_nests,
    "leaf_spoke_structures": compute_leaf_spoke_structures,
}

EARLY_MODULES = [
    "pigment_network", "lacunae", "central_white_patch", "glcm_texture",
    "symmetry", "color_variegation", "blue_white_veil", "vascular_pattern"
]

MEL_EXTRA = ["streaks", "dots_globules", "regression"]
KER_EXTRA = [
    "milia_cysts", "comedo_openings", "fissures_ridges", "border_sharpness",
    "strawberry_pattern", "surface_texture", "arborizing_vessels", "ovoid_nests", "leaf_spoke_structures"
]


def extract_feature_modules(
    image_rgb,
    mask,
    modules: List[str],
    feature_switches: Dict,
    existing: Dict = None,
) -> Tuple[Dict, List[str]]:
    if existing is None:
        existing = {}

    out = dict(existing)
    extracted = []

    for name in modules:
        if name in out:
            continue
        sw = feature_switches.get(name, {"enabled": True})
        if not sw.get("enabled", True):
            continue
        fn = EXTRACTORS.get(name)
        if fn is None:
            continue
        out[name] = fn(image_rgb, mask)
        extracted.append(name)

    return out, extracted