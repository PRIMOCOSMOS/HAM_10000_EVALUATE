from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    dataset_root: Path
    output_dir: Path
    image_size: int = 160
    val_ratio: float = 0.2
    random_state: int = 42
    n_jobs: int = -1

    cv_iter: int = 80
    cv_lambda1: float = 1.0
    cv_lambda2: float = 1.0

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    blackhat_kernel_size: int = 17
    inpaint_radius: int = 3
    preprocessing_enabled: bool = True
    enable_hair_removal: bool = True
    enable_clahe: bool = True

    lbp_points: int = 8
    lbp_radius: int = 1
    lbp_method: str = "default"
    lbp_bins: int = 256

    include_lbp_roi: bool = True
    include_lbp_bimf1: bool = True
    include_glcm: bool = True
    include_hsv: bool = True
    include_abcd: bool = True

    mremd_window_size: int = 5
    mremd_smoothing_kernel: int = 3
    mremd_max_bimfs: int = 4
    hsv_bins: tuple[int, int, int] = (24, 8, 8)
    glcm_distances: tuple[int, int] = (1, 2)
    glcm_angles: tuple[float, float, float, float] = (0.0, 0.785398, 1.570796, 2.356194)

    smote_k_neighbors: int = 5
    enn_k_neighbors: int = 3
    # Imbalance strategy: paper_balanced_subset / none / smote / smoteenn
    imbalance_strategy: str = "paper_balanced_subset"

    model_name: str = "svm"  # one of: svm, ann
    svm_c: float = 6.0
    svm_gamma: str = "scale"
    svm_probability: bool = False

    ann_hidden_layer_sizes: tuple[int, int] = (256, 128)
    ann_alpha: float = 1e-4
    ann_max_iter: int = 400

    # Split mode: paper_fixed_count (original paper style) / stratified_ratio.
    split_mode: str = "paper_fixed_count"

    # Paper-style balanced subset options.
    per_class_limit: int | None = 115
    train_per_class: int = 70
    val_per_class: int = 45

    # Balanced-subset coverage mode: iterate class-balanced subsets until majority-class samples are covered.
    coverage_enabled: bool = False
    coverage_max_rounds: int = 0  # 0 means auto (full coverage by subset size)

    def metadata_csv(self) -> Path:
        return self.dataset_root / "HAM10000_metadata.csv"

    def image_roots(self) -> tuple[Path, Path]:
        return (
            self.dataset_root / "HAM10000_images_part_1",
            self.dataset_root / "HAM10000_images_part_2",
        )
