from __future__ import annotations

import argparse
from pathlib import Path

from ham_pipeline.config import PipelineConfig
from ham_pipeline.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAM10000 traditional ML pipeline")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cv-iter", type=int, default=80)
    parser.add_argument("--cv-lambda1", type=float, default=1.0)
    parser.add_argument("--cv-lambda2", type=float, default=1.0)
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--enn-k-neighbors", type=int, default=3)
    parser.add_argument(
        "--imbalance-strategy",
        type=str,
        default="paper_balanced_subset",
        choices=["paper_balanced_subset", "none", "smote", "smoteenn"],
    )
    parser.add_argument("--disable-glcm", action="store_true")
    parser.add_argument("--disable-hsv", action="store_true")
    parser.add_argument("--disable-abcd", action="store_true")
    parser.add_argument("--disable-lbp-roi", action="store_true")
    parser.add_argument("--disable-lbp-bimf1", action="store_true")
    parser.add_argument("--disable-hair-removal", action="store_true")
    parser.add_argument("--disable-clahe", action="store_true")
    parser.add_argument("--model", type=str, default="svm", choices=["svm", "ann"])
    parser.add_argument("--svm-c", type=float, default=6.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--svm-probability", action="store_true")
    parser.add_argument("--ann-hidden", type=str, default="256,128")
    parser.add_argument("--ann-alpha", type=float, default=1e-4)
    parser.add_argument("--ann-max-iter", type=int, default=400)
    parser.add_argument("--split-mode", type=str, default="paper_fixed_count", choices=["paper_fixed_count", "stratified_ratio"])
    parser.add_argument("--per-class-limit", type=int, default=115)
    parser.add_argument("--train-per-class", type=int, default=70)
    parser.add_argument("--val-per-class", type=int, default=45)
    parser.add_argument("--coverage-enabled", action="store_true")
    parser.add_argument("--coverage-max-rounds", type=int, default=0)
    parser.add_argument("--paper-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ann_hidden = tuple(int(x.strip()) for x in args.ann_hidden.split(",") if x.strip())

    config = PipelineConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        cv_iter=args.cv_iter,
        cv_lambda1=args.cv_lambda1,
        cv_lambda2=args.cv_lambda2,
        smote_k_neighbors=args.smote_k_neighbors,
        enn_k_neighbors=args.enn_k_neighbors,
        imbalance_strategy=args.imbalance_strategy,
        include_lbp_roi=not args.disable_lbp_roi,
        include_lbp_bimf1=not args.disable_lbp_bimf1,
        include_glcm=not args.disable_glcm,
        include_hsv=not args.disable_hsv,
        include_abcd=not args.disable_abcd,
        preprocessing_enabled=not (args.disable_hair_removal and args.disable_clahe),
        enable_hair_removal=not args.disable_hair_removal,
        enable_clahe=not args.disable_clahe,
        model_name=args.model,
        svm_c=args.svm_c,
        svm_gamma=args.svm_gamma,
        svm_probability=args.svm_probability,
        ann_hidden_layer_sizes=ann_hidden,
        ann_alpha=args.ann_alpha,
        ann_max_iter=args.ann_max_iter,
        split_mode=args.split_mode,
        per_class_limit=args.per_class_limit if args.per_class_limit > 0 else None,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        coverage_enabled=args.coverage_enabled,
        coverage_max_rounds=args.coverage_max_rounds,
    )

    if args.paper_mode:
        # Aligns with Samsudin et al. protocol: no preprocessing, ROI/BIMF1 LBP only,
        # balanced 115/class and ANN classifier without SMOTE.
        config.enable_hair_removal = False
        config.enable_clahe = False
        config.include_lbp_roi = True
        config.include_lbp_bimf1 = True
        config.include_glcm = False
        config.include_hsv = False
        config.include_abcd = False
        config.preprocessing_enabled = False
        config.per_class_limit = 115
        config.train_per_class = 70
        config.val_per_class = 45
        config.split_mode = "paper_fixed_count"
        config.imbalance_strategy = "paper_balanced_subset"
        config.model_name = "ann"
        config.coverage_enabled = False
        config.coverage_max_rounds = 0

    run_training(config)


if __name__ == "__main__":
    main()