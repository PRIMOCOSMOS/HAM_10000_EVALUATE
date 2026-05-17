# HAM10000_EVALUATE Refactored

This refactor removes the legacy multi-model benchmark flow and implements one medically interpretable pipeline aligned with your proposed method:

1. Hair artifact removal: black-hat morphology + inpainting
2. Illumination correction: CLAHE on LAB L-channel
3. Lesion segmentation: Chan-Vese active contour
4. Feature engineering:
   - True MREMD decomposition (fixed-window envelopes, multi-resolution down/up sampling) and BIMF1
   - LBP on ROI and BIMF1 (256 + 256 = 512 texture features)
   - GLCM texture statistics
   - HSV histogram over lesion mask
   - ABCD clinical priors
5. Class imbalance handling: paper-balanced subset / none / SMOTE / SMOTE-ENN
6. Classification: StandardScaler + RBF-SVM or ANN (MLP)

## Core Configs

All required experiment controls are centralized in `ham_pipeline/config.py` and used directly in runtime logic:

1. Preprocessing on/off:
   - `preprocessing_enabled`
   - `enable_hair_removal`
   - `enable_clahe`
2. Feature participation:
   - `include_lbp_roi`, `include_lbp_bimf1`, `include_glcm`, `include_hsv`, `include_abcd`
3. Imbalance strategy:
   - `imbalance_strategy`: `paper_balanced_subset | none | smote | smoteenn`
4. Classifier choice:
   - `model_name`: `svm | ann`
5. Train/validation split:
   - `split_mode`: `paper_fixed_count | stratified_ratio`
   - `val_ratio` (used in `stratified_ratio`)
   - `per_class_limit`, `train_per_class`, `val_per_class` (used in `paper_fixed_count`)

## Structure

```text
HAM_10000_EVALUATE_refactored/
├── main.py
├── requirements.txt
└── ham_pipeline/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── preprocess.py
    ├── segment.py
    ├── features.py
    ├── balancing.py
    ├── model.py
    ├── evaluate.py
    └── trainer.py
```

## Dataset layout

Expected under `--dataset-root`:

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py \
  --dataset-root /path/to/HAM10000 \
  --output-dir artifacts \
  --image-size 160 \
  --split-mode paper_fixed_count \
  --per-class-limit 115 \
  --train-per-class 70 \
  --val-per-class 45 \
  --imbalance-strategy paper_balanced_subset \
  --model svm \
  --svm-c 6.0 \
  --svm-gamma scale \
  --random-state 42 \
  --n-jobs -1
```

Optional feature switches:

- `--disable-glcm`
- `--disable-hsv`
- `--disable-abcd`
- `--disable-lbp-roi`
- `--disable-lbp-bimf1`

Imbalance options:

- `--imbalance-strategy paper_balanced_subset`
- `--imbalance-strategy none`
- `--imbalance-strategy smote`
- `--imbalance-strategy smoteenn`

Model options:

- `--model svm`
- `--model ann --ann-hidden 256,128 --ann-alpha 1e-4 --ann-max-iter 400`

Paper-style replication preset:

- `--paper-mode`

This preset enforces: no hair removal, no CLAHE, only ROI/BIMF1 LBP features, balanced 115/class with 70/45 split, ANN classifier, and no SMOTE.

Ratio split example:

```bash
python main.py \
  --dataset-root /path/to/HAM10000 \
  --output-dir artifacts_ratio \
  --split-mode stratified_ratio \
  --val-ratio 0.2 \
  --imbalance-strategy smoteenn \
  --model svm
```

Balanced-subset coverage over full HAM10000 (round-robin by class subset):

```bash
python main.py \
  --dataset-root /path/to/HAM10000 \
  --output-dir artifacts_coverage \
  --split-mode paper_fixed_count \
  --per-class-limit 115 \
  --train-per-class 70 \
  --val-per-class 45 \
  --imbalance-strategy paper_balanced_subset \
  --model ann \
  --coverage-enabled \
  --coverage-max-rounds 0
```

`coverage-max-rounds=0` means auto rounds until majority-class samples are covered.

Enable posterior probabilities from SVM:

- `--svm-probability`

Note: when `--svm-probability` is enabled, SVC fits an extra Platt scaling stage, which increases training time and memory but allows `predict_proba` during inference.

## Inference

```bash
python predict.py \
  --artifacts-dir artifacts \
  --image-path /path/to/sample.jpg
```

## Outputs

- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/classification_report.csv`
- `artifacts/model.joblib`
- `artifacts/feature_columns.json`
- `artifacts/label_classes.json` (when `--model ann`)
- `artifacts/class_distribution_train.csv`
- `artifacts/class_distribution_val.csv`
- `artifacts/run_config.json`

Coverage mode extra outputs (`coverage_enabled=true`):

- `artifacts/round_*/` per-round metrics and models
- `artifacts/coverage_metrics.csv`
- `artifacts/coverage_summary.json`

## Notes

- There is no widely adopted Python package that exposes the exact Samsudin-style MREMD procedure directly; this repo implements that paper flow explicitly in `ham_pipeline/features.py`.
- BIMF1 extraction follows the MREMD flow with fixed-window extrema envelopes and multiresolution downsampling/upsampling.
- The default split follows the paper-style balanced protocol (70 train + 45 validation per class from up to 115 samples per class).
- `paper_balanced_subset` follows the original paper's class-balancing spirit (equalized per-class subset before training).
- If your accuracy is low on full-dataset settings, try `--paper-mode` first, then add extra feature groups one-by-one to check if they hurt performance.
- This repository is for research and reproducibility, not clinical deployment.