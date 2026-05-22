from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .balancing import apply_sampling
from .config import PipelineConfig


def _build_svm_pipeline(config: PipelineConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=config.svm_c,
                    gamma=config.svm_gamma,
                    class_weight="balanced",
                    probability=config.svm_probability,
                ),
            ),
        ]
    )


def build_classifier(config: PipelineConfig) -> Pipeline:
    if config.model_name == "ann":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "ann",
                    MLPClassifier(
                        hidden_layer_sizes=config.ann_hidden_layer_sizes,
                        activation="relu",
                        solver="adam",
                        alpha=config.ann_alpha,
                        max_iter=config.ann_max_iter,
                        random_state=config.random_state,
                        early_stopping=True,
                        n_iter_no_change=20,
                    ),
                ),
            ]
        )

    if config.model_name != "svm":
        raise ValueError(f"Unsupported model_name={config.model_name}")

    return _build_svm_pipeline(config)


@dataclass(slots=True)
class TwoStageModel:
    stage1_binary: Pipeline
    stage2_multi: Pipeline
    stage2_classes: list[str]


def fit_two_stage_model(config: PipelineConfig, X_train: np.ndarray, y_train: np.ndarray) -> TwoStageModel:
    if config.model_name != "svm":
        raise ValueError("two_stage_enabled currently supports model='svm' only")

    y_stage1 = np.where(y_train == "nv", "nv", "non_nv")
    X1, y1 = apply_sampling(X_train, y_stage1, _sampling_clone_config(config, config.stage1_imbalance_strategy))
    m1 = _build_svm_pipeline(config)
    m1.fit(X1, y1)

    non_nv_mask = y_train != "nv"
    X2_src = X_train[non_nv_mask]
    y2_src = y_train[non_nv_mask]
    X2, y2 = apply_sampling(X2_src, y2_src, _sampling_clone_config(config, config.stage2_imbalance_strategy))
    m2 = _build_svm_pipeline(config)
    m2.fit(X2, y2)

    return TwoStageModel(
        stage1_binary=m1,
        stage2_multi=m2,
        stage2_classes=[str(c) for c in m2.classes_.tolist()],
    )


def predict_proba_two_stage(model: TwoStageModel, X: np.ndarray, labels: list[str]) -> np.ndarray:
    label_to_idx = {k: i for i, k in enumerate(labels)}

    p1 = model.stage1_binary.predict_proba(X)
    classes1 = [str(c) for c in model.stage1_binary.classes_.tolist()]
    p_nv = p1[:, classes1.index("nv")]
    p_non_nv = p1[:, classes1.index("non_nv")]

    p2 = model.stage2_multi.predict_proba(X)
    classes2 = [str(c) for c in model.stage2_multi.classes_.tolist()]

    probs = np.zeros((X.shape[0], len(labels)), dtype=np.float64)
    if "nv" in label_to_idx:
        probs[:, label_to_idx["nv"]] = p_nv

    for j, cls in enumerate(classes2):
        if cls in label_to_idx:
            probs[:, label_to_idx[cls]] = p_non_nv * p2[:, j]

    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    probs = probs / row_sum
    return probs.astype(np.float32)


def _sampling_clone_config(base: PipelineConfig, strategy: str) -> PipelineConfig:
    cfg = PipelineConfig(
        dataset_root=base.dataset_root,
        output_dir=base.output_dir,
        image_size=base.image_size,
        val_ratio=base.val_ratio,
        random_state=base.random_state,
        n_jobs=base.n_jobs,
    )
    cfg.smote_k_neighbors = base.smote_k_neighbors
    cfg.enn_k_neighbors = base.enn_k_neighbors
    cfg.imbalance_strategy = strategy
    return cfg