from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from .config import PipelineConfig


def apply_sampling(X: np.ndarray, y: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.imbalance_strategy in {"none", "paper_balanced_subset"}:
        return X, y

    # ENN and some SciPy-backed operations require numeric labels on newer SciPy.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    smote = SMOTE(k_neighbors=config.smote_k_neighbors, random_state=config.random_state)
    if config.imbalance_strategy == "smote":
        sampler = smote
    elif config.imbalance_strategy == "smoteenn":
        enn = EditedNearestNeighbours(n_neighbors=config.enn_k_neighbors, kind_sel="mode")
        sampler = SMOTEENN(smote=smote, enn=enn, random_state=config.random_state)
    else:
        raise ValueError(f"Unsupported imbalance_strategy={config.imbalance_strategy}")

    X_res, y_res_encoded = sampler.fit_resample(X, y_encoded)
    y_res = label_encoder.inverse_transform(y_res_encoded)
    return X_res, y_res
