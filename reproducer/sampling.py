from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE


def apply_sampling(
    x_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "none":
        return x_train, y_train

    if strategy == "smote":
        sampler = SMOTE(random_state=random_state)
        return sampler.fit_resample(x_train, y_train)

    if strategy == "smoteenn":
        sampler = SMOTEENN(random_state=random_state)
        return sampler.fit_resample(x_train, y_train)

    raise ValueError(f"Unknown sampling strategy: {strategy}")
