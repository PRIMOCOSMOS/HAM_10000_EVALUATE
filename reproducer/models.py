from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_models(random_state: int, n_jobs: int) -> dict[str, Pipeline]:
    rf = Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            )
        ]
    )

    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance", p=2)),
        ]
    )

    svm_poly = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="poly",
                    degree=3,
                    C=6.0,
                    gamma="scale",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "random_forest": rf,
        "knn": knn,
        "svm_poly": svm_poly,
    }
