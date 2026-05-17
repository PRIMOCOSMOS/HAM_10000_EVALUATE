from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import PipelineConfig


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
