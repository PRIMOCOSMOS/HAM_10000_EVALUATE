from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    dataset_root: Path
    metadata_csv: Path
    image_dirs: tuple[Path, ...]
    output_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: int = 96
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    sampling_strategies: tuple[str, ...] = ("none", "smote", "smoteenn")


def build_paths(dataset_root: str | Path, output_dir: str | Path) -> Paths:
    root = Path(dataset_root).resolve()
    out = Path(output_dir).resolve()
    return Paths(
        dataset_root=root,
        metadata_csv=root / "HAM10000_metadata.csv",
        image_dirs=(
            root / "HAM10000_images_part_1",
            root / "HAM10000_images_part_2",
            root,
        ),
        output_dir=out,
        cache_dir=out / ".cache",
    )
