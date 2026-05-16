from __future__ import annotations

import argparse

from config import ExperimentConfig, build_paths
from reproducer.experiment import run_reproduction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce IEEE ICICT 2024 DOI:10.1109/ICICT60155.2024.10673538 on HAM10000"
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths(args.dataset_root, args.output_dir)
    config = ExperimentConfig(
        image_size=args.image_size,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    run_reproduction(paths, config)


if __name__ == "__main__":
    main()
