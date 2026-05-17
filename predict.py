from __future__ import annotations

import argparse
import json
from pathlib import Path

from ham_pipeline.infer import predict_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a HAM10000 class for one image")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_one(
        artifacts_dir=args.artifacts_dir,
        image_path=args.image_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
