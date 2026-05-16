from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from joblib import Memory


def create_memory(cache_dir: Path) -> Memory:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Memory(location=str(cache_dir), verbose=0)


def stable_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
