#!/usr/bin/env python3
"""
Phase-level file cache helpers.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Callable, TypeVar, Optional, Dict, Any

T = TypeVar("T")

_CACHE_ROOT = Path(__file__).resolve().parents[2] / "outputs" / ".phase_cache"


def _resolve_paths(namespace: str, key: str) -> tuple[Path, Path]:
    digest = hashlib.sha1(f"{namespace}:{key}".encode("utf-8")).hexdigest()
    cache_dir = _CACHE_ROOT / namespace
    return cache_dir / f"{digest}.pkl", cache_dir / f"{digest}.meta.json"


def fetch_with_file_cache(
    namespace: str,
    key: str,
    fetch_fn: Callable[[], T],
    ttl_seconds: int,
    metrics: Optional[Dict[str, Any]] = None,
) -> T:
    """Fetch from cache if valid, otherwise compute and write-through."""
    if metrics is not None:
        metrics.clear()
        metrics["hit"] = False

    if ttl_seconds <= 0:
        if metrics is not None:
            metrics["disabled"] = True
        return fetch_fn()

    data_path, meta_path = _resolve_paths(namespace, key)
    now = time.time()

    try:
        if data_path.exists() and meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            created_at = float(meta.get("created_at", 0.0))
            if now - created_at <= ttl_seconds:
                with data_path.open("rb") as f:
                    if metrics is not None:
                        metrics["hit"] = True
                        metrics["age_seconds"] = max(0.0, now - created_at)
                    return pickle.load(f)
    except Exception:
        pass

    value = fetch_fn()

    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_data = data_path.with_suffix(".tmp")
        tmp_meta = meta_path.with_suffix(".tmp")

        with tmp_data.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        with tmp_meta.open("w", encoding="utf-8") as f:
            json.dump({"created_at": now}, f)

        tmp_data.replace(data_path)
        tmp_meta.replace(meta_path)
    except Exception:
        pass

    return value
