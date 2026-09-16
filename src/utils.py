"""Shared logging, timing, and formatting helpers."""
from __future__ import annotations

import functools
import logging
import os
import time
from pathlib import Path

# Best-effort: disable Chroma's anonymized telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def timed(fn):
    """Logs how long the wrapped function took."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger = get_logger(fn.__module__)
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{fn.__name__} took {elapsed_ms:.1f}ms")
        return result

    return wrapper


def format_source(doc_metadata: dict, score: float | None = None) -> str:
    """One-line, human-readable summary of a retrieved source."""
    tags = doc_metadata.get("tags", [])
    tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
    score_str = f" (score={score:.3f})" if score is not None else ""
    return f"[{doc_metadata.get('url', 'no-url')}] tags: {tag_str}{score_str}"
