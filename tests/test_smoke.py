"""Minimal smoke tests. Catch import/wiring errors early."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_utils_format_source():
    from src.utils import format_source

    out = format_source({"url": "http://example.com", "tags": ["python", "cuda"]}, score=0.42)
    assert "example.com" in out
    assert "python" in out
    assert "0.420" in out


def test_setup_fallback_sample_shape():
    from setup import load_fallback_sample

    records = load_fallback_sample()
    assert len(records) >= 1
    for r in records:
        assert "question" in r
        assert "answer" in r
