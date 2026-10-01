"""
voice_ai.utils.timing — timing utilities.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Generator


@contextmanager
def timer(label: str = "") -> Generator[dict, None, None]:
    """
    Context manager that measures elapsed time in milliseconds.

    Usage:
        with timer("ASR") as t:
            result = asr.transcribe(audio)
        print(f"Took {t['ms']:.0f}ms")
    """
    info: dict = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["ms"] = (time.perf_counter() - start) * 1000
