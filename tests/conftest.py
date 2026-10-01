"""
Shared pytest configuration and fixtures.
"""

import numpy as np
import pytest

from voice_ai.audio.audio_utils import CHUNK_SAMPLES, SAMPLE_RATE


@pytest.fixture
def silent_chunk() -> np.ndarray:
    """A 32ms silent audio chunk at 16 kHz (512 samples — Silero VAD v5+ minimum)."""
    return np.zeros(CHUNK_SAMPLES, dtype=np.float32)


@pytest.fixture
def noise_chunk() -> np.ndarray:
    """A 30ms noise chunk (random samples)."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(CHUNK_SAMPLES).astype(np.float32) * 0.1


@pytest.fixture
def one_second_silence() -> np.ndarray:
    """One second of silence at 16 kHz."""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)
