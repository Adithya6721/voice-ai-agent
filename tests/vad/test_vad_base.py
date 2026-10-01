"""
tests/vad/test_vad_base.py — unit tests for the abstract VAD interface.
"""

import numpy as np
import pytest

from voice_ai.vad.base import BaseVAD


class ConcreteVAD(BaseVAD):
    """Minimal implementation for testing the abstract interface."""

    def __init__(self, fixed_prob: float) -> None:
        self._prob = fixed_prob

    def speech_probability(self, chunk: np.ndarray) -> float:
        return self._prob


class TestBaseVAD:
    def test_is_speech_above_threshold(self):
        vad = ConcreteVAD(0.8)
        chunk = np.zeros(480, dtype=np.float32)
        assert vad.is_speech(chunk, threshold=0.5) is True

    def test_is_speech_below_threshold(self):
        vad = ConcreteVAD(0.2)
        chunk = np.zeros(480, dtype=np.float32)
        assert vad.is_speech(chunk, threshold=0.5) is False

    def test_is_speech_at_threshold(self):
        vad = ConcreteVAD(0.5)
        chunk = np.zeros(480, dtype=np.float32)
        assert vad.is_speech(chunk, threshold=0.5) is True

    def test_reset_state_noop(self):
        vad = ConcreteVAD(0.5)
        vad.reset_state()  # should not raise

    def test_name_returns_class_name(self):
        vad = ConcreteVAD(0.5)
        assert vad.name == "ConcreteVAD"
