"""
tests/asr/test_asr_base.py — unit tests for ASR base classes.
"""

import numpy as np
import pytest

from voice_ai.asr.base import BaseASR, TranscriptionResult


class ConcreteASR(BaseASR):
    """Minimal ASR implementation for testing."""

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        return TranscriptionResult(
            text="hello world",
            language="en",
            confidence=0.9,
            duration_s=len(audio) / 16000,
        )


class TestTranscriptionResult:
    def test_str_returns_text(self):
        r = TranscriptionResult(text="hello")
        assert str(r) == "hello"

    def test_is_empty_true(self):
        r = TranscriptionResult(text="   ")
        assert r.is_empty()

    def test_is_empty_false(self):
        r = TranscriptionResult(text="hello")
        assert not r.is_empty()

    def test_default_language(self):
        r = TranscriptionResult(text="hi")
        assert r.language == "en"


class TestBaseASR:
    def test_transcribe_returns_result(self):
        asr = ConcreteASR()
        audio = np.zeros(16000, dtype=np.float32)
        result = asr.transcribe(audio)
        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"

    def test_name_returns_class_name(self):
        asr = ConcreteASR()
        assert asr.name == "ConcreteASR"
