"""
tests/audio/test_audio_utils.py — unit tests for audio utilities.
"""

import numpy as np
import pytest

from voice_ai.audio.audio_utils import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    chunk_audio,
    ms_to_samples,
    samples_to_ms,
    to_float32,
    to_mono,
    is_silent,
    calculate_rms,
)


class TestConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000

    def test_chunk_samples(self):
        # 32ms @ 16kHz = 512 samples (minimum for Silero VAD v5+)
        assert CHUNK_SAMPLES == 512


class TestToFloat32:
    def test_passthrough_float32(self):
        audio = np.array([0.5, -0.3], dtype=np.float32)
        result = to_float32(audio)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, audio)

    def test_int16_normalised(self):
        audio = np.array([32767, -32768], dtype=np.int16)
        result = to_float32(audio)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(1.0, abs=1e-4)
        assert result[1] == pytest.approx(-1.0, abs=1e-4)


class TestToMono:
    def test_already_mono(self):
        audio = np.ones(100, dtype=np.float32)
        result = to_mono(audio)
        assert result.shape == (100,)

    def test_stereo_to_mono(self):
        stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
        mono = to_mono(stereo)
        assert mono.shape == (2,)
        np.testing.assert_allclose(mono, [0.0, 0.5])


class TestChunkAudio:
    def test_exact_division(self):
        audio = np.zeros(1024, dtype=np.float32)  # 2 chunks of 512
        chunks = chunk_audio(audio, chunk_size=512)
        assert len(chunks) == 2
        assert all(len(c) == 512 for c in chunks)

    def test_padding_last_chunk(self):
        audio = np.zeros(600, dtype=np.float32)  # 512 + 88 → padded to 512
        chunks = chunk_audio(audio, chunk_size=512)
        assert len(chunks) == 2
        assert len(chunks[1]) == 512


class TestMsConversions:
    def test_ms_to_samples_32ms(self):
        assert ms_to_samples(32) == 512

    def test_samples_to_ms_512(self):
        assert samples_to_ms(512) == pytest.approx(32.0)


class TestSilence:
    def test_silent_array(self):
        silent = np.zeros(480, dtype=np.float32)
        assert is_silent(silent)

    def test_loud_array(self):
        loud = np.ones(480, dtype=np.float32)
        assert not is_silent(loud)
