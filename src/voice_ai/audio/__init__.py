"""
voice_ai.audio — audio capture and preprocessing package.
"""

from voice_ai.audio.audio_utils import (
    CHUNK_SAMPLES,
    DTYPE,
    SAMPLE_RATE,
    chunk_audio,
    ms_to_samples,
    samples_to_ms,
    samples_to_seconds,
    to_float32,
    to_mono,
)
from voice_ai.audio.recorder import MicrophoneRecorder
from voice_ai.audio.resampler import Resampler

__all__ = [
    "SAMPLE_RATE",
    "CHANNELS",
    "DTYPE",
    "CHUNK_SAMPLES",
    "MicrophoneRecorder",
    "Resampler",
    "to_float32",
    "to_mono",
    "chunk_audio",
    "ms_to_samples",
    "samples_to_ms",
    "samples_to_seconds",
]
