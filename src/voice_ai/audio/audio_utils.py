"""
voice_ai.audio.audio_utils — shared audio helper functions.

All audio in this project is:
  - Sample rate : 16,000 Hz
  - Channels    : 1 (mono)
  - Dtype       : float32  (values in [-1.0, 1.0])

These constants and helpers are the single source of truth for
audio format throughout the pipeline.
"""

from __future__ import annotations

import numpy as np

# ── Project-wide audio constants ──────────────────────
SAMPLE_RATE: int = 16_000       # Hz
CHANNELS: int = 1               # mono
DTYPE: str = "float32"          # PCM float32
CHUNK_MS: int = 32              # VAD chunk size in milliseconds
# Silero VAD v5+ requires sr / chunk_size <= 31.25
# 16000 / 512 = 31.25 exactly — minimum safe size
CHUNK_SAMPLES: int = 512        # = 32ms @ 16kHz


def ms_to_samples(ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Convert milliseconds to sample count."""
    return int(sample_rate * ms / 1000)


def samples_to_ms(samples: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Convert sample count to milliseconds."""
    return samples * 1000 / sample_rate


def samples_to_seconds(samples: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Convert sample count to seconds."""
    return samples / sample_rate


def to_float32(audio: np.ndarray) -> np.ndarray:
    """
    Normalise any integer PCM array to float32 in [-1.0, 1.0].
    If already float32 the input is returned unchanged.
    """
    if audio.dtype == np.float32:
        return audio
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        return audio.astype(np.float32) / max_val
    return audio.astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Collapse a multi-channel array to mono by averaging channels.
    Expects shape (samples,) or (samples, channels).
    Returns shape (samples,).
    """
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def chunk_audio(audio: np.ndarray, chunk_size: int = CHUNK_SAMPLES) -> list[np.ndarray]:
    """
    Split a 1-D audio array into fixed-size chunks.
    The last chunk is zero-padded if shorter than chunk_size.
    """
    chunks = []
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    return chunks


def calculate_rms(audio: np.ndarray) -> float:
    """Return the Root Mean Square energy of an audio chunk."""
    return float(np.sqrt(np.mean(audio**2)))


def is_silent(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True if the audio chunk is effectively silent."""
    return calculate_rms(audio) < threshold
