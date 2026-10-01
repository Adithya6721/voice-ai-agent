"""
voice_ai.asr — Automatic Speech Recognition package.

Supported models:
  - WhisperASR  (faster-whisper)  ← default
  - ParakeetASR (NVIDIA NeMo)     ← stub, benchmarking pending
  - MoonshineASR (UsefulSensors)  ← stub, benchmarking pending

Swap models by passing a different BaseASR to Transcriber.
"""

from voice_ai.asr.base import BaseASR, TranscriptionResult
from voice_ai.asr.factory import DEFAULT_FALLBACK_CHAIN, FallbackASR, create_asr_engine
from voice_ai.asr.moonshine import MoonshineASR
from voice_ai.asr.parakeet import ParakeetASR
from voice_ai.asr.transcriber import Transcriber
from voice_ai.asr.whisper import WhisperASR

__all__ = [
    "BaseASR",
    "TranscriptionResult",
    "WhisperASR",
    "ParakeetASR",
    "MoonshineASR",
    "FallbackASR",
    "create_asr_engine",
    "DEFAULT_FALLBACK_CHAIN",
    "Transcriber",
]

