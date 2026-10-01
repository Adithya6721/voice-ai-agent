"""
voice_ai.asr.moonshine — UsefulSensors Moonshine ASR (stub for future implementation).

Moonshine is a small, fast, on-device ASR model optimized for English.

  GitHub: https://github.com/usefulsensors/moonshine
  HuggingFace: https://huggingface.co/UsefulSensors/moonshine

It targets very low latency on CPU and embedded devices.

TODO: Implement after benchmarking vs Whisper.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from voice_ai.asr.base import BaseASR, TranscriptionResult


class MoonshineASR(BaseASR):
    """
    UsefulSensors Moonshine ASR — stub implementation.

    Placeholder for Moonshine integration. Will be implemented
    after benchmark evaluation against Whisper base/small.
    """

    def __init__(self, model_name: str = "moonshine/base") -> None:
        logger.warning(
            "MoonshineASR is not yet implemented. "
            "Use WhisperASR for now. "
            "Moonshine will be added after benchmark evaluation."
        )
        self._model_name = model_name

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        raise NotImplementedError(
            "Moonshine ASR is not yet implemented. "
            "See voice_ai/asr/moonshine.py for the implementation stub."
        )

    @property
    def name(self) -> str:
        return f"Moonshine ({self._model_name})"
