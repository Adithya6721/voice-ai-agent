"""
voice_ai.asr.base — abstract ASR interface.

All ASR implementations must subclass BaseASR and implement
the `transcribe()` method.

This decouples the pipeline from any specific ASR model.
The transcriber and pipeline never import Whisper/Parakeet directly —
they only talk to BaseASR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TranscriptionResult:
    """
    Structured result from a single ASR transcription call.

    Attributes:
        text:           Cleaned transcript text.
        language:       Detected language code (e.g. 'en').
        confidence:     Average log-probability converted to [0,1] if available.
        duration_s:     Duration of the audio input in seconds.
        latency_ms:     Wall-clock time taken for transcription (milliseconds).
        segments:       Optional word/segment-level timing data.
        model_name:     Which model produced this result.
    """

    text: str
    language: str = "en"
    confidence: Optional[float] = None
    duration_s: float = 0.0
    latency_ms: float = 0.0
    segments: list[dict] = field(default_factory=list)
    model_name: str = ""

    def __str__(self) -> str:
        return self.text

    def is_empty(self) -> bool:
        """Return True if no meaningful text was transcribed."""
        return not self.text.strip()


class BaseASR(ABC):
    """
    Abstract Automatic Speech Recogniser.

    Contract:
      - Accepts a 1-D float32 numpy array at 16 kHz.
      - Returns a TranscriptionResult.
      - Must be thread-safe if the pipeline calls it from a worker thread.
    """

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: 1-D float32 numpy array at 16 kHz.
                   Duration is typically 0.5–30 seconds.

        Returns:
            TranscriptionResult with at minimum a `text` field.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self.__class__.__name__

    def warmup(self, duration_s: float = 1.0) -> None:
        """
        Optional warmup pass — run a dummy transcription to pre-load
        model weights into GPU/CPU cache.

        Default implementation generates silent audio and transcribes it.
        Override if the model has a better warmup mechanism.
        """
        import numpy as np
        from voice_ai.audio.audio_utils import SAMPLE_RATE
        dummy = np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)
        self.transcribe(dummy)
