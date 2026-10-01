"""
voice_ai.asr.transcriber — high-level ASR transcription orchestrator.

The Transcriber wraps a BaseASR model and provides:
  - Warmup on startup
  - Metrics collection (latency, RTF, WER if reference given)
  - Model swapping without restarting the pipeline

This is the object the pipeline talks to. It never imports
WhisperASR or ParakeetASR directly.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from loguru import logger

from voice_ai.asr.base import BaseASR, TranscriptionResult
from voice_ai.audio.audio_utils import samples_to_seconds


class Transcriber:
    """
    High-level transcription orchestrator wrapping any BaseASR model.

    Usage:
        asr = WhisperASR(device=device, model_size="small")
        transcriber = Transcriber(asr)
        transcriber.warmup()

        result = transcriber.transcribe(audio_array)
        print(result.text)

    The transcriber can swap models at runtime:
        transcriber.model = ParakeetASR(device=device)
    """

    def __init__(
        self,
        model: BaseASR,
        warmup: bool = True,
    ) -> None:
        """
        Args:
            model:  Any BaseASR implementation.
            warmup: If True, run a warmup pass immediately after init.
        """
        self.model = model
        self._total_transcriptions: int = 0
        self._total_latency_ms: float = 0.0
        self._total_audio_s: float = 0.0

        if warmup:
            logger.info(f"Warming up {model.name}…")
            model.warmup()
            logger.info(f"{model.name} warmup complete.")

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio and update internal metrics.

        Args:
            audio: 1-D float32 numpy array at 16 kHz.

        Returns:
            TranscriptionResult
        """
        duration_s = samples_to_seconds(len(audio))
        result = self.model.transcribe(audio)

        # Track metrics
        self._total_transcriptions += 1
        self._total_latency_ms += result.latency_ms
        self._total_audio_s += duration_s

        return result

    def swap_model(self, new_model: BaseASR, warmup: bool = True) -> None:
        """
        Hot-swap the underlying ASR model without restarting the pipeline.

        Args:
            new_model: New BaseASR implementation to use.
            warmup:    Run warmup on the new model before switching.
        """
        logger.info(f"Swapping ASR model: {self.model.name} → {new_model.name}")
        if warmup:
            new_model.warmup()
        self.model = new_model
        logger.info(f"ASR model swapped to: {new_model.name}")

    @property
    def stats(self) -> dict:
        """Return running transcription statistics."""
        avg_latency = (
            self._total_latency_ms / self._total_transcriptions
            if self._total_transcriptions > 0
            else 0.0
        )
        avg_rtf = (
            (self._total_latency_ms / 1000) / self._total_audio_s
            if self._total_audio_s > 0
            else 0.0
        )
        return {
            "model": self.model.name,
            "total_transcriptions": self._total_transcriptions,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_rtf": round(avg_rtf, 3),
            "total_audio_s": round(self._total_audio_s, 1),
        }
