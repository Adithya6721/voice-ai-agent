"""
voice_ai.vad.silero — Silero VAD implementation.

Silero VAD is a lightweight, real-time capable Voice Activity Detector:
  - Model size:  ~1 MB
  - Backend:     ONNX (CPU) or PyTorch
  - Sample rate: 16 kHz (matches our pipeline)
  - Input:       30ms or 60ms chunks (480 or 960 samples at 16 kHz)
  - Output:      speech probability [0.0, 1.0]

GitHub: https://github.com/snakers4/silero-vad
HuggingFace: https://huggingface.co/snakers4/silero-vad

Design decision:
  VAD always runs on CPU — there is no benefit to putting a 1MB model on GPU.
  This keeps the GPU free for the much larger ASR model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch  # type: ignore[import-untyped]
from loguru import logger

from voice_ai.audio.audio_utils import CHUNK_SAMPLES, SAMPLE_RATE
from voice_ai.vad.base import BaseVAD


class SileroVAD(BaseVAD):
    """
    Silero VAD running on CPU via PyTorch.

    Uses the official silero-vad package (v5+) which wraps the
    ONNX-optimised model with a clean Python API.

    Example:
        vad = SileroVAD(threshold=0.5)
        for chunk in recorder.stream():
            if vad.is_speech(chunk):
                accumulate(chunk)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 100,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """
        Args:
            threshold:                Speech probability threshold [0.0, 1.0].
            min_speech_duration_ms:   Minimum speech segment length (ms).
            min_silence_duration_ms:  Silence gap before ending segment (ms).
            speech_pad_ms:            Padding added around detected speech (ms).
            sample_rate:              Expected input sample rate (must be 16000).
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        self._model: Optional[object] = None
        self._load_model()

    # ── Loading ────────────────────────────────────────

    def _load_model(self) -> None:
        """Load Silero VAD from the silero-vad package (auto-downloads on first run)."""
        logger.info("Loading Silero VAD model…")
        try:
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]
            self._model = load_silero_vad()
            self._model.eval()  # type: ignore[union-attr]
            logger.info("Silero VAD loaded successfully (CPU).")
        except ImportError:
            # Fallback: load via torch.hub (downloads from GitHub)
            logger.warning(
                "silero-vad package not found. Falling back to torch.hub. "
                "Install with: pip install silero-vad"
            )
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model.eval()  # type: ignore[union-attr]
            logger.info("Silero VAD loaded via torch.hub (CPU).")

    # ── BaseVAD interface ──────────────────────────────

    def speech_probability(self, chunk: np.ndarray) -> float:
        """
        Return speech probability for a single 30ms audio chunk.

        Args:
            chunk: 1-D float32 numpy array, 16 kHz, 480 samples.

        Returns:
            float in [0.0, 1.0]
        """
        if len(chunk) != CHUNK_SAMPLES:
            # Silero VAD v5+ requires exactly 512 samples (32ms @ 16kHz).
            # sr / chunk_size must not exceed 31.25: 16000 / 512 = 31.25 ✓
            chunk = np.resize(chunk, CHUNK_SAMPLES)

        tensor = torch.from_numpy(chunk)
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate)  # type: ignore[operator]
        return float(prob)

    def reset_state(self) -> None:
        """Reset Silero's internal LSTM hidden states between utterances."""
        if self._model is not None:
            self._model.reset_states()  # type: ignore[union-attr]

    @property
    def name(self) -> str:
        return "Silero VAD"

    def __repr__(self) -> str:
        return (
            f"SileroVAD("
            f"threshold={self.threshold}, "
            f"min_speech={self.min_speech_duration_ms}ms, "
            f"min_silence={self.min_silence_duration_ms}ms"
            f")"
        )
