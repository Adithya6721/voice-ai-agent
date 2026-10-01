"""
voice_ai.pipeline.utterance — utterance data model.

An Utterance is the unit of work that flows from VAD → ASR.
It carries both the raw audio and (after ASR) the transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Utterance:
    """
    A single detected speech segment with optional transcript.

    Lifecycle:
      1. VAD detects speech → creates Utterance(audio=..., transcript=None)
      2. ASR processes audio → sets transcript field
      3. (future) SLM post-processes → sets final_text field
    """

    audio: np.ndarray                       # Raw 16kHz float32 audio
    transcript: Optional[str] = None        # Raw ASR output
    final_text: Optional[str] = None        # After SLM post-processing
    language: Optional[str] = None          # Detected language code
    confidence: Optional[float] = None      # ASR confidence [0, 1]
    duration_s: float = 0.0                 # Audio duration in seconds
    latency_ms: float = 0.0                 # End-to-end latency

    def __post_init__(self) -> None:
        if self.duration_s == 0.0:
            from voice_ai.audio.audio_utils import SAMPLE_RATE
            self.duration_s = len(self.audio) / SAMPLE_RATE

    @property
    def text(self) -> str:
        """Return the best available text (final > transcript > empty)."""
        return self.final_text or self.transcript or ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def __repr__(self) -> str:
        return (
            f"Utterance("
            f"duration={self.duration_s:.2f}s, "
            f"text={self.text[:50]!r}"
            f")"
        )
