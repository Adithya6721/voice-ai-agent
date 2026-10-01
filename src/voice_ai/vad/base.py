"""
voice_ai.vad.base — abstract VAD interface.

All VAD implementations must subclass BaseVAD and implement
the single `is_speech()` method.

This decouples the pipeline from any specific VAD model.
Swapping Silero for another VAD only requires creating a new subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseVAD(ABC):
    """
    Abstract Voice Activity Detector.

    Contract:
      - Accepts a single audio chunk (1-D float32 numpy array at 16 kHz).
      - Returns a float in [0.0, 1.0] — speech probability.
      - Optionally exposes `is_speech()` as a boolean shortcut.
    """

    @abstractmethod
    def speech_probability(self, chunk: np.ndarray) -> float:
        """
        Return the probability [0.0, 1.0] that the chunk contains speech.

        Args:
            chunk: 1-D float32 numpy array, 16 kHz mono, 30ms (480 samples).

        Returns:
            float in [0.0, 1.0]
        """
        ...

    def is_speech(self, chunk: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Convenience method — returns True if speech probability >= threshold.

        Args:
            chunk:     Audio chunk (480 samples, 16 kHz, float32).
            threshold: Decision boundary (default 0.5).

        Returns:
            bool
        """
        return self.speech_probability(chunk) >= threshold

    def reset_state(self) -> None:
        """
        Reset any internal model state between utterances.
        Default implementation is a no-op; stateful models should override.
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self.__class__.__name__
