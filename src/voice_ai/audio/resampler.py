"""
voice_ai.audio.resampler — audio resampling utilities.

Converts any incoming sample rate to the project-standard 16 kHz
required by both Silero VAD and Whisper.

Uses scipy.signal.resample_poly for high-quality resampling with
minimal latency overhead.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from voice_ai.audio.audio_utils import SAMPLE_RATE


class Resampler:
    """
    Resample audio from any source sample rate to the target rate.

    The resampler pre-computes the GCD-reduced up/down factors at
    construction time, so the per-chunk cost is only the resample itself.

    Example:
        resampler = Resampler(source_rate=44100, target_rate=16000)
        audio_16k = resampler.resample(audio_44k)
    """

    def __init__(
        self,
        source_rate: int,
        target_rate: int = SAMPLE_RATE,
    ) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate

        if source_rate == target_rate:
            self._passthrough = True
            logger.debug(
                f"Resampler: source == target ({source_rate} Hz), passthrough mode."
            )
            return

        self._passthrough = False
        from math import gcd
        common = gcd(source_rate, target_rate)
        self._up = target_rate // common
        self._down = source_rate // common

        logger.debug(
            f"Resampler: {source_rate} Hz → {target_rate} Hz "
            f"(up={self._up}, down={self._down})"
        )

    def resample(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample a float32 mono audio array.

        Args:
            audio: 1-D float32 numpy array at source_rate.

        Returns:
            1-D float32 numpy array at target_rate.
        """
        if self._passthrough:
            return audio
        resampled = resample_poly(audio, self._up, self._down)
        return resampled.astype(np.float32)

    @property
    def ratio(self) -> float:
        """Ratio of target / source sample rates."""
        return self.target_rate / self.source_rate
