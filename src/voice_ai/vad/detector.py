"""
voice_ai.vad.detector — stateful speech segment detector.

Wraps a BaseVAD model and accumulates audio chunks into
complete speech utterances using a state machine:

    SILENCE → (speech detected) → SPEECH → (silence detected) → SILENCE
                                              ↓
                                      yields speech segment

This is the layer between raw VAD probability and the ASR input.
"""

from __future__ import annotations

from collections.abc import Generator
from enum import Enum, auto
from typing import Optional

import numpy as np
from loguru import logger

from voice_ai.audio.audio_utils import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    ms_to_samples,
    samples_to_seconds,
)
from voice_ai.vad.base import BaseVAD


class _State(Enum):
    SILENCE = auto()
    SPEECH = auto()


class SpeechDetector:
    """
    Stateful speech segment collector using a VAD model.

    Accepts raw audio chunks and yields complete speech segments
    (numpy arrays) that are ready to send to ASR.

    State machine:
        SILENCE: discard audio, watch for speech.
        SPEECH:  accumulate audio, watch for end of speech.

    Usage:
        detector = SpeechDetector(vad=SileroVAD())
        for chunk in recorder.stream():
            for segment in detector.process(chunk):
                transcript = asr.transcribe(segment)
    """

    def __init__(
        self,
        vad: BaseVAD,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 100,
        max_utterance_s: float = 30.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """
        Args:
            vad:                      VAD model instance.
            threshold:                Speech probability threshold.
            min_speech_duration_ms:   Minimum speech to start accumulating.
            min_silence_duration_ms:  Silence gap to end an utterance.
            speech_pad_ms:            Extra audio padding before and after speech.
            max_utterance_s:          Hard cap on utterance duration (seconds).
            sample_rate:              Audio sample rate (should be 16000).
        """
        self.vad = vad
        self.threshold = threshold
        self.sample_rate = sample_rate

        # Convert ms thresholds to chunk counts
        chunk_ms = CHUNK_SAMPLES * 1000 / sample_rate  # 30ms
        self._min_speech_chunks = max(1, int(min_speech_duration_ms / chunk_ms))
        self._min_silence_chunks = max(1, int(min_silence_duration_ms / chunk_ms))
        self._speech_pad_chunks = max(0, int(speech_pad_ms / chunk_ms))
        self._max_utterance_chunks = int(max_utterance_s * sample_rate / CHUNK_SAMPLES)

        # State
        self._state = _State.SILENCE
        self._speech_chunks: int = 0
        self._silence_chunks: int = 0
        self._utterance_buffer: list[np.ndarray] = []
        self._pre_speech_ring: list[np.ndarray] = []  # rolling pre-speech padding

        logger.debug(
            f"SpeechDetector initialized: "
            f"min_speech={self._min_speech_chunks} chunks, "
            f"min_silence={self._min_silence_chunks} chunks, "
            f"pad={self._speech_pad_chunks} chunks"
        )

    def process(self, chunk: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Feed a single audio chunk and yield complete speech segments.

        Args:
            chunk: 480-sample float32 chunk at 16 kHz.

        Yields:
            np.ndarray: Complete speech segment ready for ASR.
        """
        prob = self.vad.speech_probability(chunk)
        is_speech = prob >= self.threshold

        if self._state == _State.SILENCE:
            # Keep a rolling ring buffer for pre-speech padding
            self._pre_speech_ring.append(chunk)
            if len(self._pre_speech_ring) > self._speech_pad_chunks + 1:
                self._pre_speech_ring.pop(0)

            if is_speech:
                self._speech_chunks += 1
                if self._speech_chunks >= self._min_speech_chunks:
                    # Confirmed speech — transition to SPEECH state
                    self._state = _State.SPEECH
                    self._silence_chunks = 0
                    # Start utterance buffer with pre-speech padding
                    self._utterance_buffer = list(self._pre_speech_ring)
                    self._pre_speech_ring = []
                    logger.debug(f"Speech started (prob={prob:.2f})")
            else:
                self._speech_chunks = 0

        elif self._state == _State.SPEECH:
            self._utterance_buffer.append(chunk)

            if is_speech:
                self._silence_chunks = 0
            else:
                self._silence_chunks += 1

            # Hard cap on utterance length
            if len(self._utterance_buffer) >= self._max_utterance_chunks:
                logger.warning("Utterance hit maximum length cap — forcing end.")
                segment = self._flush()
                if segment is not None:
                    yield segment
                return

            # End of speech: enough silence accumulated
            if self._silence_chunks >= self._min_silence_chunks:
                logger.debug(
                    f"Speech ended after {samples_to_seconds(len(self._utterance_buffer) * CHUNK_SAMPLES):.2f}s"
                )
                # Add post-speech padding (already in buffer from silence chunks)
                segment = self._flush()
                if segment is not None:
                    yield segment

    def _flush(self) -> Optional[np.ndarray]:
        """Concatenate buffer into a segment, reset state, and return the segment."""
        if not self._utterance_buffer:
            return None

        segment = np.concatenate(self._utterance_buffer)

        # Reset state
        self._state = _State.SILENCE
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._utterance_buffer = []
        self._pre_speech_ring = []
        self.vad.reset_state()

        duration = samples_to_seconds(len(segment))
        logger.info(f"Speech segment ready: {duration:.2f}s ({len(segment)} samples)")
        return segment

    def reset(self) -> None:
        """Fully reset detector state (e.g. between recording sessions)."""
        self._state = _State.SILENCE
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._utterance_buffer = []
        self._pre_speech_ring = []
        self.vad.reset_state()
