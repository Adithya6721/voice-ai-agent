"""
voice_ai.audio.recorder — microphone capture via sounddevice.

Captures audio from the system microphone and streams it as
fixed-size float32 numpy chunks at 16 kHz mono.

Design:
  - Uses sounddevice.InputStream for low-latency callback-based capture.
  - All resampling to 16 kHz happens internally via Resampler.
  - Consumers iterate over the recorder with a `for chunk in recorder` loop.
  - Thread-safe via a queue between the sounddevice callback and consumer.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator
from typing import Optional

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
from loguru import logger

from voice_ai.audio.audio_utils import (
    CHUNK_SAMPLES,
    DTYPE,
    SAMPLE_RATE,
    to_float32,
    to_mono,
)
from voice_ai.audio.resampler import Resampler


class MicrophoneRecorder:
    """
    Captures microphone audio and yields 16 kHz float32 chunks.

    Usage:
        recorder = MicrophoneRecorder()
        with recorder:
            for chunk in recorder.stream():
                # chunk: np.ndarray, shape=(480,), dtype=float32
                process(chunk)

    The recording stops when:
      - `recorder.stop()` is called from another thread, or
      - the `with` block exits.
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        target_sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        queue_maxsize: int = 100,
    ) -> None:
        """
        Args:
            device:             sounddevice input device index or name.
                                None = system default microphone.
            target_sample_rate: output sample rate (should always be 16000).
            chunk_samples:      samples per yielded chunk (default 480 = 30ms).
            queue_maxsize:      internal buffer capacity in chunks.
        """
        self._device = device
        self._target_rate = target_sample_rate
        self._chunk_samples = chunk_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self._source_rate: Optional[int] = None
        self._resampler: Optional[Resampler] = None
        self._buffer = np.array([], dtype=np.float32)

    # ── Context manager ────────────────────────────────

    def __enter__(self) -> "MicrophoneRecorder":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ── Public API ─────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream."""
        if self._stream is not None:
            logger.warning("Recorder already started.")
            return

        # Query source sample rate from device
        device_info = sd.query_devices(self._device, "input")
        self._source_rate = int(device_info["default_samplerate"])
        logger.info(
            f"Microphone: '{device_info['name']}' "
            f"at {self._source_rate} Hz → resampling to {self._target_rate} Hz"
        )

        self._resampler = Resampler(
            source_rate=self._source_rate,
            target_rate=self._target_rate,
        )
        self._stop_event.clear()

        self._stream = sd.InputStream(
            samplerate=self._source_rate,
            channels=1,
            dtype=DTYPE,
            device=self._device,
            blocksize=0,            # variable blocksize — callback decides
            callback=self._callback,
            latency="low",
        )
        self._stream.start()
        logger.info("Recording started. Press Ctrl+C to stop.")

    def stop(self) -> None:
        """Stop the microphone stream."""
        self._stop_event.set()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                logger.debug(f"Error stopping stream: {exc}")
            logger.info("Recording stopped.")

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Yield fixed-size float32 audio chunks (16 kHz, mono).

        Each chunk has exactly `chunk_samples` frames (default 480 = 30 ms).
        Blocks until a chunk is available or stop() is called.
        """
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                continue

    @staticmethod
    def list_devices() -> None:
        """Print all available audio input devices to console."""
        print(sd.query_devices())

    # ── Internal ───────────────────────────────────────

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — called from audio thread."""
        if status:
            logger.debug(f"Audio stream status: {status}")
        if self._stop_event.is_set():
            return

        # Convert to mono float32
        audio = to_mono(to_float32(indata.copy()))

        # Resample to target rate (16 kHz)
        if self._resampler is not None:
            audio = self._resampler.resample(audio)

        # Accumulate into buffer and emit fixed-size chunks
        self._buffer = np.concatenate([self._buffer, audio])
        while len(self._buffer) >= self._chunk_samples:
            chunk = self._buffer[: self._chunk_samples].copy()
            self._buffer = self._buffer[self._chunk_samples :]
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                logger.warning("Audio queue full — dropping chunk. Consumer too slow?")
