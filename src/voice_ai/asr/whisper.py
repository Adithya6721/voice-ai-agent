"""
voice_ai.asr.whisper — Whisper ASR via faster-whisper.

faster-whisper is a CTranslate2-based reimplementation of OpenAI's Whisper
that is 4× faster than the original with equivalent accuracy.

  GitHub:     https://github.com/SYSTRAN/faster-whisper
  HuggingFace: https://huggingface.co/Systran/faster-whisper-*

Model sizes available:
  tiny      (39M)   — fastest, lowest accuracy
  base      (74M)   — good CPU option
  small     (244M)  — ← default for this project (good balance)
  medium    (769M)  — better accuracy, needs CUDA
  large-v3-turbo (809M) — best distilled, needs CUDA

Compute types:
  CPU:  int8  (fastest on CPU)
  CUDA: float16 (Tensor Core acceleration)

The model is downloaded automatically on first use to:
  ~/.cache/huggingface/hub/  (Linux/Mac)
  C:\\Users\\<user>\\.cache\\huggingface\\hub\\ (Windows)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from loguru import logger

from voice_ai.asr.base import BaseASR, TranscriptionResult
from voice_ai.audio.audio_utils import SAMPLE_RATE, samples_to_seconds
from voice_ai.inference.device import DeviceInfo, DeviceType, whisper_compute_type


class WhisperASR(BaseASR):
    """
    faster-whisper Whisper ASR implementation.

    Auto-selects compute type based on device:
      - CUDA  → float16 (or configurable)
      - CPU   → int8    (or configurable)

    Example:
        from voice_ai.inference import get_device
        device = get_device()  # auto-detects CUDA or CPU

        asr = WhisperASR(device=device, model_size="small")
        result = asr.transcribe(audio_array)
        print(result.text)
    """

    def __init__(
        self,
        device: Optional[DeviceInfo] = None,
        model_size: str = "small",
        language: Optional[str] = "en",
        task: str = "transcribe",
        beam_size: int = 5,
        compute_type: str = "auto",
        download_root: Optional[str] = None,
        no_speech_threshold: float = 0.6,
    ) -> None:
        """
        Args:
            device:        DeviceInfo from get_device(). Defaults to auto-detect.
            model_size:    Whisper model variant: tiny|base|small|medium|large-v3-turbo
            language:      Language code e.g. 'en'. None = auto-detect.
            task:          'transcribe' or 'translate' (translate → English).
            beam_size:     Beam search width (higher = more accurate, slower).
            compute_type:  'auto'|'int8'|'float16'|'float32'
            download_root: Custom model cache directory. None = HF default.
        """
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        # Resolve device
        if device is None:
            from voice_ai.inference.device import get_device
            device = get_device()

        self._device = device
        self._model_size = model_size
        self._language = language
        self._task = task
        self._beam_size = beam_size
        self._no_speech_threshold = no_speech_threshold

        resolved_compute = whisper_compute_type(device, compute_type)
        self._compute_type = resolved_compute

        logger.info(
            f"Loading Whisper {model_size} | "
            f"device={device} | "
            f"compute={resolved_compute} | "
            f"language={language or 'auto'}"
        )

        load_start = time.perf_counter()
        self._model = WhisperModel(
            model_size_or_path=model_size,
            device=device.ct2_device,
            device_index=device.index if device.is_cuda else 0,
            compute_type=resolved_compute,
            download_root=download_root,
        )
        load_ms = (time.perf_counter() - load_start) * 1000
        logger.info(f"Whisper {model_size} loaded in {load_ms:.0f}ms.")

    # ── BaseASR interface ──────────────────────────────

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe a float32 mono audio array at 16 kHz.

        Args:
            audio: 1-D float32 numpy array at 16 kHz.

        Returns:
            TranscriptionResult
        """
        t_start = time.perf_counter()
        duration_s = samples_to_seconds(len(audio))

        # faster-whisper accepts numpy float32 directly
        segments_iter, info = self._model.transcribe(
            audio,
            language=self._language,
            task=self._task,
            beam_size=self._beam_size,
            word_timestamps=False,
            vad_filter=False,   # we have our own VAD
        )

        # Materialize the lazy iterator
        raw_segments = list(segments_iter)

        latency_ms = (time.perf_counter() - t_start) * 1000

        # ── Hallucination filter ───────────────────────────────────────────
        # Whisper reports no_speech_prob per segment. If the average is above
        # our threshold, the audio was almost certainly silence or noise and
        # Whisper hallucinated. Return an empty result.
        if raw_segments:
            avg_no_speech = sum(s.no_speech_prob for s in raw_segments) / len(raw_segments)
            if avg_no_speech > self._no_speech_threshold:
                logger.debug(
                    f"Whisper hallucination suppressed "
                    f"(avg_no_speech_prob={avg_no_speech:.2f} > {self._no_speech_threshold})"
                )
                return TranscriptionResult(
                    text="",
                    language=info.language,
                    duration_s=duration_s,
                    latency_ms=latency_ms,
                    model_name=self.name,
                )

        # Combine all segment texts
        text = " ".join(seg.text.strip() for seg in raw_segments).strip()

        segments_data = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            }
            for seg in raw_segments
        ]

        # Average confidence from log-probs
        confidence: Optional[float] = None
        if raw_segments:
            avg_logprob = sum(s.avg_logprob for s in raw_segments) / len(raw_segments)
            confidence = float(min(1.0, max(0.0, (avg_logprob + 1.0))))

        result = TranscriptionResult(
            text=text,
            language=info.language,
            confidence=confidence,
            duration_s=duration_s,
            latency_ms=latency_ms,
            segments=segments_data,
            model_name=self.name,
        )

        rtf = latency_ms / (duration_s * 1000) if duration_s > 0 else 0.0
        logger.debug(
            f"Whisper: '{text[:80]}' | "
            f"{latency_ms:.0f}ms | RTF={rtf:.2f} | lang={info.language}"
        )

        return result

    @property
    def name(self) -> str:
        return f"Whisper {self._model_size}"

    def __repr__(self) -> str:
        return (
            f"WhisperASR("
            f"model={self._model_size}, "
            f"device={self._device}, "
            f"compute={self._compute_type}, "
            f"lang={self._language}"
            f")"
        )
