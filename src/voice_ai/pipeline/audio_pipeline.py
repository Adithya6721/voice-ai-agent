"""
voice_ai.pipeline.audio_pipeline — the main pipeline orchestrator.

Wires together:
  MicrophoneRecorder → SpeechDetector (VAD) → Transcriber (ASR)

This is the first functional slice:
  - Microphone captures 16 kHz audio in 30ms chunks
  - VAD accumulates chunks, yields complete speech segments
  - ASR transcribes each segment → prints raw transcript

Architecture:
  - Main thread: runs recorder.stream() loop
  - ASR runs synchronously in a worker thread (blocking call)
  - Events emitted to registered callbacks (for future WebSocket/IPC)

Usage:
    pipeline = AudioPipeline.from_config("config/config.yaml")
    pipeline.run()  # blocks until Ctrl+C
"""

from __future__ import annotations

import queue
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from voice_ai.asr import BaseASR, Transcriber, create_asr_engine
from voice_ai.audio.recorder import MicrophoneRecorder
from voice_ai.audio.enhancer import AudioEnhancer
from voice_ai.inference.device import DeviceInfo, get_device
from voice_ai.pipeline.events import (
    ErrorEvent,
    EventType,
    PipelineEvent,
    SLMEvent,
    TranscriptionEvent,
)
from voice_ai.slm import BaseSLM, LocalSLM
from voice_ai.pipeline.utterance import Utterance

from voice_ai.vad.detector import SpeechDetector
from voice_ai.vad.silero import SileroVAD


EventCallback = Callable[[PipelineEvent], None]


class AudioPipeline:
    """
    End-to-end audio pipeline: Microphone → VAD → ASR → Transcript.

    Architecture:
      - Main thread: Drains microphone chunks -> runs VAD (non-blocking) -> puts speech segments into queue
      - Background worker thread: Dequeues speech segments -> runs ASR model (asynchronous)
    """

    def __init__(
        self,
        device: Optional[DeviceInfo] = None,
        # Audio
        mic_device: Optional[int | str] = None,
        # VAD
        vad_threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 400,
        speech_pad_ms: int = 100,
        max_utterance_s: float = 30.0,
        # ASR
        asr_model: Optional[BaseASR] = None,
        asr_engine: str = "fallback",
        whisper_model: str = "small",
        fallback_chain: Optional[list[dict]] = None,
        language: Optional[str] = "en",
        beam_size: int = 5,
        compute_type: str = "auto",
        asr_warmup: bool = True,
        # Enhancement
        enable_enhancement: bool = True,
        # SLM
        enable_slm: bool = True,
        slm_model: Optional[BaseSLM] = None,
        slm_model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    ) -> None:
        # Resolve device (auto-detect CUDA → CPU)
        self.device = device or get_device()

        # Build components
        logger.info("Initializing audio pipeline…")

        self._recorder = MicrophoneRecorder(device=mic_device)

        vad = SileroVAD(
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._detector = SpeechDetector(
            vad=vad,
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            max_utterance_s=max_utterance_s,
        )

        self._enhancer = AudioEnhancer() if enable_enhancement else None

        if asr_model is None:
            asr_model = create_asr_engine(
                engine_type=asr_engine,
                model_size=whisper_model,
                fallback_chain=fallback_chain,
                device=self.device,
                language=language,
                beam_size=beam_size,
                compute_type=compute_type,
            )

        self._transcriber = Transcriber(model=asr_model, warmup=asr_warmup)

        # SLM Post-processor
        self._slm: Optional[BaseSLM] = None
        if enable_slm:
            if slm_model is not None:
                self._slm = slm_model
            else:
                try:
                    self._slm = LocalSLM(model_id=slm_model_id, device=self.device)
                except Exception as exc:
                    logger.warning(f"Could not load Local SLM ({exc}). Continuing without SLM.")
                    self._slm = None

        self._callbacks: dict[EventType, list[EventCallback]] = {}
        self._running = False
        self._asr_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        logger.info("Pipeline ready.")


    # ── Factory ────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "AudioPipeline":
        """Load pipeline configuration from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found at {config_path}, using defaults.")
            return cls()

        with open(path) as f:
            cfg = yaml.safe_load(f)

        inf = cfg.get("inference", {})
        audio_cfg = cfg.get("audio", {})
        vad_cfg = cfg.get("vad", {})
        asr_cfg = cfg.get("asr", {})
        whisper = asr_cfg.get("whisper", {})
        pipe_cfg = cfg.get("pipeline", {})

        device = get_device(
            preference=inf.get("device", "auto"),
            cuda_index=inf.get("cuda_device_index", 0),
        )

        asr_engine = asr_cfg.get("engine", asr_cfg.get("model", "fallback"))
        fallback_chain = asr_cfg.get("fallback_chain", None)

        slm_cfg = cfg.get("slm", {})
        enable_slm = slm_cfg.get("enabled", True)
        slm_model_id = slm_cfg.get("model_id", "HuggingFaceTB/SmolLM2-360M-Instruct")

        return cls(
            device=device,
            vad_threshold=vad_cfg.get("threshold", 0.5),
            min_speech_ms=vad_cfg.get("min_speech_duration_ms", 250),
            min_silence_ms=vad_cfg.get("min_silence_duration_ms", 400),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
            max_utterance_s=pipe_cfg.get("silence_timeout_s", 30.0),
            asr_engine=asr_engine,
            fallback_chain=fallback_chain,
            whisper_model=whisper.get("model_size", "small"),
            language=whisper.get("language", "en"),
            beam_size=whisper.get("beam_size", 5),
            compute_type=whisper.get("compute_type", "auto"),
            enable_slm=enable_slm,
            slm_model_id=slm_model_id,
            enable_enhancement=audio_cfg.get("enhancement", {}).get("enabled", True),
        )



    # ── Public API ─────────────────────────────────────

    def on_event(self, event_type: EventType, callback: EventCallback) -> None:
        """Register a callback for a specific event type."""
        self._callbacks.setdefault(event_type, []).append(callback)

    def run(self) -> None:
        """
        Start the pipeline. Blocks until Ctrl+C or stop() is called.

        Flow:
          - Main thread: recorder.stream() -> VAD -> asr_queue.put(segment)
          - Background worker thread: asr_queue.get() -> transcriber -> emit event
        """
        self._running = True

        # Graceful Ctrl+C shutdown
        def _handle_sigint(*_: object) -> None:
            logger.info("Interrupted — stopping pipeline.")
            self.stop()

        signal.signal(signal.SIGINT, _handle_sigint)

        # Start ASR worker thread
        self._worker_thread = threading.Thread(target=self._asr_worker_loop, daemon=True)
        self._worker_thread.start()

        logger.info(
            f"\n{'─'*60}\n"
            f"  Pipeline running | Device: {self.device}\n"
            f"  Speak into your microphone. Press Ctrl+C to stop.\n"
            f"{'─'*60}"
        )

        try:
            with self._recorder:
                for chunk in self._recorder.stream():
                    if not self._running:
                        break
                    if self._enhancer is not None:
                        chunk = self._enhancer.process(chunk)
                    for segment in self._detector.process(chunk):
                        self._asr_queue.put(segment)
        except Exception as exc:
            logger.error(f"Pipeline error: {exc}")
            self._emit(ErrorEvent(
                timestamp_s=time.time(),
                message=str(exc),
                exception=exc,
            ))
        finally:
            self.stop()
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
            stats = self._transcriber.stats
            logger.info(f"\nSession stats: {stats}")

    def stop(self) -> None:
        """Signal the pipeline to stop."""
        self._running = False
        self._recorder.stop()

    def _asr_worker_loop(self) -> None:
        """Background worker thread processing speech segments asynchronously."""
        while self._running or not self._asr_queue.empty():
            try:
                segment = self._asr_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._handle_segment(segment)
            except Exception as exc:
                logger.error(f"ASR worker error: {exc}")
            finally:
                self._asr_queue.task_done()


    # ── Internal ───────────────────────────────────────

    def _handle_segment(self, audio_segment: "import numpy as np; np.ndarray") -> None:  # type: ignore
        """Transcribe a speech segment and emit results."""
        import numpy as np
        try:
            result = self._transcriber.transcribe(audio_segment)

            if result.is_empty():
                logger.debug("Empty transcription — skipping.")
                return

            rtf = result.latency_ms / (result.duration_s * 1000) if result.duration_s > 0 else 0.0

            event = TranscriptionEvent(
                timestamp_s=time.time(),
                text=result.text,
                language=result.language,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                rtf=rtf,
            )
            self._emit(event)

            # Always print raw ASR output (preserves existing behavior)
            print(f"\n📝 Raw transcript: {result.text}")
            logger.debug(
                f"  lang={result.language} | "
                f"latency={result.latency_ms:.0f}ms | "
                f"RTF={rtf:.2f} | "
                f"duration={result.duration_s:.2f}s"
            )

            # SLM Post-processing layer
            if self._slm is not None:
                try:
                    slm_res = self._slm.process(result.text)
                    slm_event = SLMEvent(
                        timestamp_s=time.time(),
                        raw_transcript=result.text,
                        clean_text=slm_res.clean_text,
                        intent=slm_res.intent,
                        intent_data=slm_res.intent_data,
                        format=slm_res.format,
                        format_data=slm_res.format_data,
                        final_output=slm_res.final_output,
                        latency_ms=slm_res.latency_ms,
                    )
                    self._emit(slm_event)
                    print(f"✨ Cleaned text:   {slm_res.clean_text}")
                    if slm_res.intent != "none":
                        print(f"🎯 Intent:         {slm_res.intent} {slm_res.intent_data}")
                    if slm_res.format != "none":
                        print(f"📋 Format:         {slm_res.format}")
                    print(f"🚀 Final output:\n{slm_res.final_output}\n")
                except Exception as slm_exc:
                    logger.error(f"SLM post-processing error: {slm_exc}")

        except Exception as exc:
            logger.error(f"Transcription error: {exc}")


    def _emit(self, event: PipelineEvent) -> None:
        """Dispatch an event to all registered callbacks."""
        for cb in self._callbacks.get(event.type, []):
            try:
                cb(event)
            except Exception as exc:
                logger.warning(f"Event callback error: {exc}")
