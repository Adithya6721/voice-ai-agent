"""
voice_ai.asr.factory — ASR model factory and dynamic fallback orchestrator.

Provides:
  - Default compute-aware fallback chain (Parakeet v2 -> Whisper Small -> Distil-Whisper -> Whisper Base)
  - `FallbackASR`: BaseASR implementation that automatically falls back to lower compute models
    if high-tier models fail at initialization or runtime.
  - `create_asr_engine`: Factory function to instantiate single models or fallback wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger

from voice_ai.asr.base import BaseASR, TranscriptionResult
from voice_ai.asr.parakeet import ParakeetASR
from voice_ai.asr.whisper import WhisperASR
from voice_ai.inference.device import DeviceInfo, get_device

# Default fallback order prioritized by performance/accuracy on standard compute:
# 1. Parakeet TDT v2 (5.9% WER, 0.21 RTF, ~445MB RAM)
# 2. Whisper Small   (6.0% WER, 0.50 RTF, ~306MB RAM)
# 3. Distil-Whisper  (7.9% WER, 0.40 RTF, ~160MB RAM)
# 4. Whisper Base    (8.2% WER, 0.19 RTF, ~159MB RAM)
DEFAULT_FALLBACK_CHAIN: List[Dict[str, Any]] = [
    {
        "engine": "parakeet",
        "model_size": "parakeet-tdt-v2",
        "name": "Parakeet TDT v2",
    },
    {
        "engine": "whisper",
        "model_size": "small",
        "name": "Whisper Small",
    },
    {
        "engine": "whisper",
        "model_size": "distil-small.en",
        "name": "Distil-Whisper",
    },
    {
        "engine": "whisper",
        "model_size": "base",
        "name": "Whisper Base",
    },
]


def create_single_model(
    engine: str,
    model_size: str = "small",
    device: Optional[DeviceInfo] = None,
    language: Optional[str] = "en",
    beam_size: int = 5,
    compute_type: str = "auto",
    **kwargs: Any,
) -> BaseASR:
    """
    Instantiate a specific ASR engine.

    Args:
        engine:     "parakeet" | "whisper"
        model_size: Model identifier e.g. "parakeet-tdt-v2", "small", "base"
        device:     DeviceInfo target
    """
    if device is None:
        device = get_device()

    engine_lower = engine.lower()
    if engine_lower == "parakeet":
        return ParakeetASR(device=device, model_size=model_size)
    elif engine_lower == "whisper":
        return WhisperASR(
            device=device,
            model_size=model_size,
            language=language,
            beam_size=beam_size,
            compute_type=compute_type,
        )
    else:
        raise ValueError(f"Unknown ASR engine '{engine}'")


class FallbackASR(BaseASR):
    """
    BaseASR implementation wrapping a prioritized chain of ASR models.

    Attempts to load the highest priority model first. If loading fails
    (e.g., missing dependencies, insufficient VRAM/RAM, network issue),
    it gracefully steps down to the next model in the fallback chain.
    """

    def __init__(
        self,
        chain: Optional[List[Dict[str, Any]]] = None,
        device: Optional[DeviceInfo] = None,
        language: Optional[str] = "en",
        beam_size: int = 5,
        compute_type: str = "auto",
    ) -> None:
        self.device = device or get_device()
        self.chain = chain if chain is not None else DEFAULT_FALLBACK_CHAIN
        self.language = language
        self.beam_size = beam_size
        self.compute_type = compute_type

        self.active_model: Optional[BaseASR] = None
        self._active_index: int = -1

        self._initialize_fallback_chain()

    def _initialize_fallback_chain(self) -> None:
        """Attempt initialization of candidates in order until one succeeds."""
        errors: List[str] = []
        for idx, item in enumerate(self.chain):
            engine = item.get("engine", "whisper")
            model_size = item.get("model_size", "small")
            name = item.get("name", f"{engine}-{model_size}")

            logger.info(f"Attempting ASR candidate [{idx + 1}/{len(self.chain)}]: {name}…")
            try:
                model = create_single_model(
                    engine=engine,
                    model_size=model_size,
                    device=self.device,
                    language=self.language,
                    beam_size=self.beam_size,
                    compute_type=self.compute_type,
                )
                self.active_model = model
                self._active_index = idx
                logger.info(f"Activated ASR model: {model.name}")
                return
            except Exception as exc:
                logger.warning(f"Candidate '{name}' initialization failed: {exc}")
                errors.append(f"[{name}] {exc}")

        raise RuntimeError(
            "All ASR models in fallback chain failed to initialize!\nDetails:\n"
            + "\n".join(errors)
        )

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if self.active_model is None:
            raise RuntimeError("FallbackASR has no active model loaded.")

        try:
            return self.active_model.transcribe(audio)
        except Exception as exc:
            logger.error(
                f"Active model '{self.active_model.name}' failed during transcription: {exc}"
            )
            # Try next models in fallback chain at runtime
            while self._active_index + 1 < len(self.chain):
                self._active_index += 1
                next_item = self.chain[self._active_index]
                name = next_item.get("name", next_item.get("model_size", "unknown"))
                logger.info(f"Attempting runtime fallback to [{self._active_index + 1}/{len(self.chain)}]: {name}…")
                try:
                    self.active_model = create_single_model(
                        engine=next_item.get("engine", "whisper"),
                        model_size=next_item.get("model_size", "small"),
                        device=self.device,
                        language=self.language,
                        beam_size=self.beam_size,
                        compute_type=self.compute_type,
                    )
                    logger.info(f"Runtime fallback succeeded. Active model is now '{self.active_model.name}'")
                    return self.active_model.transcribe(audio)
                except Exception as fb_exc:
                    logger.warning(f"Runtime fallback to '{name}' failed: {fb_exc}")

            raise exc

    @property
    def name(self) -> str:
        if self.active_model:
            return f"FallbackASR[{self.active_model.name}]"
        return "FallbackASR[Uninitialized]"

    def warmup(self, duration_s: float = 1.0) -> None:
        if self.active_model:
            self.active_model.warmup(duration_s=duration_s)


def create_asr_engine(
    engine_type: str = "fallback",
    model_size: str = "parakeet-tdt-v2",
    fallback_chain: Optional[List[Dict[str, Any]]] = None,
    device: Optional[DeviceInfo] = None,
    language: Optional[str] = "en",
    beam_size: int = 5,
    compute_type: str = "auto",
) -> BaseASR:
    """
    Main factory entrypoint for creating ASR models.

    Args:
        engine_type: "fallback" | "parakeet" | "whisper"
        model_size: Specific model size when engine_type is single engine.
        fallback_chain: Custom fallback chain list if engine_type == "fallback".
        device: Target DeviceInfo.
    """
    if engine_type.lower() == "fallback" or engine_type.lower() == "auto":
        return FallbackASR(
            chain=fallback_chain,
            device=device,
            language=language,
            beam_size=beam_size,
            compute_type=compute_type,
        )

    return create_single_model(
        engine=engine_type,
        model_size=model_size,
        device=device,
        language=language,
        beam_size=beam_size,
        compute_type=compute_type,
    )
