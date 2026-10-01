"""
voice_ai.slm.base — abstract base class and structured results for Local SLM.

Provides:
  - `SLMResult`: Structured container for cleaned text, intent, formatting, and final output.
  - `BaseSLM`: Interface decouples the SLM backend from pipeline logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SLMResult:
    """
    Structured post-processing result from the SLM.

    Attributes:
        clean_text:       Cleaned transcript with fillers & repetitions removed,
                          punctuation restored, capitalization fixed.
        intent:           Actionable intent ("reminder", "schedule", "summarize", "format", "none", etc.)
        intent_data:      Key-value pairs for actionable parameters (action, object, time, etc.)
        format:           Detected formatting style ("bullet_list", "numbered_list", "code_block", "none")
        format_data:      Structured formatting elements (e.g. list items).
        final_output:     Rendered text ready for presentation/insertion.
        raw_transcript:   The original raw ASR transcript.
        latency_ms:       Inference latency in milliseconds.
        model_name:       Name of the SLM engine.
    """
    clean_text: str
    intent: str = "none"
    intent_data: Dict[str, Any] = field(default_factory=dict)
    format: str = "none"
    format_data: Dict[str, Any] = field(default_factory=dict)
    final_output: str = ""
    raw_transcript: str = ""
    latency_ms: float = 0.0
    model_name: str = ""

    def __post_init__(self) -> None:
        if not self.final_output:
            self.final_output = self.clean_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean_text": self.clean_text,
            "intent": self.intent,
            "intent_data": self.intent_data,
            "format": self.format,
            "format_data": self.format_data,
            "final_output": self.final_output,
            "raw_transcript": self.raw_transcript,
            "latency_ms": round(self.latency_ms, 1),
            "model_name": self.model_name,
        }


class BaseSLM(ABC):
    """
    Abstract Small Language Model interface for local post-processing.
    """

    @abstractmethod
    def process(self, transcript: str) -> SLMResult:
        """
        Process a raw ASR transcript into cleaned text, recognized intent, and formatted output.

        Args:
            transcript: Raw text output from ASR.

        Returns:
            SLMResult
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self.__class__.__name__

    def warmup(self) -> None:
        """Optional warmup pass to pre-load weights into CPU/GPU cache."""
        try:
            self.process("Warmup transcription test")
        except Exception:
            pass
