"""
voice_ai.pipeline.events — event types for the audio pipeline.

Events decouple the pipeline internals from consumers
(CLI, API server, future Electron IPC, etc.).

The pipeline emits events; consumers subscribe and react.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class EventType(Enum):
    RECORDING_STARTED = auto()
    RECORDING_STOPPED = auto()
    SPEECH_STARTED = auto()
    SPEECH_ENDED = auto()
    TRANSCRIPTION_READY = auto()
    SLM_READY = auto()
    ERROR = auto()


@dataclass
class PipelineEvent:
    """Base class for all pipeline events."""
    type: EventType
    timestamp_s: float = 0.0


@dataclass
class SpeechStartedEvent(PipelineEvent):
    type: EventType = EventType.SPEECH_STARTED


@dataclass
class SpeechEndedEvent(PipelineEvent):
    type: EventType = EventType.SPEECH_ENDED
    duration_s: float = 0.0
    audio: Optional[np.ndarray] = None


@dataclass
class TranscriptionEvent(PipelineEvent):
    type: EventType = EventType.TRANSCRIPTION_READY
    text: str = ""
    language: str = "en"
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    rtf: float = 0.0


@dataclass
class SLMEvent(PipelineEvent):
    type: EventType = EventType.SLM_READY
    raw_transcript: str = ""
    clean_text: str = ""
    intent: str = "none"
    intent_data: dict = field(default_factory=dict)
    format: str = "none"
    format_data: dict = field(default_factory=dict)
    final_output: str = ""
    latency_ms: float = 0.0


@dataclass
class ErrorEvent(PipelineEvent):
    type: EventType = EventType.ERROR
    message: str = ""
    exception: Optional[Exception] = None

