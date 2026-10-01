"""
voice_ai.slm — Local Small Language Model (SLM) post-processing package.

Provides:
  - BaseSLM, SLMResult: Interface and structured output dataclass
  - LocalSLM: Transformers-based local model runner (SmolLM2, Qwen2.5, etc.)
  - TextCleaner: Deterministic filler and stutter cleaner
  - IntentExtractor: Structured intent & slot extraction
  - Formatter: Formatting request detection (bullets, numbered lists, code)
"""

from voice_ai.slm.base import BaseSLM, SLMResult
from voice_ai.slm.cleanup import TextCleaner
from voice_ai.slm.formatting import Formatter
from voice_ai.slm.intent import IntentExtractor
from voice_ai.slm.model import LocalSLM

__all__ = [
    "BaseSLM",
    "SLMResult",
    "LocalSLM",
    "TextCleaner",
    "IntentExtractor",
    "Formatter",
]
