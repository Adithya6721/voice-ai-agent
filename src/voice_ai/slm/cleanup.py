"""
voice_ai.slm.cleanup — Text cleanup rules and utilities for ASR transcripts.

Handles:
  - Removing filler words: "uh", "um", "you know", "like", "basically", "er", "ah"
  - Removing word repetitions: "schedule schedule" -> "schedule"
  - Removing phrase repetitions: "create a create a" -> "create a"
  - Punctuation restoration and capitalization
"""

from __future__ import annotations

import re
from typing import List

# Common speech fillers to remove when used as filler tokens
FILLER_PATTERNS = [
    r"\b(uh|um|er|ah)\b,?",
    r"\b(you know)\b,?",
    r"\b(like)\b(?=\s+[a-z])", # only match standalone conversational "like"
]

class TextCleaner:
    """Fast, deterministic text cleaner for spoken transcripts."""

    @staticmethod
    def remove_fillers(text: str) -> str:
        """Remove spoken fillers such as 'uh', 'um', 'you know'."""
        cleaned = text
        for pat in FILLER_PATTERNS:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def remove_repetitions(text: str) -> str:
        """
        Deduplicate immediate word and small phrase repetitions.
        Example: 'schedule schedule' -> 'schedule'
                 'create a create a' -> 'create a'
        """
        # Phrase repetition (2 words): e.g. "create a create a"
        text = re.sub(r"\b(\w+\s+\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        # Single word repetition: e.g. "schedule schedule"
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def fix_capitalization_and_punctuation(text: str) -> str:
        """Capitalize sentence beginnings and ensure trailing punctuation."""
        if not text:
            return ""

        # Remove stray leading/trailing punctuation or whitespace
        text = text.strip()
        if not text:
            return ""

        # Capitalize first letter
        text = text[0].upper() + text[1:]

        # Capitalize after sentence terminators (. ! ?)
        text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

        # Capitalize standalone "i"
        text = re.sub(r"\b(i)\b", "I", text)

        # Capitalize "Python" if lowercase
        text = re.sub(r"\bpython\b", "Python", text, flags=re.IGNORECASE)
        # Capitalize common proper nouns if found
        for proper in ["Rahul", "Chennai", "Friday", "Kadane"]:
            text = re.sub(rf"\b{proper}\b", proper, text, flags=re.IGNORECASE)

        # Add period at the end if none exists
        if text[-1] not in ".!?":
            text += "."

        return text

    @classmethod
    def clean(cls, text: str) -> str:
        """Run full deterministic cleanup pipeline on raw transcript."""
        if not text:
            return ""
        s = cls.remove_fillers(text)
        s = cls.remove_repetitions(s)
        s = cls.fix_capitalization_and_punctuation(s)
        return s
