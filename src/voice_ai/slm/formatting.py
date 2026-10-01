"""
voice_ai.slm.formatting — Detection and rendering of user formatting requests.

Handles:
  - bullet_list: "make them bullet points" -> "• Item 1\n• Item 2"
  - numbered_list: "numbered list" -> "1. Item 1\n2. Item 2"
  - code_block: "in code block" -> "```python\n...\n```"
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class Formatter:
    """Detects and renders explicit formatting requests."""

    @staticmethod
    def detect_format(text: str) -> str:
        """Detect formatting type requested in text."""
        text_lower = text.lower()
        if any(p in text_lower for p in ["bullet point", "bullet list", "bulleted list", "bullets"]):
            return "bullet_list"
        if any(p in text_lower for p in ["numbered list", "numbered points", "number list"]):
            return "numbered_list"
        if any(p in text_lower for p in ["code block", "as code", "in code"]):
            return "code_block"
        return "none"

    @classmethod
    def format_text(cls, text: str, format_type: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract items and format the text accordingly.

        Returns:
            (formatted_text, detected_format, format_data)
        """
        if not format_type or format_type in ("auto", "none"):
            format_type = cls.detect_format(text)


        if format_type == "bullet_list":
            # Strip formatting instructions from the text
            clean_prompt = re.sub(
                r",?\s*(make them|in|as)?\s*(bullet points?|bullet list|bullets)[.]?",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

            items = cls._split_items(clean_prompt)
            if len(items) > 1:
                formatted = "\n".join(f"• {item.capitalize()}" for item in items)
                return formatted, "bullet_list", {"items": items}
            return clean_prompt, "bullet_list", {"items": items}


        elif format_type == "numbered_list":
            clean_prompt = re.sub(
                r",?\s*(make them|in|as)?\s*(numbered list|numbered points)[.]?",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

            items = cls._split_items(clean_prompt)
            formatted = "\n".join(f"{idx + 1}. {item.capitalize()}" for idx, item in enumerate(items))
            return formatted, "numbered_list", {"items": items}

        elif format_type == "code_block":
            clean_prompt = re.sub(
                r",?\s*(in|as)?\s*(a\s+)?(code block)[.]?",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
            formatted = f"```\n{clean_prompt}\n```"
            return formatted, "code_block", {"raw_code": clean_prompt}

        return text, "none", {}

    @staticmethod
    def _split_items(text: str) -> List[str]:
        """Split a list phrase into clean list items."""
        # Check for "first ... second ... third ..."
        numbered_split = re.split(r"\b(?:first|second|third|fourth|fifth|1st|2nd|3rd)\b:?", text, flags=re.IGNORECASE)
        if len(numbered_split) > 2:
            return [re.sub(r"^[,\s]+|[,\s.]+$", "", s).strip() for s in numbered_split if s.strip()]

        # Check for phrase after "are " or "is " or "tasks: "
        lead_in = re.search(r"\b(?:are|is|include|including|following)\s+(.*)", text, flags=re.IGNORECASE)
        target = lead_in.group(1) if lead_in else text

        # Split on commas or "and"
        parts = re.split(r",|\band\b", target, flags=re.IGNORECASE)
        items = []
        count_m = re.search(r"\b(two|three|four|five|2|3|4|5)\s+(?:things|tasks|items|points|steps)\b", text, flags=re.IGNORECASE)
        for p in parts:
            p_clean = p.strip()
            if not p_clean:
                continue
            words = p_clean.split()
            if count_m and len(words) > 1 and "," not in target:
                items.extend(words)
            else:
                items.append(p_clean)

        return items if items else [text]

