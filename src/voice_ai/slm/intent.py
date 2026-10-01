"""
voice_ai.slm.intent — Intent recognition and structured field extraction.

Supported intents:
  - reminder:  {"intent": "reminder", "action": "...", "object": "...", "time": "..."}
  - schedule:  {"intent": "schedule", "action": "...", "object": "...", "time": "..."}
  - summarize: {"intent": "summarize", "action": "summarize", "object": "..."}
  - task:      {"intent": "task", "action": "...", "object": "...", "time": "..."}
  - none:      {"intent": "none"}
"""

from __future__ import annotations

import re
from typing import Any, Dict


class IntentExtractor:
    """Extract actionable intents and parameters from transcript."""

    @staticmethod
    def extract_fallback(text: str) -> Dict[str, Any]:
        """
        Fast heuristic extractor when SLM is unavailable or for validation.
        """
        text_lower = text.lower().strip()

        # 1. Reminder
        # e.g. "remind me to submit the machine learning assignment tomorrow at 6"
        remind_m = re.search(r"remind me to\s+(?P<action>\w+)\s+(?P<rest>.*)", text_lower)
        if remind_m:
            action = remind_m.group("action")
            rest = remind_m.group("rest")
            time_m = re.search(r"\b(tomorrow(?:\s+at\s+[\w:]+)?|today|next\s+\w+|at\s+\d+.*|before\s+\w+)", rest)
            time_str = time_m.group(0) if time_m else ""
            obj = rest[:time_m.start()].strip() if time_m else rest.strip()
            # Clean trailing punct
            obj = re.sub(r"[.,!?]$", "", obj)
            time_str = re.sub(r"[.,!?]$", "", time_str)
            res = {"intent": "reminder", "action": action, "object": obj}
            if time_str:
                res["time"] = time_str
            return res

        # 2. Scheduling / Meeting
        # e.g. "schedule a meeting with Rahul tomorrow at ten in the morning"
        schedule_m = re.search(r"(?:schedule|reschedule)\s+(?:a\s+)?(?P<action>meeting|call|session)?\s*(?:with\s+(?P<person>[\w\s]+?))?\s+(?P<time>(?:tomorrow|today|on\s+\w+|at\s+[\w\s]+).*)", text_lower)
        if schedule_m or ("schedule" in text_lower and "meeting" in text_lower):
            action = "meeting"
            time_str = ""
            obj = ""
            if schedule_m:
                time_str = schedule_m.group("time") or ""
                obj = (schedule_m.group("person") or "").strip()
            if not obj:
                with_m = re.search(r"with\s+([A-Za-z]+)", text)
                if with_m:
                    obj = with_m.group(1)
            if not time_str:
                t_m = re.search(r"\b(tomorrow.*|at\s+\d+.*)", text_lower)
                if t_m:
                    time_str = t_m.group(0)

            time_str = re.sub(r"[.,!?]$", "", time_str)
            res = {"intent": "schedule", "action": "meeting"}
            if obj:
                res["object"] = obj
            if time_str:
                res["time"] = time_str
            return res

        # 3. Summarization
        # e.g. "please summarize this document and give me the three most important points"
        if "summarize" in text_lower or "summary" in text_lower:
            doc_m = re.search(r"summarize\s+(this\s+\w+|the\s+\w+)", text_lower)
            target = doc_m.group(1) if doc_m else "this document"
            return {"intent": "summarize", "action": "summarize", "object": target}

        # 4. Action / Task
        # e.g. "send the report to my professor before Friday"
        send_m = re.search(r"\b(send|email|submit|create)\s+(?P<obj>.*?)(?:\s+to\s+(?P<target>[\w\s]+?))?(?:\s+(?P<time>before\s+\w+|by\s+\w+|tomorrow.*))?$", text_lower)
        if send_m and ("send" in text_lower or "create a python" in text_lower):
            action = send_m.group(1)
            obj = (send_m.group("obj") or "").strip()
            res = {"intent": "task", "action": action}
            if obj:
                res["object"] = re.sub(r"[.,!?]$", "", obj)
            if send_m.group("target"):
                res["target"] = send_m.group("target").strip()
            if send_m.group("time"):
                res["time"] = send_m.group("time").strip()
            return res

        return {"intent": "none"}
