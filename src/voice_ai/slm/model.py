"""
voice_ai.slm.model — Local Small Language Model (SLM) post-processing engine.

Implements BaseSLM using HuggingFace Transformers for completely local execution.
Supports:
  - SmolLM2 (360M / 135M)
  - Qwen2.5 (0.5B / 1.5B)
  - Llama 3.2 (1B)
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

import numpy as np
import psutil
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from voice_ai.inference.device import DeviceInfo, get_device
from voice_ai.slm.base import BaseSLM, SLMResult
from voice_ai.slm.cleanup import TextCleaner
from voice_ai.slm.formatting import Formatter
from voice_ai.slm.intent import IntentExtractor


DEFAULT_SLM_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

SYSTEM_PROMPT = (
    "You are a local voice transcript post-processor. "
    "Given a raw speech transcript, perform three tasks:\n"
    "1. Text cleanup: remove spoken filler words (uh, um, you know, etc.) and stutter/word repetitions. "
    "Restore proper capitalization and punctuation. Preserve the exact meaning.\n"
    "2. Intent recognition: classify the actionable intent ('reminder', 'schedule', 'summarize', 'task', 'format', or 'none') "
    "and extract action, object, and time parameters.\n"
    "3. Formatting: detect requested formats ('bullet_list', 'numbered_list', 'code_block', or 'none').\n\n"
    "Example 1:\n"
    'Raw: "uh I think we should schedule schedule a meeting tomorrow"\n'
    '{"clean_text": "I think we should schedule a meeting tomorrow.", "intent": "schedule", "intent_data": {"action": "meeting", "time": "tomorrow"}, "format": "none", "final_output": "I think we should schedule a meeting tomorrow."}\n\n'
    "Example 2:\n"
    'Raw: "remind me to submit the assignment tomorrow at 6"\n'
    '{"clean_text": "Remind me to submit the assignment tomorrow at 6.", "intent": "reminder", "intent_data": {"action": "submit", "object": "assignment", "time": "tomorrow at 6"}, "format": "none", "final_output": "Remind me to submit the assignment tomorrow at 6."}\n\n'
    "Example 3:\n"
    'Raw: "the three tasks are testing deployment and documentation make them bullet points"\n'
    '{"clean_text": "The three tasks are testing, deployment, and documentation.", "intent": "format", "intent_data": {}, "format": "bullet_list", "final_output": "• Testing\\n• Deployment\\n• Documentation"}\n\n'
    "Respond strictly with a single JSON object. Do not output anything else."

)



class LocalSLM(BaseSLM):
    """
    Local SLM inference engine powered by HuggingFace Transformers.

    Decouples model loading from inference:
      - Initialized once upon startup.
      - Cached in memory for low-latency calls.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_SLM_MODEL,
        device: Optional[DeviceInfo] = None,
        max_new_tokens: int = 90,
        temperature: float = 0.1,
    ) -> None:
        self._model_id = model_id
        self._device_info = device or get_device()
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        # Resolve torch device
        if self._device_info.is_cuda:
            self._torch_device = torch.device("cuda")
            self._dtype = torch.float16
        else:
            self._torch_device = torch.device("cpu")
            self._dtype = torch.float32

        logger.info(
            f"Loading Local SLM: {model_id} | device={self._torch_device} | dtype={self._dtype}"
        )
        t_start = time.perf_counter()
        
        proc = psutil.Process(os.getpid())
        ram_before = proc.memory_info().rss / (1024 * 1024)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self._dtype,
        )
        self._model.to(self._torch_device)
        self._model.eval()

        ram_after = proc.memory_info().rss / (1024 * 1024)
        load_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            f"Local SLM loaded in {load_ms:.0f}ms | RAM footprint: ~{ram_after - ram_before:.0f} MB"
        )

    @property
    def name(self) -> str:
        return f"LocalSLM[{self._model_id.split('/')[-1]}]"

    def process(self, transcript: str) -> SLMResult:
        """
        Process raw ASR transcript and return structured SLMResult.
        """
        if not transcript or not transcript.strip():
            return SLMResult(
                clean_text="",
                raw_transcript=transcript,
                model_name=self.name,
            )

        t_start = time.perf_counter()
        raw = transcript.strip()

        # Build prompt using chat template or fallback format
        prompt = self._build_prompt(raw)

        # Run inference
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._torch_device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # Slice out the generated completion tokens
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            completion = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        except Exception as exc:
            logger.error(f"SLM generation failed: {exc}")
            completion = ""

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Parse structured output from completion
        result = self._parse_completion(raw, completion, latency_ms)
        logger.debug(
            f"SLM processed in {latency_ms:.1f}ms | Intent: {result.intent} | Format: {result.format}"
        )
        return result

    def _build_prompt(self, transcript: str) -> str:
        """Construct the prompt using ChatML template or standard format."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Raw transcript: "{transcript}"'},
        ]
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback template
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nRaw transcript: \"{transcript}\"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _parse_completion(self, raw: str, completion: str, latency_ms: float) -> SLMResult:
        """Parse JSON response from SLM, with rule-based safety fallback."""
        data: Dict[str, Any] = {}

        # Look for JSON block in markdown or text
        json_match = re.search(r"\{[\s\S]*\}", completion)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                data = {}

        # 1. Cleaned text extraction or fallback
        clean_text = data.get("clean_text")
        if not clean_text or not isinstance(clean_text, str) or len(clean_text) < 2:
            clean_text = TextCleaner.clean(raw)
        else:
            clean_text = TextCleaner.clean(clean_text)

        # 2. Intent extraction or fallback
        intent = data.get("intent", "none")
        intent_data = data.get("intent_data", {})
        if not isinstance(intent_data, dict):
            intent_data = {}

        if intent in ["none", ""] or not intent_data:
            heuristic_intent = IntentExtractor.extract_fallback(raw)
            if heuristic_intent.get("intent") != "none":
                intent = heuristic_intent["intent"]
                intent_data = {k: v for k, v in heuristic_intent.items() if k != "intent"}

        # 3. Formatting extraction or fallback
        format_type = data.get("format", "none")
        if "|" in format_type or format_type not in ["bullet_list", "numbered_list", "code_block"]:
            format_type = "none"

        final_output = data.get("final_output")

        # Check heuristic formatter if model didn't catch explicit formatting
        formatted_text, detected_fmt, fmt_data = Formatter.format_text(raw, format_type=format_type)
        if detected_fmt != "none" and format_type == "none":
            format_type = detected_fmt
            final_output = formatted_text
        elif not final_output:
            final_output = formatted_text if format_type != "none" else clean_text


        return SLMResult(
            clean_text=clean_text,
            intent=intent,
            intent_data=intent_data,
            format=format_type,
            format_data=fmt_data,
            final_output=final_output,
            raw_transcript=raw,
            latency_ms=latency_ms,
            model_name=self.name,
        )
