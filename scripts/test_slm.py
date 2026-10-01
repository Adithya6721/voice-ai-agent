#!/usr/bin/env python3
"""
scripts/test_slm.py — CLI test harness for Local SLM post-processing.

Usage:
    # Test on default example sentences
    python scripts/test_slm.py

    # Test custom transcript
    python scripts/test_slm.py --text "uh remind me to submit the assignment tomorrow at 6"

    # Run evaluation suite across datasets
    python scripts/test_slm.py --eval
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loguru import logger
from voice_ai.slm import LocalSLM, TextCleaner, IntentExtractor, Formatter

# Configure cleaner logger
logger.remove()
logger.add(sys.stderr, level="INFO")

SAMPLE_TRANSCRIPTS = [
    # 1. Text cleanup test
    "uh I think we should schedule schedule a meeting tomorrow",
    # 2. Intent recognition test
    "remind me to submit the machine learning assignment tomorrow at 6",
    # 3. Formatting test
    "the three things we need to finish are testing deployment and documentation, make them bullet points",
    # 4. Long / combined sentence
    "uh I actually wanted to reschedule the meeting with Rahul because I have another class tomorrow at 10",
]


def print_result(res):
    print("=" * 60)
    print("RAW TRANSCRIPT:")
    print(res.raw_transcript)
    print("\nCLEANED TEXT:")
    print(res.clean_text)
    print("\nINTENT:")
    print(json.dumps({"intent": res.intent, **res.intent_data}, indent=2))
    print("\nFORMAT:")
    print(res.format)
    print("\nFINAL OUTPUT:")
    print(res.final_output)
    print(f"\n[Model: {res.model_name} | Latency: {res.latency_ms:.1f}ms]")
    print("=" * 60 + "\n")


def run_evaluation(slm):
    eval_dir = os.path.join("data", "evaluation", "slm")
    if not os.path.exists(eval_dir):
        print(f"Evaluation directory not found: {eval_dir}")
        return

    print("\n" + "=" * 60)
    print("RUNNING LOCAL SLM EVALUATION SUITE")
    print("=" * 60)

    for eval_file in ["cleanup.jsonl", "intent.jsonl", "formatting.jsonl"]:
        path = os.path.join(eval_dir, eval_file)
        if not os.path.exists(path):
            continue

        print(f"\n--- Evaluating: {eval_file} ---")
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        for item in lines:
            raw = item["raw"]
            res = slm.process(raw)
            print(f"\n[ID: {item.get('id', 'N/A')}]")
            print(f"  Input:    {raw}")
            print(f"  Clean:    {res.clean_text}")
            print(f"  Intent:   {res.intent} ({res.intent_data})")
            print(f"  Format:   {res.format}")
            print(f"  Output:\n{res.final_output}")
            print(f"  Latency:  {res.latency_ms:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Test Local SLM Post-Processing")
    parser.add_argument("--text", type=str, default=None, help="Custom transcript to test")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct", help="HuggingFace model ID")
    parser.add_argument("--eval", action="store_true", help="Run benchmark on evaluation dataset")
    args = parser.parse_args()

    print(f"Initializing SLM model ({args.model})...")
    slm = LocalSLM(model_id=args.model)

    if args.eval:
        run_evaluation(slm)
    elif args.text:
        res = slm.process(args.text)
        print_result(res)
    else:
        print("\nTesting default sample transcripts:\n")
        for sample in SAMPLE_TRANSCRIPTS:
            res = slm.process(sample)
            print_result(res)


if __name__ == "__main__":
    main()
