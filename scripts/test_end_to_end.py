#!/usr/bin/env python3
"""
scripts/test_end_to_end.py — Full Pipeline Verification:
Audio -> VAD -> ASR -> Raw Transcript -> SLM -> Final Output

Demonstrates the complete local voice processing pipeline on real audio.
"""

import os
import sys
import time
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from loguru import logger
from voice_ai.inference.device import get_device

from voice_ai.vad.silero import SileroVAD
from voice_ai.vad.detector import SpeechDetector
from voice_ai.asr import create_asr_engine, Transcriber
from voice_ai.slm import LocalSLM

logger.remove()
logger.add(sys.stderr, level="INFO")


def run_pipeline_on_file(wav_path: str, asr_engine, slm_engine):
    print("\n" + "=" * 65)
    print(f"[*] PROCESSING AUDIO: {os.path.basename(wav_path)}")
    print("=" * 65)

    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # 1. VAD & Speech Detection
    print("\n[Step 1: VAD -- Voice Activity Detection]")
    vad = SileroVAD(threshold=0.5)
    detector = SpeechDetector(vad=vad, min_speech_duration_ms=250)

    # Feed chunks of 512 samples (32ms) to detector
    chunk_size = 512
    speech_segments = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        for seg in detector.process(chunk):
            speech_segments.append(seg)

    print(f"  VAD detected {len(speech_segments)} speech segment(s) (total duration: {len(audio)/sr:.2f}s)")

    # 2. ASR -- Automatic Speech Recognition
    print("\n[Step 2: ASR -- Speech Recognition]")
    t0 = time.perf_counter()
    transcription = asr_engine.transcribe(audio)
    raw_text = transcription.text
    asr_time = (time.perf_counter() - t0) * 1000

    print(f"  Raw ASR Transcript: \"{raw_text}\"")
    print(f"  ASR Model: {asr_engine.name} | Latency: {asr_time:.0f}ms")

    # 3. SLM -- Post-Processing Layer
    print("\n[Step 3: SLM -- Local Post-Processing]")
    t1 = time.perf_counter()
    slm_result = slm_engine.process(raw_text)
    slm_time = (time.perf_counter() - t1) * 1000

    print(f"  Cleaned Text: \"{slm_result.clean_text}\"")
    print(f"  Intent:       {slm_result.intent} ({slm_result.intent_data})")
    print(f"  Format:       {slm_result.format}")
    print(f"  SLM Model:    {slm_result.model_name} | Latency: {slm_time:.0f}ms")

    # 4. Final Output
    print("\n" + "-" * 65)
    print(">> FINAL STRUCTURED OUTPUT:")

    print("-" * 65)
    print(slm_result.final_output)
    print("=" * 65 + "\n")


def main():
    device = get_device()
    print(f"Initializing complete pipeline on device: {device.type.value.upper()}...")

    # Load ASR (Fallback chain: Parakeet -> Whisper Small -> Distil -> Base)
    print("Initializing ASR Engine...")
    asr = create_asr_engine(engine_type="fallback", device=device)

    # Load SLM (SmolLM2-360M)
    print("Initializing Local SLM Engine...")
    slm = LocalSLM(model_id="HuggingFaceTB/SmolLM2-360M-Instruct", device=device)

    # Run on recorded samples
    samples = ["s1.wav", "s2.wav", "s4.wav"]
    for s in samples:
        path = os.path.join("data", "benchmark", s)
        if os.path.exists(path):
            run_pipeline_on_file(path, asr, slm)


if __name__ == "__main__":
    main()
