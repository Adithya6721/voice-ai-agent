#!/usr/bin/env python3
"""
scripts/benchmark_asr.py - Benchmark multiple ASR models.

Computes WER, RTF, Latency, and saves to result.md.
"""

import sys
import os
import time
import glob
import soundfile as sf
import numpy as np
import jiwer
import psutil
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voice_ai.inference.device import get_device
from voice_ai.asr.whisper import WhisperASR
from voice_ai.asr.parakeet import ParakeetASR

# Suppress debug logs
logger.remove()
logger.add(sys.stderr, level="INFO")

GROUND_TRUTH = {
    "s1.wav": "I think we should schedule a meeting with Rahul tomorrow at ten in the morning.",
    "s2.wav": "Uh, I actually wanted to reschedule the meeting because I have another class at that time.",
    "s3.wav": "Write a Python function that finds the maximum subarray sum using Kadane’s algorithm.",
    "s4.wav": "Please summarize this document and give me the three most important points as bullet points.",
    "s5.wav": "The weather in Chennai is quite hot today, but I still want to go out in the evening.",
    "s6.wav": "Can you remind me to submit the machine learning assignment before Friday and send the report to my professor?"
}

MODELS_TO_TEST = [
    {"name": "Whisper Base", "engine": "whisper", "id": "base"},
    {"name": "Whisper Small", "engine": "whisper", "id": "small"},
    {"name": "Distil-Whisper", "engine": "whisper", "id": "distil-small.en"},
    {"name": "Parakeet TDT v2", "engine": "parakeet", "id": "parakeet-tdt-v2"},
]

def clean_text(text: str) -> str:
    """Basic normalization for WER calculation."""
    text = text.lower()
    for p in ",.?!”“'":
        text = text.replace(p, "")
    text = text.replace("’", "")
    return text.strip()

def main():
    device = get_device(preference="cpu")
    print(f"Starting benchmark on {device.type.value.upper()}...")
    
    results = []
    
    for model_info in MODELS_TO_TEST:
        print(f"\nEvaluating: {model_info['name']}")
        
        # 1. Initialize model
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        if model_info["engine"] == "whisper":
            asr = WhisperASR(device=device, model_size=model_info["id"], compute_type="auto")
        elif model_info["engine"] == "parakeet":
            asr = ParakeetASR(device=device, model_size=model_info["id"])
            
        mem_after = process.memory_info().rss / (1024 * 1024)
        ram_used = mem_after - mem_before
        
        # Warmup
        asr.transcribe(np.zeros(16000, dtype=np.float32))
        
        # 2. Run benchmark
        model_wer = []
        model_rtf = []
        model_lat = []
        
        for wav_file, ref_text in GROUND_TRUTH.items():
            path = os.path.join("data", "benchmark", wav_file)
            if not os.path.exists(path):
                print(f"  Missing {wav_file} - skipping")
                continue
                
            audio, sr = sf.read(path)
            if len(audio.shape) > 1: audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            
            start = time.perf_counter()
            res = asr.transcribe(audio)
            latency = (time.perf_counter() - start) * 1000
            
            duration = len(audio) / sr
            rtf = (latency / 1000) / duration
            
            hyp = clean_text(res.text)
            ref = clean_text(ref_text)
            wer = jiwer.wer(ref, hyp) * 100
            
            model_wer.append(wer)
            model_rtf.append(rtf)
            model_lat.append(latency)
            
            print(f"  {wav_file}: WER={wer:.1f}% | RTF={rtf:.2f}")
            print(f"    Ref: {ref}")
            print(f"    Hyp: {hyp}")
            
        if model_wer:
            results.append({
                "Model": model_info["name"],
                "WER": np.mean(model_wer),
                "RTF": np.mean(model_rtf),
                "Latency": np.mean(model_lat),
                "RAM (MB)": ram_used
            })
            
    # 3. Write Report
    write_report(results, device.name)
    print("\n✅ Benchmark complete! See result.md")

def write_report(results, device_name):
    with open("result.md", "w", encoding="utf-8") as f:
        f.write(f"# ASR Benchmark Results\n\n")
        f.write(f"**Hardware Device:** {device_name}\n\n")
        f.write("| Model | Avg WER (%) | Avg RTF | Avg Latency (ms) | Est RAM (MB) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['Model']} | {r['WER']:.1f}% | {r['RTF']:.2f} | {r['Latency']:.0f} | ~{r['RAM (MB)']:.0f} |\n")

if __name__ == "__main__":
    main()
