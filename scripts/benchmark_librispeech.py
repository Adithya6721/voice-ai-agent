#!/usr/bin/env python3
"""
scripts/benchmark_librispeech.py — Benchmark ASR and VAD+ASR pipeline on LibriSpeech test-clean.

Features:
  - Streams samples directly from HuggingFace (`openslr/librispeech_asr`, 'clean', split='test')
  - Evaluates both:
      1. Standalone ASR (Direct audio -> ASR)
      2. Full VAD + ASR Pipeline (Audio -> Silero VAD chunking -> ASR)
  - Calculates WER (Word Error Rate), RTF (Real-Time Factor), Latency (ms), and RAM usage (MB)
  - Extrapolates total estimated time to run the complete 2,620 test-clean dataset (~5.4 hours of audio)
  - Outputs a structured markdown report and prints results.

Usage:
    # Run on 15 samples (quick sanity check & extrapolation)
    python scripts/benchmark_librispeech.py --samples 15

    # Run only Standalone ASR or only Pipeline
    python scripts/benchmark_librispeech.py --samples 10 --mode asr
"""

import argparse
import io
import os
import sys
import time
from typing import Any, Dict, List, Tuple

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jiwer
import numpy as np
import psutil
import soundfile as sf
from datasets import Audio, load_dataset
from loguru import logger

from voice_ai.asr.parakeet import ParakeetASR
from voice_ai.asr.whisper import WhisperASR
from voice_ai.inference.device import get_device
from voice_ai.vad.detector import SpeechDetector
from voice_ai.vad.silero import SileroVAD

# Suppress debug logs
logger.remove()
logger.add(sys.stderr, level="ERROR")

# Total size of LibriSpeech test.clean:
# 2,620 utterances, approx 5.4 hours (19,440 seconds)
TOTAL_LIBRISPEECH_CLEAN_SAMPLES = 2620
TOTAL_LIBRISPEECH_CLEAN_HOURS = 5.4
TOTAL_LIBRISPEECH_CLEAN_SECONDS = TOTAL_LIBRISPEECH_CLEAN_HOURS * 3600

MODELS_CONFIG = [
    {"name": "Parakeet TDT v2", "engine": "parakeet", "id": "parakeet-tdt-v2"},
    {"name": "Whisper Small", "engine": "whisper", "id": "small"},
    {"name": "Distil-Whisper", "engine": "whisper", "id": "distil-small.en"},
    {"name": "Whisper Base", "engine": "whisper", "id": "base"},
]


def normalize_text(text: str) -> str:
    """Normalize text for standard LibriSpeech WER evaluation."""
    text = text.lower()
    for char in ",.?!'\"`;:-()[]{}":
        text = text.replace(char, "")
    text = text.replace("’", "")
    return " ".join(text.split()).strip()


def load_sample_stream(num_samples: int) -> List[Dict[str, Any]]:
    """Stream and cache N samples into memory for rapid cross-model evaluation."""
    print(f"[*] Streaming {num_samples} samples from LibriSpeech test-clean...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    cached_samples = []
    total_audio_s = 0.0

    for idx, row in enumerate(ds):
        if idx >= num_samples:
            break
        # Read audio array with soundfile
        audio_arr, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if len(audio_arr.shape) > 1:
            audio_arr = audio_arr.mean(axis=1)
        audio_arr = audio_arr.astype(np.float32)

        dur_s = len(audio_arr) / sr
        total_audio_s += dur_s

        cached_samples.append({
            "id": row.get("id", f"sample_{idx+1}"),
            "audio": audio_arr,
            "sr": sr,
            "duration_s": dur_s,
            "reference": row["text"],
        })
        print(f"    Loaded sample {idx+1}/{num_samples}: {dur_s:.2f}s | Ref: {row['text'][:40]}...", end="\r", flush=True)

    print(f"\n[+] Streamed {len(cached_samples)} samples (Total audio: {total_audio_s:.1f}s / {total_audio_s/60:.2f} mins)\n", flush=True)
    return cached_samples


def run_standalone_asr(asr, sample: Dict[str, Any]) -> Tuple[str, float]:
    """Run direct ASR transcription on audio."""
    t0 = time.perf_counter()
    res = asr.transcribe(sample["audio"])
    latency_ms = (time.perf_counter() - t0) * 1000
    return res.text, latency_ms


def run_pipeline(vad, detector, asr, sample: Dict[str, Any]) -> Tuple[str, float]:
    """Run VAD chunking -> ASR on speech segments."""
    audio = sample["audio"]
    chunk_size = 512
    t0 = time.perf_counter()

    detector.reset()
    segments = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        for seg in detector.process(chunk):
            segments.append(seg)

    # In case audio ended without trailing silence flush
    if not segments:
        segments = [audio]

    # Transcribe segments
    transcripts = []
    for seg in segments:
        res = asr.transcribe(seg)
        if res.text.strip():
            transcripts.append(res.text.strip())

    total_text = " ".join(transcripts)
    latency_ms = (time.perf_counter() - t0) * 1000
    return total_text, latency_ms


def evaluate_model(model_info: Dict[str, str], samples: List[Dict[str, Any]], run_vad: bool = True, device=None) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluating Model: {model_info['name']}")
    print(f"=======================================================")

    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)

    # Load ASR
    if model_info["engine"] == "parakeet":
        asr = ParakeetASR(device=device, model_size=model_info["id"])
    else:
        asr = WhisperASR(device=device, model_size=model_info["id"])

    # Warmup
    asr.transcribe(np.zeros(16000, dtype=np.float32))

    mem_after = proc.memory_info().rss / (1024 * 1024)
    ram_mb = max(0, mem_after - mem_before)

    # Initialize VAD if needed
    vad = SileroVAD(threshold=0.5) if run_vad else None
    detector = SpeechDetector(vad=vad, min_speech_duration_ms=250) if run_vad else None

    asr_wer_list, asr_lat_list = [], []
    pipe_wer_list, pipe_lat_list = [], []
    durations = []

    for s in samples:
        ref_clean = normalize_text(s["reference"])
        durations.append(s["duration_s"])

        # 1. Standalone ASR
        hyp_asr, lat_asr = run_standalone_asr(asr, s)
        hyp_asr_clean = normalize_text(hyp_asr)
        wer_asr = jiwer.wer(ref_clean, hyp_asr_clean) * 100
        asr_wer_list.append(wer_asr)
        asr_lat_list.append(lat_asr)

        # 2. VAD + ASR Pipeline
        if run_vad:
            hyp_pipe, lat_pipe = run_pipeline(vad, detector, asr, s)
            hyp_pipe_clean = normalize_text(hyp_pipe)
            wer_pipe = jiwer.wer(ref_clean, hyp_pipe_clean) * 100
            pipe_wer_list.append(wer_pipe)
            pipe_lat_list.append(lat_pipe)

        print(f"  [{s['id']}] Dur: {s['duration_s']:.1f}s | ASR WER: {wer_asr:.1f}% ({lat_asr:.0f}ms)", end="", flush=True)
        if run_vad:
            print(f" | Pipe WER: {wer_pipe:.1f}% ({lat_pipe:.0f}ms)", flush=True)
        else:
            print(flush=True)

    total_dur_s = sum(durations)
    total_asr_time_s = sum(asr_lat_list) / 1000.0
    asr_rtf = total_asr_time_s / total_dur_s if total_dur_s > 0 else 0.0

    # Extrapolate full LibriSpeech 2,620 test-clean duration:
    extrapolated_hours = (TOTAL_LIBRISPEECH_CLEAN_SECONDS * asr_rtf) / 3600.0

    out = {
        "model": model_info["name"],
        "ram_mb": ram_mb,
        "asr_wer": np.mean(asr_wer_list),
        "asr_rtf": asr_rtf,
        "asr_latency": np.mean(asr_lat_list),
        "extrapolated_hours": extrapolated_hours,
        "extrapolated_mins": extrapolated_hours * 60,
    }

    if run_vad:
        total_pipe_time_s = sum(pipe_lat_list) / 1000.0
        pipe_rtf = total_pipe_time_s / total_dur_s if total_dur_s > 0 else 0.0
        pipe_hours = (TOTAL_LIBRISPEECH_CLEAN_SECONDS * pipe_rtf) / 3600.0
        out.update({
            "pipe_wer": np.mean(pipe_wer_list),
            "pipe_rtf": pipe_rtf,
            "pipe_latency": np.mean(pipe_lat_list),
            "pipe_extrapolated_hours": pipe_hours,
            "pipe_extrapolated_mins": pipe_hours * 60,
        })

    return out


def main():
    parser = argparse.ArgumentParser(description="LibriSpeech Benchmark")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples to evaluate")
    parser.add_argument("--model", type=str, default="all", choices=["all", "parakeet", "whisper-small", "distil-whisper", "whisper-base"], help="Specific model to test")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "asr", "pipeline"], help="Evaluation mode")
    parser.add_argument("--max-seconds", type=float, default=10800.0, help="Maximum total execution time in seconds (default: 10,800s = 3 hours)")
    args = parser.parse_args()

    device = get_device()
    print(f"Starting LibriSpeech Benchmark on {device.type.value.upper()}...")
    print(f"Total LibriSpeech test-clean reference: 2,620 samples (~5.4 hours)")
    print(f"Evaluation sample size: {args.samples} samples\n")

    samples = load_sample_stream(args.samples)

    run_vad = args.mode in ["all", "pipeline"]
    results = []

    models_to_run = MODELS_CONFIG
    if args.model != "all":
        target = args.model.lower().replace("-", " ")
        models_to_run = [m for m in MODELS_CONFIG if target in m["name"].lower()]

    for model_info in models_to_run:
        res = evaluate_model(model_info, samples, run_vad=run_vad, device=device)
        results.append(res)


    # Print Summary Table
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY & FULL-DATASET TIME EXTRAPOLATION")
    print("=" * 90)
    print(f"{'Model':<18} | {'ASR WER':<8} | {'ASR RTF':<8} | {'Latency':<8} | {'RAM (MB)':<8} | {'Full Libri (Est)':<16}")
    print("-" * 90)
    for r in results:
        est_str = f"{r['extrapolated_hours']:.2f}h ({r['extrapolated_mins']:.0f}m)"
        print(f"{r['model']:<18} | {r['asr_wer']:>6.1f}%  | {r['asr_rtf']:>6.2f}   | {r['asr_latency']:>5.0f}ms  | {r['ram_mb']:>6.0f}   | {est_str:<16}")

    if run_vad:
        print("\n" + "=" * 90)
        print("VAD + ASR PIPELINE COMPARISON (Clipping & Segmentation Impact)")
        print("=" * 90)
        print(f"{'Model':<18} | {'ASR Only WER':<13} | {'VAD+ASR WER':<12} | {'VAD Overhead RTF':<16} | {'Full Pipe (Est)':<16}")
        print("-" * 90)
        for r in results:
            pipe_est = f"{r['pipe_extrapolated_hours']:.2f}h ({r['pipe_extrapolated_mins']:.0f}m)"
            diff_rtf = r["pipe_rtf"] - r["asr_rtf"]
            print(f"{r['model']:<18} | {r['asr_wer']:>9.1f}%    | {r['pipe_wer']:>8.1f}%   | +{diff_rtf:>5.2f} RTF         | {pipe_est:<16}")

    # Write report to result_librispeech.md
    report_path = "result_librispeech.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# LibriSpeech `test.clean` Benchmark & Extrapolation\n\n")
        f.write(f"**Hardware Device:** {device.type.value.upper()}\n\n")
        f.write(f"**Sample Slice Tested:** {len(samples)} samples (~{sum(s['duration_s'] for s in samples):.1f}s audio)\n\n")
        f.write(f"**Full Dataset Projection:** 2,620 utterances (~5.4 hours audio)\n\n")
        
        f.write("## 1. Standalone ASR Performance & Estimated Full Runtime\n\n")
        f.write("| Model | Sample WER (%) | Sample RTF | Avg Latency (ms) | RAM (MB) | Full 2.6k Libri Est Time |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| **{r['model']}** | {r['asr_wer']:.1f}% | {r['asr_rtf']:.2f} | {r['asr_latency']:.0f}ms | ~{r['ram_mb']:.0f} | **{r['extrapolated_hours']:.2f} hours** (~{r['extrapolated_mins']:.0f} mins) |\n")

        if run_vad:
            f.write("\n## 2. VAD + ASR Pipeline Impact\n\n")
            f.write("| Model | Standalone WER (%) | VAD+ASR WER (%) | VAD Overhead (RTF) | Full Pipeline Est Time |\n")
            f.write("|---|---|---|---|---|\n")
            for r in results:
                diff_rtf = r["pipe_rtf"] - r["asr_rtf"]
                f.write(f"| **{r['model']}** | {r['asr_wer']:.1f}% | {r['pipe_wer']:.1f}% | +{diff_rtf:.2f} RTF | {r['pipe_extrapolated_hours']:.2f} hours |\n")

    print(f"\n[+] Detailed report written to {report_path}")


if __name__ == "__main__":
    main()
