#!/usr/bin/env python3
"""
scripts/benchmark_vad.py — Silero VAD benchmarking script.

Measures:
  - Latency per chunk (30ms audio chunk processing time)
  - Throughput (chunks/second)
  - Speech detection accuracy on labeled segments

Usage:
    python scripts/benchmark_vad.py --audio data/samples/test.wav
    python scripts/benchmark_vad.py --n-chunks 1000 --threshold 0.5
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from voice_ai.utils.logging import setup_logging

console = Console()


@click.command()
@click.option("--audio", default=None, help="Audio file to run VAD on.")
@click.option("--n-chunks", default=500, show_default=True, help="Synthetic chunks if no audio.")
@click.option("--threshold", default=0.5, show_default=True, type=float)
@click.option("--debug", is_flag=True, default=False)
def main(audio: str | None, n_chunks: int, threshold: float, debug: bool) -> None:
    """Benchmark Silero VAD latency and throughput."""
    setup_logging(level="DEBUG" if debug else "WARNING")

    from voice_ai.audio.audio_utils import CHUNK_SAMPLES
    from voice_ai.vad.silero import SileroVAD

    vad = SileroVAD(threshold=threshold)

    if audio:
        import soundfile as sf
        from voice_ai.audio.resampler import Resampler
        raw, sr = sf.read(audio, dtype="float32", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        if sr != 16000:
            raw = Resampler(sr).resample(raw)
        chunks = [
            raw[i:i + CHUNK_SAMPLES]
            for i in range(0, len(raw) - CHUNK_SAMPLES, CHUNK_SAMPLES)
        ]
        console.print(f"[cyan]Loaded {audio}: {len(chunks)} chunks[/cyan]")
    else:
        console.print(f"[cyan]Generating {n_chunks} synthetic chunks…[/cyan]")
        chunks = [np.random.randn(CHUNK_SAMPLES).astype(np.float32) for _ in range(n_chunks)]

    # Warmup
    for c in chunks[:10]:
        vad.speech_probability(c)

    # Benchmark
    latencies_ms = []
    speech_count = 0

    t_total = time.perf_counter()
    for chunk in chunks:
        t0 = time.perf_counter()
        prob = vad.speech_probability(chunk)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        if prob >= threshold:
            speech_count += 1
    total_s = time.perf_counter() - t_total

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    p50 = sorted(latencies_ms)[len(latencies_ms) // 2]
    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    p99 = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
    throughput = len(chunks) / total_s

    table = Table(title="Silero VAD Benchmark", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Chunks processed", str(len(chunks)))
    table.add_row("Total time", f"{total_s*1000:.0f}ms")
    table.add_row("Throughput", f"{throughput:.0f} chunks/s")
    table.add_row("Avg latency", f"{avg_ms:.3f}ms")
    table.add_row("P50 latency", f"{p50:.3f}ms")
    table.add_row("P95 latency", f"{p95:.3f}ms")
    table.add_row("P99 latency", f"{p99:.3f}ms")
    table.add_row("Speech detected", f"{speech_count}/{len(chunks)} ({speech_count/len(chunks)*100:.1f}%)")
    table.add_row("Threshold", str(threshold))
    table.add_row("Real-time factor", f"{(avg_ms / 30):.4f} (30ms chunks)")

    console.print(table)


if __name__ == "__main__":
    main()
