#!/usr/bin/env python3
"""
scripts/record_sample.py - Run the full Mic -> VAD -> ASR pipeline.

This is the PRIMARY entry point for the first functional slice.

Usage:
    python scripts/record_sample.py
    python scripts/record_sample.py --model tiny --device cpu
    python scripts/record_sample.py --model small --language auto
    python scripts/record_sample.py --list-devices

Press Ctrl+C to stop recording.

Output:
    Raw transcripts are printed to console as you speak.
"""

import sys
import os

# Allow running from project root without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click
from loguru import logger

from voice_ai.utils.logging import setup_logging


@click.command()
@click.option(
    "--model",
    default="base",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v3-turbo"]),
    show_default=True,
    help="Whisper model size. 'tiny' is recommended for CPU. 'small' for better accuracy.",
)
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cuda", "cpu"]),
    show_default=True,
    help="Inference device. 'auto' detects CUDA and falls back to CPU.",
)
@click.option(
    "--language",
    default="en",
    show_default=True,
    help="Language code (e.g. en, fr, de) or 'auto' for auto-detection.",
)
@click.option(
    "--compute",
    default="auto",
    type=click.Choice(["auto", "int8", "float16", "float32"]),
    show_default=True,
    help="Computation precision for Whisper ('auto' uses int8 on CPU, float16 on GPU).",
)
@click.option(
    "--threshold",
    default=0.85,
    type=float,
    show_default=True,
    help="VAD speech probability threshold (0.0-1.0). Higher = less sensitive to background noise.",
)
@click.option(
    "--config",
    default=None,
    help="Path to config YAML. If set, all other flags are ignored.",
)
@click.option(
    "--list-devices",
    is_flag=True,
    default=False,
    help="List available audio input devices and exit.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def main(
    model: str,
    device: str,
    language: str,
    compute: str,
    threshold: float,
    config: str | None,
    list_devices: bool,
    debug: bool,
) -> None:
    """
    local-voice-ai: first functional slice.

    Microphone -> VAD -> ASR -> Raw Transcript (console output)
    """
    setup_logging(level="DEBUG" if debug else "INFO")

    if list_devices:
        from voice_ai.audio.recorder import MicrophoneRecorder
        MicrophoneRecorder.list_devices()
        return

    if config is not None:
        from voice_ai.pipeline.audio_pipeline import AudioPipeline
        pipeline = AudioPipeline.from_config(config)
    else:
        from voice_ai.inference.device import get_device
        from voice_ai.pipeline.audio_pipeline import AudioPipeline

        resolved_device = get_device(preference=device)

        pipeline = AudioPipeline(
            device=resolved_device,
            whisper_model=model,
            language=None if language == "auto" else language,
            compute_type=compute,
            vad_threshold=threshold,
        )

    pipeline.run()


if __name__ == "__main__":
    main()
