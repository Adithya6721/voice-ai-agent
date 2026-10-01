#!/usr/bin/env python3
"""
scripts/download_models.py — Pre-download all model weights.

Run this once before first use to download models to local cache.
All models are open-source and downloaded from HuggingFace Hub.

Downloads:
  - Silero VAD (~2 MB)  — via torch.hub or silero-vad package
  - Whisper tiny        (~39 MB)
  - Whisper base        (~74 MB)
  - Whisper small       (~244 MB)   ← default, always downloaded

After download, all inference runs offline — no internet needed.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from voice_ai.utils.logging import setup_logging

console = Console()


@click.command()
@click.option(
    "--models",
    default="vad,whisper-tiny",
    help="Comma-separated list: vad, whisper-tiny, whisper-base, whisper-small, whisper-medium",
)
@click.option("--all", "download_all", is_flag=True, default=False, help="Download all models.")
def main(models: str, download_all: bool) -> None:
    """Download model weights for offline use."""
    setup_logging(level="INFO")

    if download_all:
        to_download = ["vad", "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"]
    else:
        to_download = [m.strip() for m in models.split(",")]

    console.print(f"\n[bold cyan]Downloading: {', '.join(to_download)}[/bold cyan]\n")

    for model_id in to_download:
        if model_id == "vad":
            _download_silero_vad()
        elif model_id.startswith("whisper-"):
            size = model_id.replace("whisper-", "")
            _download_whisper(size)
        else:
            console.print(f"[red]Unknown model: {model_id}[/red]")

    console.print("\n[bold green]✓ All models downloaded. Ready for offline use.[/bold green]")


def _download_silero_vad() -> None:
    """Download Silero VAD weights."""
    console.print("[yellow]→ Silero VAD[/yellow]")
    try:
        from silero_vad import load_silero_vad  # type: ignore[import-untyped]
        model = load_silero_vad()
        console.print("  [green]✓ Silero VAD downloaded via silero-vad package.[/green]")
    except ImportError:
        import torch
        console.print("  [dim]Falling back to torch.hub…[/dim]")
        torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        console.print("  [green]✓ Silero VAD downloaded via torch.hub.[/green]")


def _download_whisper(size: str) -> None:
    """Download a Whisper model via faster-whisper (HuggingFace)."""
    console.print(f"[yellow]→ Whisper {size}[/yellow]")
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

    # Loading the model triggers the download
    WhisperModel(
        model_size_or_path=size,
        device="cpu",
        compute_type="int8",
    )
    console.print(f"  [green]✓ Whisper {size} downloaded.[/green]")


if __name__ == "__main__":
    main()
