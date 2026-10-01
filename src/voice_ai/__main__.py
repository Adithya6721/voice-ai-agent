"""
voice_ai.__main__ — CLI entry point.

Run with:
    python -m voice_ai
"""

import click
from voice_ai.utils.logging import setup_logging


@click.group()
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
def main(debug: bool) -> None:
    """local-voice-ai: privacy-first, fully local voice assistant."""
    level = "DEBUG" if debug else "INFO"
    setup_logging(level=level)


@main.command()
@click.option("--config", default="config/config.yaml", help="Path to config file.")
def run(config: str) -> None:
    """Start the voice pipeline (mic → VAD → ASR → transcript)."""
    from voice_ai.pipeline.audio_pipeline import AudioPipeline

    pipeline = AudioPipeline.from_config(config)
    pipeline.run()


if __name__ == "__main__":
    main()
