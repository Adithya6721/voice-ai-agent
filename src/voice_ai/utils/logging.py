"""
voice_ai.utils.logging — centralised logging setup using loguru + rich.
"""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(level: str = "INFO", rich_format: bool = True) -> None:
    """
    Configure loguru with a rich console handler.

    Args:
        level:       Minimum log level (DEBUG, INFO, WARNING, ERROR).
        rich_format: If True, use colourful rich formatting.
    """
    logger.remove()  # Remove default handler

    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=rich_format,
        backtrace=True,
        diagnose=True,
    )
