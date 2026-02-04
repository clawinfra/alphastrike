"""Shared logging configuration for CLI commands."""

from __future__ import annotations

import logging


def setup_logging(log_level: str, silence_third_party: bool = True) -> None:
    """
    Configure logging with structured output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        silence_third_party: Whether to silence noisy third-party libraries
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(handler)

    if silence_third_party:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
