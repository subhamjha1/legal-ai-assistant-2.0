"""
Structured logging configuration.

Why this exists:
In a production ingestion pipeline, silent failures on page 47 of a 300-page
judgment are the difference between a system you can trust and one you can't.
Every service logs through this module so log format, level, and destination
are controlled in one place.
"""
import logging
import sys

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    root = logging.getLogger()
    if root.handlers:
        # Avoid duplicate handlers on reload (e.g. uvicorn --reload)
        return

    root.setLevel(settings.log_level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
