"""
core/logging_config.py — Logging setup for the PrivaRepo API.

Does not touch the backend's own module-level loggers (rag_pipeline,
vector_store, llm_interface, etc. each call logging.getLogger(__name__)
already) — this just configures the root handler/format/level once, so
those existing loggers start producing output in a consistent format.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging. Safe to call once at API startup."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        # Already configured (e.g. re-import during reload) — don't duplicate handlers.
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)

    # Quiet down noisy third-party loggers that PrivaRepo's own CLI logs
    # already showed flooding output at DEBUG (httpcore/httpx per-request
    # tracing, chromadb telemetry).
    for noisy in ("httpx", "httpcore", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
