# app/core/logger_config.py
from __future__ import annotations

import logging
import os
from typing import override


def config_logger(log_level: str) -> None:
    """
    Initialise (or re‑initialise) root logging *once*,
    using a formatter that keeps only the last 5 segments of record.pathname.
    """
    # Convert level string to numeric constant
    numeric_level = getattr(logging, log_level.upper(), logging.ERROR)

    # Build the short‑path formatter
    class _ShortPath(logging.Formatter):
        @override
        def format(self, record: logging.LogRecord) -> str:
            """Format log record with shortened pathname."""
            parts = record.pathname.split(os.sep)
            record.shortpathname = os.sep.join(parts[-5:])  # last 5 parts
            return super().format(record)

    fmt = _ShortPath(
        '[%(asctime)s] [%(process)d] [%(levelname)s] [%(shortpathname)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z',
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if root.handlers:
        # Logger already configured, just update level/format
        for h in root.handlers:
            h.setLevel(numeric_level)
            h.setFormatter(fmt)
        return

    # First time: attach a single StreamHandler
    h = logging.StreamHandler()
    h.setLevel(numeric_level)
    h.setFormatter(fmt)
    root.addHandler(h)
