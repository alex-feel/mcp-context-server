from __future__ import annotations

import logging
import os
from typing import override

# Markers for installed-package directories (with trailing separator).
# site-packages: standard Python (pip, virtualenvs, Docker non-editable installs)
# dist-packages: Debian/Ubuntu system Python (apt-installed packages)
_PACKAGE_DIR_MARKERS = (
    'site-packages' + os.sep,
    'dist-packages' + os.sep,
)

_MAX_PATH_SEGMENTS = 5


def config_logger(log_level: str) -> None:
    """Initialise (or re-initialise) root logging once.

    Uses a formatter that strips package-directory prefixes
    (site-packages, dist-packages) from record.pathname and
    caps the result at a fixed number of trailing path segments.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.ERROR)

    class _ShortPath(logging.Formatter):
        @override
        def format(self, record: logging.LogRecord) -> str:
            """Format log record with shortened pathname."""
            path = record.pathname

            # Strip everything up to and including the package-directory marker.
            for marker in _PACKAGE_DIR_MARKERS:
                idx = path.find(marker)
                if idx >= 0:
                    after = path[idx + len(marker):]
                    n = after.count(os.sep)
                    record.shortpathname = (
                        after if n < _MAX_PATH_SEGMENTS
                        else os.sep.join(after.split(os.sep)[-_MAX_PATH_SEGMENTS:])
                    )
                    return super().format(record)

            # No package-directory marker: cap at N trailing segments.
            parts = path.split(os.sep)
            record.shortpathname = (
                os.sep.join(parts[-_MAX_PATH_SEGMENTS:])
                if len(parts) > _MAX_PATH_SEGMENTS
                else path
            )
            return super().format(record)

    fmt = _ShortPath(
        '[%(asctime)s] [%(process)d] [%(levelname)s] '
        '[%(shortpathname)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z',
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if root.handlers:
        for h in root.handlers:
            h.setLevel(numeric_level)
            h.setFormatter(fmt)
        return

    h = logging.StreamHandler()
    h.setLevel(numeric_level)
    h.setFormatter(fmt)
    root.addHandler(h)
