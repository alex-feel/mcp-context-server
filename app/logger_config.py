
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
    """Initialize (or re-initialize) root logging once.

    Uses a formatter that strips package-directory prefixes
    (site-packages, dist-packages) from record.pathname and
    caps the result at a fixed number of trailing path segments.

    Also aligns the 'fastmcp' logger tree with the same level. When
    FASTMCP_LOG_ENABLED is true (default), FastMCP configures that tree at
    package-import time as NON-PROPAGATING with its own handler and level (from
    FASTMCP_LOG_LEVEL, default INFO), so configuring the root logger alone left
    FastMCP-internal INFO/WARNING lines printing regardless of LOG_LEVEL;
    aligning it here makes LOG_LEVEL the single effective verbosity control.
    When FASTMCP_LOG_ENABLED is false, FastMCP leaves the tree unconfigured
    (propagating, no handler), which would let FastMCP-internal records reach the
    root handler and print; this function then silences the tree (propagate off
    plus a NullHandler) to honor the documented "removes FastMCP-internal log
    output entirely" contract. FASTMCP_ENABLE_RICH_LOGGING (handler format) keeps
    its import-time effect. This governs only the in-process 'fastmcp' tree; the
    HTTP transports' uvicorn loggers are set by passing log_level to mcp.run().
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

    # Align the 'fastmcp' logger tree with LOG_LEVEL. Two import-time states are
    # possible, and they need opposite handling:
    #   * FASTMCP_LOG_ENABLED is true (default): FastMCP configured the tree as
    #     NON-PROPAGATING with its own handler, so it needs its own level and its
    #     handlers leveled too (a pre-leveled handler could otherwise re-widen the
    #     effective verbosity).
    #   * FASTMCP_LOG_ENABLED is false: FastMCP left the tree UNCONFIGURED
    #     (propagate=True, no handler), so a FastMCP-internal record at or above
    #     LOG_LEVEL would propagate to the root handler and print -- the opposite
    #     of the documented "removes FastMCP-internal log output entirely". Silence
    #     the tree to honor that contract: stop propagation and attach a
    #     NullHandler so no record escapes to root and no "no handlers" warning fires.
    # config_logger runs at server import AFTER FastMCP is imported, so an empty
    # handler list reliably means the disabled case rather than an unimported one.
    fastmcp_logger = logging.getLogger('fastmcp')
    fastmcp_logger.setLevel(numeric_level)
    if fastmcp_logger.handlers:
        for fastmcp_handler in fastmcp_logger.handlers:
            fastmcp_handler.setLevel(numeric_level)
    else:
        fastmcp_logger.propagate = False
        fastmcp_logger.addHandler(logging.NullHandler())

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
