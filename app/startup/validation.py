"""
Parameter validation utilities for MCP tool functions.

This module contains:
- Date parameter validation and normalization
- Text truncation for display purposes
- Connection-pool acquire-timeout validation

These utilities are used by MCP tool functions in app/tools/ to validate
and normalize input parameters before processing.
"""

import logging
from datetime import UTC
from datetime import date
from datetime import datetime as dt

from fastmcp.exceptions import ToolError

from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Hostname fragment identifying the Supabase / Supavisor pooler endpoint.
# Both session mode (port 5432) and transaction mode (port 6543) share this
# host; only session mode (5432) enforces the per-session client cap that
# produces "MaxClientsInSessionMode". This (and the port below) is a fixed
# external-service protocol fact used only inside the detection predicate --
# like the 'SHOW POOL_VERSION' probe and error code 42704 used for Pgpool-II
# detection -- NOT an operator setting. The configurable advisory threshold
# lives in StorageSettings.postgresql_session_pooler_max_clients.
_SUPABASE_POOLER_HOST_FRAGMENT = 'pooler.supabase.com'

# Supavisor session-mode port. Transaction mode (6543) has no per-session cap.
_SUPABASE_SESSION_MODE_PORT = 5432


def validate_pool_acquire_timeout() -> None:
    """Validate POSTGRESQL_POOL_TIMEOUT_S is sufficient for connection acquisition.

    POSTGRESQL_POOL_TIMEOUT_S is asyncpg's pool acquire-wait timeout: how long a
    caller waits for a free pooled connection when every connection is busy. A
    connection is held only for the duration of in-transaction database work
    (bounded by POSTGRESQL_COMMAND_TIMEOUT_S), never for embedding generation,
    which runs before any connection is acquired. The recommended floor is
    therefore a small multiple of the command timeout, not embedding wall-time.

    Logs an INFO-level advisory when the configured acquire timeout is below that
    floor, directing operators to the levers that actually govern behavior under
    embedding load: POSTGRESQL_POOL_MAX (pool size) and POSTGRESQL_COMMAND_TIMEOUT_S.
    """
    command_timeout = settings.storage.postgresql_command_timeout_s
    pool_timeout = settings.storage.postgresql_pool_timeout_s

    # A waiter blocks at most until in-flight database work frees a connection;
    # allow headroom for several queued waiters beyond a single command's ceiling.
    minimum_pool_timeout = max(60.0, command_timeout * 2)

    if pool_timeout < minimum_pool_timeout:
        logger.info(
            f'POSTGRESQL_POOL_TIMEOUT_S ({pool_timeout}s) is below the recommended '
            f'minimum ({minimum_pool_timeout:.1f}s) for connection acquisition '
            f'(2x POSTGRESQL_COMMAND_TIMEOUT_S={command_timeout}s). This is how long '
            f'a caller waits for a free pooled connection. If acquire timeouts occur '
            f'under load, raise POSTGRESQL_POOL_MAX or lower per-query time via '
            f'POSTGRESQL_COMMAND_TIMEOUT_S; embedding generation runs outside the '
            f'pool and does not affect this timeout.',
        )


def is_supabase_session_pooler(host: str, port: int) -> bool:
    """Return True when (host, port) identifies a Supabase Session Pooler.

    The Supabase / Supavisor pooler host contains ``pooler.supabase.com`` and
    serves Session mode on port 5432 (the mode that enforces a per-session
    client cap and raises ``MaxClientsInSessionMode`` when exceeded).
    Transaction mode (port 6543) shares the host but has no per-session cap and
    is intentionally NOT flagged. Non-Supabase hosts are never flagged.

    Args:
        host: Lowercased hostname.
        port: TCP port.

    Returns:
        bool: True only for Supabase Session Pooler (host fragment + port 5432).
    """
    return _SUPABASE_POOLER_HOST_FRAGMENT in host and port == _SUPABASE_SESSION_MODE_PORT


def truncate_text(text: str | None, max_length: int = 300) -> tuple[str | None, bool]:
    """Truncate text at word boundary when possible.

    Args:
        text: The text to truncate
        max_length: Maximum character length (default: 300)

    Returns:
        tuple: (truncated_text, is_truncated)
    """
    if not text or len(text) <= max_length:
        return text, False

    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')

    if last_space > max_length * 0.7:  # Only use word boundary if it's not too short
        truncated = truncated[:last_space]

    return truncated + '...', True


def validate_date_param(date_str: str | None, param_name: str) -> str | None:
    """Validate and canonicalize a date parameter for database filtering.

    Accepts every ISO 8601 form Python's ``fromisoformat`` parses (extended
    'YYYY-MM-DD' / 'YYYY-MM-DDTHH:MM:SS', the space-separated datetime, the
    basic/compact form '20250601', week dates '2025-W23-1', and timezone
    suffixes Z or +HH:MM) and returns the value RE-SERIALIZED in the extended
    ISO 8601 form. Canonicalization is load-bearing for cross-backend parity:
    the raw accepted superset is wider than what the storage layer parses
    (SQLite's ``datetime()`` returns NULL for compact and week-date forms,
    silently filtering out every row, and the PostgreSQL parameter parser keys
    its branch on the 'T' separator), so both backends must receive one
    normalized representation.

    For end_date with date-only format: automatically expands to end-of-day (T23:59:59)
    to match user expectations. This follows Elasticsearch's precedent where missing time
    components are replaced with max values for 'lte' (less-than-or-equal) operations.
    See: https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-range-query

    Args:
        date_str: ISO 8601 date string or None
        param_name: Parameter name for error messages (e.g., 'start_date', 'end_date')

    Returns:
        The canonical extended-ISO date string (expanded to end-of-day for a
        date-only end_date) or None if input was None

    Raises:
        ToolError: If date format is invalid
    """
    if date_str is None:
        return None

    # Detect date-only format by checking for absence of time separators
    # Date-only: '2025-11-29' or '20250601' (no 'T' or space separator)
    # Datetime: '2025-11-29T10:00:00' or '2025-11-29 10:00:00'
    is_date_only = 'T' not in date_str and ' ' not in date_str

    # Validate AND canonicalize: isoformat() re-serializes the parsed value in
    # the extended form ('2025-06-01' / '2025-11-29T10:00:00[+HH:MM]'), so a
    # compact, week-date, or space-separated input reaches the backends in the
    # one representation both parse.
    # Python 3.11+ handles the 'Z' suffix natively on the datetime branch; an
    # aware input keeps its offset (serialized as +HH:MM).
    try:
        canonical = (
            date.fromisoformat(date_str).isoformat()
            if is_date_only
            else dt.fromisoformat(date_str).isoformat()
        )
    except ValueError:
        raise ToolError(
            f'Invalid {param_name} format: "{date_str}". '
            f'Use ISO 8601 format (e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ) from None

    # For end_date with date-only format: expand to end-of-day with microsecond precision
    # This follows Elasticsearch precedent where missing time components are replaced
    # with max values for 'lte' operations, matching user expectations that
    # end_date='2025-11-29' should include ALL entries on November 29th.
    #
    # Uses T23:59:59.999999 (microsecond precision) for PostgreSQL compatibility:
    # PostgreSQL's CURRENT_TIMESTAMP stores microseconds (e.g., 23:59:59.500000),
    # so T23:59:59 (microsecond=0) would exclude entries at 23:59:59.xxx.
    # SQLite is unaffected as CURRENT_TIMESTAMP stores second precision only.
    if param_name == 'end_date' and is_date_only:
        canonical = f'{canonical}T23:59:59.999999'

    return canonical


def validate_date_range(start_date: str | None, end_date: str | None) -> None:
    """Validate that start_date is not after end_date.

    Args:
        start_date: Validated start date string
        end_date: Validated end date string

    Raises:
        ToolError: If start_date is after end_date
    """

    def _parse_and_normalize(date_str: str) -> dt:
        """Parse date string and normalize to a UTC instant for comparison.

        Handles all ISO 8601 formats: date-only, datetime, datetime+tz, datetime+Z.
        Aware datetimes are CONVERTED to UTC and naive ones interpreted AS UTC --
        matching the storage layer, which compares absolute UTC instants on both
        backends (SQLite's ``datetime()`` converts offsets to UTC; the PostgreSQL
        parameter parser applies the same naive-as-UTC convention). Stripping the
        offset instead would compare wall-clock values and contradict the filter
        semantics for mixed-offset ranges in both directions (rejecting valid
        ranges and accepting inverted ones).

        Returns:
            Timezone-aware UTC datetime object for comparison purposes.
        """
        # Handle Z suffix - replace with +00:00 for fromisoformat
        normalized = date_str.replace('Z', '+00:00') if date_str.endswith('Z') else date_str

        try:
            parsed = dt.fromisoformat(normalized)
        except ValueError:
            # Date-only format - convert to datetime for comparison
            return dt.combine(date.fromisoformat(date_str), dt.min.time(), tzinfo=UTC)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    if start_date and end_date:
        start_dt = _parse_and_normalize(start_date)
        end_dt = _parse_and_normalize(end_date)

        if start_dt > end_dt:
            raise ToolError(
                f'Invalid date range: start_date ({start_date}) is after end_date ({end_date})',
            )


__all__ = [
    'validate_pool_acquire_timeout',
    'is_supabase_session_pooler',
    'truncate_text',
    'validate_date_param',
    'validate_date_range',
]
