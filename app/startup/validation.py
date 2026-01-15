"""
Parameter validation utilities for MCP tool functions.

This module contains:
- JSON parameter deserialization (client compatibility workarounds)
- Date parameter validation and normalization
- Text truncation for display purposes
- Pool timeout validation for embedding operations

These utilities are used by MCP tool functions in app/tools/ to validate
and normalize input parameters before processing.
"""

import contextlib
import json
import logging
from datetime import date
from datetime import datetime as dt
from typing import cast

from fastmcp.exceptions import ToolError

from app.settings import get_settings
from app.types import JsonValue

logger = logging.getLogger(__name__)
settings = get_settings()


def validate_pool_timeout_for_embedding() -> None:
    """Validate POSTGRESQL_POOL_TIMEOUT_S is sufficient for embedding operations.

    Logs INFO-level warning if pool timeout is less than calculated minimum
    based on embedding timeout and retry configuration. This helps operators
    identify potential timeout issues during high-load semantic search operations.

    The calculation considers:
    - EMBEDDING_TIMEOUT_S * EMBEDDING_RETRY_MAX_ATTEMPTS (total timeout across retries)
    - Exponential backoff delays between retry attempts
    - 10% safety margin for network/processing overhead
    """
    if not settings.semantic_search.enabled:
        return  # Skip validation if semantic search is disabled

    # Calculate minimum required timeout
    # Formula: timeout * retries + exponential_backoff_delays + 10% margin
    timeout = settings.embedding.timeout_s
    retries = settings.embedding.retry_max_attempts
    base_delay = settings.embedding.retry_base_delay_s

    # Calculate total backoff delays (with 10% jitter estimate)
    # Backoff formula: base_delay * (2 ** attempt) for each retry
    total_backoff = 0.0
    for attempt in range(retries - 1):  # No delay after last attempt
        delay = base_delay * (2**attempt)
        jitter = delay * 0.1  # Max jitter estimate
        total_backoff += delay + jitter

    # Total maximum embedding time
    total_embedding_time = (timeout * retries) + total_backoff

    # Add 10% safety margin
    minimum_pool_timeout = total_embedding_time * 1.1

    pool_timeout = settings.storage.postgresql_pool_timeout_s

    if pool_timeout < minimum_pool_timeout:
        logger.info(
            f'POSTGRESQL_POOL_TIMEOUT_S ({pool_timeout}s) is below recommended minimum '
            f'({minimum_pool_timeout:.1f}s) for embedding operations. '
            f'Calculation: EMBEDDING_TIMEOUT_S ({timeout}s) * EMBEDDING_RETRY_MAX_ATTEMPTS ({retries}) '
            f'+ backoff ({total_backoff:.1f}s) + 10%% safety margin. '
            f'Consider increasing POSTGRESQL_POOL_TIMEOUT_S to avoid connection timeout errors '
            f'during high-load semantic search operations.',
        )


def deserialize_json_param(
    value: JsonValue | None,
) -> JsonValue | None:
    """Deserialize JSON string parameters if needed with enhanced safety checks.

    COMPATIBILITY NOTE: This function works around a known issue where some MCP clients
    (including Claude Code) send complex parameters as JSON strings instead of native
    Python objects. This is documented in multiple GitHub issues:
    - FastMCP #932: JSON Arguments Encapsulated as String Cause Validation Failure
    - Claude Code #5504: JSON objects converted to quoted strings
    - Claude Code #4192: Consecutive parameter calls fail
    - Claude Code #3084: Pydantic model parameters cause validation errors

    Enhanced to handle:
    - Double-encoding issues (JSON within JSON)
    - Single string values that should be treated as tags
    - Edge cases with special characters like forward slashes

    This function can be removed when the upstream issues are resolved.

    Args:
        value: The parameter value which might be a JSON string

    Returns:
        The deserialized value if it was a JSON string, or the original value
    """
    if isinstance(value, str):
        try:
            result = json.loads(value)
            # Check for double-encoding (JSON string within JSON)
            if isinstance(result, str):
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    # Try to decode again in case of double-encoding
                    result = json.loads(result)
            return cast(JsonValue | None, result)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON - check if it's meant to be a single tag
            if value.strip():
                # For tags parameter, a single string should become a list
                # This helps handle edge cases where a single tag is passed as string
                # The caller will need to handle this appropriately
                pass
            return value
    return value


def truncate_text(text: str | None, max_length: int = 150) -> tuple[str | None, bool]:
    """Truncate text at word boundary when possible.

    Args:
        text: The text to truncate
        max_length: Maximum character length (default: 150)

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
    """Validate and normalize date parameter for database filtering.

    Accepts ISO 8601 format dates and returns the validated string for database use.
    Both date-only (YYYY-MM-DD) and datetime (YYYY-MM-DDTHH:MM:SS) formats are supported.
    Timezone suffixes (Z or +HH:MM) are also accepted.

    For end_date with date-only format: automatically expands to end-of-day (T23:59:59)
    to match user expectations. This follows Elasticsearch's precedent where missing time
    components are replaced with max values for 'lte' (less-than-or-equal) operations.
    See: https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-range-query

    Args:
        date_str: ISO 8601 date string or None
        param_name: Parameter name for error messages (e.g., 'start_date', 'end_date')

    Returns:
        Validated date string (possibly expanded for end_date) or None if input was None

    Raises:
        ToolError: If date format is invalid
    """
    if date_str is None:
        return None

    # Detect date-only format by checking for absence of time separators
    # Date-only: '2025-11-29' (no 'T' or space separator)
    # Datetime: '2025-11-29T10:00:00' or '2025-11-29 10:00:00'
    is_date_only = 'T' not in date_str and ' ' not in date_str

    # Validate the date string
    try:
        if is_date_only:
            # Parse as date-only format (YYYY-MM-DD)
            date.fromisoformat(date_str)
        else:
            # Parse as datetime format (with optional timezone)
            # Python 3.11+ handles 'Z' natively
            dt.fromisoformat(date_str)
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
        date_str = f'{date_str}T23:59:59.999999'

    return date_str


def validate_date_range(start_date: str | None, end_date: str | None) -> None:
    """Validate that start_date is not after end_date.

    Args:
        start_date: Validated start date string
        end_date: Validated end date string

    Raises:
        ToolError: If start_date is after end_date
    """

    def _parse_and_normalize(date_str: str) -> dt:
        """Parse date string and normalize to naive datetime for comparison.

        Handles all ISO 8601 formats: date-only, datetime, datetime+tz, datetime+Z.
        Strips timezone info to allow comparison between mixed formats.

        Returns:
            Naive datetime object for comparison purposes.
        """
        # Handle Z suffix - replace with +00:00 for fromisoformat
        normalized = date_str.replace('Z', '+00:00') if date_str.endswith('Z') else date_str

        try:
            parsed = dt.fromisoformat(normalized)
            # Strip timezone info for comparison (we just need relative ordering)
            return parsed.replace(tzinfo=None)
        except ValueError:
            # Date-only format - convert to datetime for comparison
            return dt.combine(date.fromisoformat(date_str), dt.min.time())

    if start_date and end_date:
        start_dt = _parse_and_normalize(start_date)
        end_dt = _parse_and_normalize(end_date)

        if start_dt > end_dt:
            raise ToolError(
                f'Invalid date range: start_date ({start_date}) is after end_date ({end_date})',
            )


__all__ = [
    'validate_pool_timeout_for_embedding',
    'deserialize_json_param',
    'truncate_text',
    'validate_date_param',
    'validate_date_range',
]
