"""
Base repository class for database operations.

This module provides the base class that all repositories inherit from,
ensuring consistent patterns and proper connection management.
"""

import asyncio
import datetime
import sqlite3
from collections.abc import Callable
from typing import TypeVar

from app.backends._executor import run_in_executor_uninterruptible
from app.backends.base import StorageBackend

T = TypeVar('T')


def canonical_timestamp(value: object) -> str | None:
    """Render a created_at/updated_at value as a canonical UTC ISO-8601 string.

    Entry-returning tools MUST emit the SAME wire format for timestamps on both
    backends. SQLite stores them as TEXT ("YYYY-MM-DD HH:MM:SS", UTC, from
    CURRENT_TIMESTAMP) and returns the raw string; PostgreSQL returns datetime
    objects (asyncpg), which otherwise serialize as "YYYY-MM-DDTHH:MM:SS+00:00".
    Normalize both to "YYYY-MM-DDTHH:MM:SSZ" -- the exact form get_statistics
    already emits -- so every backend and tool is byte-for-byte consistent.

    ``None`` is preserved (schema-legal NULL); an unparseable string is returned
    unchanged. Idempotent: re-applying to an already-canonical value is a no-op.

    Returns:
        The canonical "YYYY-MM-DDTHH:MM:SSZ" UTC string, or ``None`` when the input
        is ``None``, or the original string when it cannot be parsed as a timestamp.
    """
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return text
        # SQLite stores a space separator; accept that and any ISO-8601 variant.
        candidate = text.replace(' ', 'T', 1) if 'T' not in text else text
        candidate = candidate.replace('Z', '+00:00')
        try:
            dt = datetime.datetime.fromisoformat(candidate)
        except ValueError:
            return text
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.UTC)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


class BaseRepository:
    """Base repository class for all database repositories.

    Provides common functionality and ensures all repositories follow
    the same patterns for database access.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize repository with storage backend.

        Args:
            backend: Storage backend for executing database operations
        """
        self.backend = backend

    @staticmethod
    async def _run_sqlite_txn(
        closure: Callable[[sqlite3.Connection], T],
        conn: sqlite3.Connection,
    ) -> T:
        """Run a synchronous SQLite closure on the executor thread pool.

        Repository methods handed a ``TransactionContext`` must not call their
        sqlite3 closures directly on the event loop: the C-level calls block
        it -- a cross-process lock holder makes ``cursor.execute`` busy-wait
        inside SQLite for up to the resolved busy timeout, freezing every
        concurrent request and the ``/health`` endpoint, and even uncontended
        large writes (image BLOB decode + insert, FTS tokenization) stall the
        loop for their full duration. Offloading matches the write-queue
        path, which runs identical closures via ``run_in_executor``; the
        writer connection is created with ``check_same_thread=False`` and the
        transaction's writer lock serializes access, so one-at-a-time
        executor use of the connection is safe.

        The executor future is shielded and, on any unwind, drained before
        the exception propagates. Cancelling the awaiting task cannot cancel
        a closure that is already executing in the worker thread; without the
        drain the CancelledError would reach ``begin_transaction``'s rollback
        while the closure keeps issuing statements on the same connection --
        anything the zombie executed after that rollback would open a fresh
        implicit transaction that the NEXT write on the pooled writer
        connection silently commits. Draining guarantees the closure has
        fully finished before the rollback runs and before the writer lock is
        released. The drain deliberately never cancels the wrapper future:
        cancelling an ``asyncio`` future succeeds immediately regardless of
        the running callable, which would end the join early and resurrect
        the race.

        Args:
            closure: Synchronous callable executing statements on ``conn``.
            conn: The transaction's SQLite connection.

        Returns:
            Whatever the closure returns.
        """
        loop = asyncio.get_running_loop()
        return await run_in_executor_uninterruptible(loop, closure, conn)

    def _placeholder(self, position: int) -> str:
        """Generate SQL parameter placeholder for the given position.

        SQLite uses '?' for all parameters, while PostgreSQL uses '$1, $2, $3' notation.

        Args:
            position: 1-based parameter position

        Returns:
            SQL placeholder string ('?' for SQLite, '$N' for PostgreSQL)

        Example:
            # SQLite
            _placeholder(1)  # Returns '?'
            _placeholder(5)  # Returns '?'

            # PostgreSQL
            _placeholder(1)  # Returns '$1'
            _placeholder(5)  # Returns '$5'
        """
        if self.backend.backend_type == 'sqlite':
            return '?'
        return f'${position}'

    def _placeholders(self, count: int, start: int = 1) -> str:
        """Generate comma-separated SQL parameter placeholders.

        Args:
            count: Number of placeholders to generate
            start: Starting position (1-based, default: 1)

        Returns:
            Comma-separated placeholder string

        Example:
            # SQLite
            _placeholders(3)      # Returns '?, ?, ?'
            _placeholders(3, 5)   # Returns '?, ?, ?'

            # PostgreSQL
            _placeholders(3)      # Returns '$1, $2, $3'
            _placeholders(3, 5)   # Returns '$5, $6, $7'
        """
        if self.backend.backend_type == 'sqlite':
            return ', '.join(['?'] * count)
        return ', '.join([f'${i}' for i in range(start, start + count)])

    def _json_extract(self, column: str, path: str) -> str:
        """Generate SQL expression for extracting JSON field.

        SQLite uses json_extract(column, '$.path') while PostgreSQL uses column->>'path'.

        Args:
            column: Column name containing JSON data
            path: JSON path to extract (without '$.' prefix)

        Returns:
            SQL expression for JSON extraction

        Example:
            # SQLite
            _json_extract('metadata', 'status')
            # Returns: "json_extract(metadata, '$.status')"

            # PostgreSQL
            _json_extract('metadata', 'status')
            # Returns: "metadata->>'status'"
        """
        if self.backend.backend_type == 'sqlite':
            return f"json_extract({column}, '$.{path}')"
        return f'{column}->>{path!r}'

    @staticmethod
    def _parse_date_for_postgresql(date_str: str | None) -> datetime.datetime | None:
        """Parse ISO 8601 date string to Python datetime object for PostgreSQL.

        asyncpg requires Python datetime objects for TIMESTAMPTZ parameters,
        not string representations. This method converts ISO 8601 date strings
        to Python datetime objects.

        Naive datetime (without timezone) is interpreted as UTC to match
        industry standards (Elasticsearch, MongoDB, DynamoDB, Firestore).
        This ensures deterministic behavior regardless of server timezone.

        Args:
            date_str: ISO 8601 date string (e.g., '2025-11-29', '2025-11-29T10:00:00',
                     '2025-11-29T10:00:00Z', '2025-11-29T10:00:00+02:00') or None

        Returns:
            datetime object with timezone info or None if input is None

        Example:
            _parse_date_for_postgresql('2025-11-29')
            # Returns: datetime(2025, 11, 29, 0, 0, 0, tzinfo=timezone.utc)

            _parse_date_for_postgresql('2025-11-29T10:00:00')
            # Returns: datetime(2025, 11, 29, 10, 0, 0, tzinfo=timezone.utc)  # UTC assumed

            _parse_date_for_postgresql('2025-11-29T10:00:00Z')
            # Returns: datetime(2025, 11, 29, 10, 0, 0, tzinfo=timezone.utc)

            _parse_date_for_postgresql('2025-11-29T10:00:00+02:00')
            # Returns: datetime(2025, 11, 29, 10, 0, 0, tzinfo=timezone(+02:00))
        """
        if date_str is None:
            return None

        # Handle Z suffix for UTC (Python 3.11+ handles this natively)
        normalized = date_str.replace('Z', '+00:00') if date_str.endswith('Z') else date_str

        # Check if it's a date-only string (no 'T' separator)
        # Date-only strings should be converted to datetime at start of day UTC
        is_date_only = 'T' not in date_str

        if is_date_only:
            # Parse as date-only and convert to datetime at start of day UTC
            try:
                parsed_date = datetime.date.fromisoformat(date_str)
                return datetime.datetime(
                    parsed_date.year,
                    parsed_date.month,
                    parsed_date.day,
                    tzinfo=datetime.UTC,
                )
            except ValueError:
                pass
        else:
            # Parse as full datetime
            try:
                parsed = datetime.datetime.fromisoformat(normalized)
                # Naive datetime - interpret as UTC (industry standard)
                # This matches Elasticsearch, MongoDB, DynamoDB, Firestore behavior
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=datetime.UTC)
                return parsed
            except ValueError:
                pass

        # If all parsing fails, return None
        # This should not happen if validation was done at the tool level
        return None
