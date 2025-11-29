"""
Base repository class for database operations.

This module provides the base class that all repositories inherit from,
ensuring consistent patterns and proper connection management.
"""

import datetime

from app.backends.base import StorageBackend


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
