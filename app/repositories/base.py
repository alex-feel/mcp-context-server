"""
Base repository class for database operations.

This module provides the base class that all repositories inherit from,
ensuring consistent patterns and proper connection management.
"""

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
