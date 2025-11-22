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
