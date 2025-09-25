"""
Base repository class for database operations.

This module provides the base class that all repositories inherit from,
ensuring consistent patterns and proper connection management.
"""

from app.db_manager import DatabaseConnectionManager


class BaseRepository:
    """Base repository class for all database repositories.

    Provides common functionality and ensures all repositories follow
    the same patterns for database access.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        """Initialize repository with database connection manager.

        Args:
            db_manager: Database connection manager for executing operations
        """
        self.db_manager = db_manager
