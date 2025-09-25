"""
Repository pattern implementation for database operations.

This module provides clean separation of concerns by isolating all database operations
into focused repository classes
"""

# Type imports
from app.db_manager import DatabaseConnectionManager
from app.repositories.context_repository import ContextRepository
from app.repositories.image_repository import ImageRepository
from app.repositories.statistics_repository import StatisticsRepository
from app.repositories.tag_repository import TagRepository


class RepositoryContainer:
    """Container for all repository instances providing dependency injection.

    This class manages repository instances and provides them to the server layer,
    ensuring proper separation of concerns and testability.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        """Initialize repository container with database manager.

        Args:
            db_manager: Database connection manager for all repositories to use
        """
        self.context = ContextRepository(db_manager)
        self.tags = TagRepository(db_manager)
        self.images = ImageRepository(db_manager)
        self.statistics = StatisticsRepository(db_manager)


__all__ = [
    'ContextRepository',
    'ImageRepository',
    'StatisticsRepository',
    'TagRepository',
    'RepositoryContainer',
]
