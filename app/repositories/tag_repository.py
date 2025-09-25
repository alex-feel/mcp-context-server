"""
Tag repository for managing context entry tags.

This module handles all database operations related to tags,
including storage and retrieval of normalized tags.
"""

import sqlite3

from app.db_manager import DatabaseConnectionManager
from app.repositories.base import BaseRepository


class TagRepository(BaseRepository):
    """Repository for tag operations.

    Handles storage and retrieval of normalized tags associated
    with context entries.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        """Initialize tag repository.

        Args:
            db_manager: Database connection manager for executing operations
        """
        super().__init__(db_manager)

    async def store_tags(self, context_id: int, tags: list[str]) -> None:
        """Store normalized tags for a context entry.

        Args:
            context_id: ID of the context entry
            tags: List of tags to store (will be normalized)
        """
        def _store_tags(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            for tag in tags:
                tag = tag.strip().lower()
                if tag:
                    cursor.execute(
                        'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                        (context_id, tag),
                    )

        await self.db_manager.execute_write(_store_tags)

    async def get_tags_for_context(self, context_id: int) -> list[str]:
        """Get all tags for a specific context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            List of tags associated with the context entry
        """
        def _get_tags(conn: sqlite3.Connection) -> list[str]:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY tag',
                (context_id,),
            )
            return [row['tag'] for row in cursor.fetchall()]

        return await self.db_manager.execute_read(_get_tags)

    async def get_tags_for_contexts(self, context_ids: list[int]) -> dict[int, list[str]]:
        """Get tags for multiple context entries in a single query.

        Args:
            context_ids: List of context entry IDs

        Returns:
            Dictionary mapping context IDs to their tags
        """
        if not context_ids:
            return {}

        def _get_tags_batch(conn: sqlite3.Connection) -> dict[int, list[str]]:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in context_ids])
            cursor.execute(
                f'''
                SELECT context_entry_id, tag
                FROM tags
                WHERE context_entry_id IN ({placeholders})
                ORDER BY context_entry_id, tag
                ''',
                tuple(context_ids),
            )

            # Group tags by context_id
            result: dict[int, list[str]] = {}
            for row in cursor.fetchall():
                ctx_id = row['context_entry_id']
                if ctx_id not in result:
                    result[ctx_id] = []
                result[ctx_id].append(row['tag'])

            # Ensure all requested IDs have an entry (even if empty)
            for ctx_id in context_ids:
                if ctx_id not in result:
                    result[ctx_id] = []

            return result

        return await self.db_manager.execute_read(_get_tags_batch)
