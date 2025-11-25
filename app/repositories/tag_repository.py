"""
Tag repository for managing context entry tags.

This module handles all database operations related to tags,
including storage and retrieval of normalized tags.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg
else:
    with contextlib.suppress(ImportError):
        import asyncpg


class TagRepository(BaseRepository):
    """Repository for tag operations.

    Handles storage and retrieval of normalized tags associated
    with context entries.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize tag repository.

        Args:
            backend: Storage backend for executing database operations
        """
        super().__init__(backend)

    async def store_tags(self, context_id: int, tags: list[str]) -> None:
        """Store normalized tags for a context entry.

        Args:
            context_id: ID of the context entry
            tags: List of tags to store (will be normalized)
        """
        if self.backend.backend_type == 'sqlite':

            def _store_tags_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()
                for tag in tags:
                    tag = tag.strip().lower()
                    if tag:
                        query = (
                            f'INSERT INTO tags (context_entry_id, tag) VALUES ({self._placeholder(1)}, {self._placeholder(2)})'
                        )
                        cursor.execute(query, (context_id, tag))

            await self.backend.execute_write(_store_tags_sqlite)
        else:  # postgresql

            async def _store_tags_postgresql(conn: asyncpg.Connection) -> None:
                for tag in tags:
                    tag = tag.strip().lower()
                    if tag:
                        query = (
                            f'INSERT INTO tags (context_entry_id, tag) VALUES ({self._placeholder(1)}, {self._placeholder(2)})'
                        )
                        await conn.execute(query, context_id, tag)

            await self.backend.execute_write(cast(Any, _store_tags_postgresql))

    async def get_tags_for_context(self, context_id: int) -> list[str]:
        """Get all tags for a specific context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            List of tags associated with the context entry
        """
        if self.backend.backend_type == 'sqlite':

            def _get_tags_sqlite(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.cursor()
                query = f'SELECT tag FROM tags WHERE context_entry_id = {self._placeholder(1)} ORDER BY tag'
                cursor.execute(query, (context_id,))
                return [row['tag'] for row in cursor.fetchall()]

            return await self.backend.execute_read(_get_tags_sqlite)

        # postgresql

        async def _get_tags_postgresql(conn: asyncpg.Connection) -> list[str]:
            query = f'SELECT tag FROM tags WHERE context_entry_id = {self._placeholder(1)} ORDER BY tag'
            rows = await conn.fetch(query, context_id)
            return [row['tag'] for row in rows]

        return await self.backend.execute_read(_get_tags_postgresql)

    async def get_tags_for_contexts(self, context_ids: list[int]) -> dict[int, list[str]]:
        """Get tags for multiple context entries in a single query.

        Args:
            context_ids: List of context entry IDs

        Returns:
            Dictionary mapping context IDs to their tags
        """
        if not context_ids:
            return {}

        if self.backend.backend_type == 'sqlite':

            def _get_tags_batch_sqlite(conn: sqlite3.Connection) -> dict[int, list[str]]:
                cursor = conn.cursor()
                placeholders = self._placeholders(len(context_ids))
                query = f'''
                    SELECT context_entry_id, tag
                    FROM tags
                    WHERE context_entry_id IN ({placeholders})
                    ORDER BY context_entry_id, tag
                '''
                cursor.execute(query, tuple(context_ids))

                result: dict[int, list[str]] = {}
                for row in cursor.fetchall():
                    ctx_id = row['context_entry_id']
                    if ctx_id not in result:
                        result[ctx_id] = []
                    result[ctx_id].append(row['tag'])

                for ctx_id in context_ids:
                    if ctx_id not in result:
                        result[ctx_id] = []

                return result

            return await self.backend.execute_read(_get_tags_batch_sqlite)

        # postgresql

        async def _get_tags_batch_postgresql(conn: asyncpg.Connection) -> dict[int, list[str]]:
            placeholders = self._placeholders(len(context_ids))
            query = f'''
                    SELECT context_entry_id, tag
                    FROM tags
                    WHERE context_entry_id IN ({placeholders})
                    ORDER BY context_entry_id, tag
                '''
            rows = await conn.fetch(query, *context_ids)

            result: dict[int, list[str]] = {}
            for row in rows:
                ctx_id = row['context_entry_id']
                if ctx_id not in result:
                    result[ctx_id] = []
                result[ctx_id].append(row['tag'])

            for ctx_id in context_ids:
                if ctx_id not in result:
                    result[ctx_id] = []

            return result

        return await self.backend.execute_read(_get_tags_batch_postgresql)

    async def replace_tags_for_context(self, context_id: int, tags: list[str]) -> None:
        """Replace all tags for a context entry.

        This method performs a complete replacement of tags:
        1. Deletes all existing tags for the context
        2. Inserts new normalized tags

        Args:
            context_id: ID of the context entry
            tags: New list of tags (will be normalized)
        """
        if self.backend.backend_type == 'sqlite':

            def _replace_tags_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()

                delete_query = f'DELETE FROM tags WHERE context_entry_id = {self._placeholder(1)}'
                cursor.execute(delete_query, (context_id,))

                for tag in tags:
                    tag = tag.strip().lower()
                    if tag:
                        insert_query = (
                            f'INSERT INTO tags (context_entry_id, tag) VALUES ({self._placeholder(1)}, {self._placeholder(2)})'
                        )
                        cursor.execute(insert_query, (context_id, tag))

            await self.backend.execute_write(_replace_tags_sqlite)
        else:  # postgresql

            async def _replace_tags_postgresql(conn: asyncpg.Connection) -> None:
                delete_query = f'DELETE FROM tags WHERE context_entry_id = {self._placeholder(1)}'
                await conn.execute(delete_query, context_id)

                for tag in tags:
                    tag = tag.strip().lower()
                    if tag:
                        insert_query = (
                            f'INSERT INTO tags (context_entry_id, tag) VALUES ({self._placeholder(1)}, {self._placeholder(2)})'
                        )
                        await conn.execute(insert_query, context_id, tag)

            await self.backend.execute_write(cast(Any, _replace_tags_postgresql))
