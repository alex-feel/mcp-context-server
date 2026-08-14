"""
Tag repository for managing context entry tags.

This module handles all database operations related to tags,
including storage and retrieval of normalized tags.
"""


import contextlib
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.backends.base import StorageBackend
from app.models import normalize_tag_list
from app.repositories._collation import byte_ordered_text
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext
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

    @staticmethod
    def normalize_tags(tags: list[str]) -> list[str]:
        """Reduce a caller-supplied tag list to the exact rows to insert.

        Tags are a SET of labels: an entry either carries a label or it does
        not, and every reader (``get_context_by_ids``, all four search tools,
        the tag statistics) exposes the stored rows verbatim. Inserting the
        same label twice therefore leaks a duplicate into every response and
        inflates the tag counts, so duplicates are removed HERE, at the single
        chokepoint every write path funnels through.

        The rule itself lives in :func:`app.models.normalize_tag_list`, the single
        definition of what makes two tags the same label; this method is the write
        path's entry point into it. Keeping one definition matters because a second
        normalizer that trimmed or folded differently would treat a pair of tags as
        one label where this one treats them as two, so the same input would store a
        different set depending on which path handled it.

        Args:
            tags: Raw tags as supplied by the caller.

        Returns:
            Trimmed, lower-cased, blank-free tags with duplicates removed,
            in first-seen order.
        """
        return normalize_tag_list(tags)

    def _insert_tag_sql(self) -> str:
        """Return the INSERT that stores one tag row, tolerating a repeat.

        ``ON CONFLICT ... DO NOTHING`` names the unique index on
        ``(context_entry_id, tag)`` EXPLICITLY rather than swallowing every constraint:
        a repeated label is the one conflict that is correct to ignore (tags are a set,
        so the row already says what this insert would say), while a foreign-key
        violation still raises, because that one means the parent entry is missing.

        Returns:
            The backend-appropriate INSERT statement.
        """
        return (
            f'INSERT INTO tags (context_entry_id, tag) '
            f'VALUES ({self._placeholder(1)}, {self._placeholder(2)}) '
            f'ON CONFLICT (context_entry_id, tag) DO NOTHING'
        )

    async def store_tags(
        self,
        context_id: str,
        tags: list[str],
        txn: 'TransactionContext | None' = None,
    ) -> None:
        """Store normalized tags for a context entry.

        Args:
            context_id: ID of the context entry
            tags: List of tags to store (normalized and deduplicated by
                :meth:`normalize_tags`)
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        normalized_tags = self.normalize_tags(tags)

        if backend_type == 'sqlite':

            def _store_tags_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()
                query = self._insert_tag_sql()
                for tag in normalized_tags:
                    cursor.execute(query, (context_id, tag))

            if txn:
                await self._run_sqlite_txn(_store_tags_sqlite, cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_store_tags_sqlite)
        else:  # postgresql

            async def _store_tags_postgresql(conn: 'asyncpg.Connection') -> None:
                query = self._insert_tag_sql()
                for tag in normalized_tags:
                    await conn.execute(query, context_id, tag)

            if txn:
                await _store_tags_postgresql(cast('asyncpg.Connection', txn.connection))
            else:
                await self.backend.execute_write(cast(Any, _store_tags_postgresql))

    async def get_tags_for_context(self, context_id: str) -> list[str]:
        """Get all tags for a specific context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            List of tags associated with the context entry, byte-ordered
            identically on both backends
        """
        if self.backend.backend_type == 'sqlite':

            def _get_tags_sqlite(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.cursor()
                order_term = byte_ordered_text('tag', 'sqlite')
                query = (
                    f'SELECT tag FROM tags WHERE context_entry_id = {self._placeholder(1)} ORDER BY {order_term}'
                )
                cursor.execute(query, (context_id,))
                return [row['tag'] for row in cursor.fetchall()]

            return await self.backend.execute_read(_get_tags_sqlite)

        # postgresql

        async def _get_tags_postgresql(conn: 'asyncpg.Connection') -> list[str]:
            order_term = byte_ordered_text('tag', 'postgresql')
            query = f'SELECT tag FROM tags WHERE context_entry_id = {self._placeholder(1)} ORDER BY {order_term}'
            rows = await conn.fetch(query, context_id)
            return [row['tag'] for row in rows]

        return await self.backend.execute_read(_get_tags_postgresql)

    async def get_tags_for_contexts(self, context_ids: list[str]) -> dict[str, list[str]]:
        """Get tags for multiple context entries in a single query.

        Args:
            context_ids: List of context entry IDs

        Returns:
            Dictionary mapping context IDs to their tags, byte-ordered
            identically on both backends
        """
        if not context_ids:
            return {}

        if self.backend.backend_type == 'sqlite':

            def _get_tags_batch_sqlite(conn: sqlite3.Connection) -> dict[str, list[str]]:
                cursor = conn.cursor()
                placeholders = self._placeholders(len(context_ids))
                order_term = byte_ordered_text('tag', 'sqlite')
                query = f'''
                    SELECT context_entry_id, tag
                    FROM tags
                    WHERE context_entry_id IN ({placeholders})
                    ORDER BY context_entry_id, {order_term}
                '''
                cursor.execute(query, tuple(context_ids))

                result: dict[str, list[str]] = {}
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

        async def _get_tags_batch_postgresql(conn: 'asyncpg.Connection') -> dict[str, list[str]]:
            placeholders = self._placeholders(len(context_ids))
            order_term = byte_ordered_text('tag', 'postgresql')
            query = f'''
                    SELECT context_entry_id, tag
                    FROM tags
                    WHERE context_entry_id IN ({placeholders})
                    ORDER BY context_entry_id, {order_term}
                '''
            rows = await conn.fetch(query, *context_ids)

            result: dict[str, list[str]] = {}
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

    async def replace_tags_for_context(
        self,
        context_id: str,
        tags: list[str],
        txn: 'TransactionContext | None' = None,
    ) -> None:
        """Replace all tags for a context entry.

        This method performs a complete replacement of tags:
        1. Deletes all existing tags for the context
        2. Inserts new normalized tags

        Args:
            context_id: ID of the context entry
            tags: New list of tags (normalized and deduplicated by
                :meth:`normalize_tags`)
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        normalized_tags = self.normalize_tags(tags)

        if backend_type == 'sqlite':

            def _replace_tags_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()

                delete_query = f'DELETE FROM tags WHERE context_entry_id = {self._placeholder(1)}'
                cursor.execute(delete_query, (context_id,))

                insert_query = self._insert_tag_sql()
                for tag in normalized_tags:
                    cursor.execute(insert_query, (context_id, tag))

            if txn:
                await self._run_sqlite_txn(_replace_tags_sqlite, cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_replace_tags_sqlite)
        else:  # postgresql

            async def _replace_tags_postgresql(conn: 'asyncpg.Connection') -> None:
                delete_query = f'DELETE FROM tags WHERE context_entry_id = {self._placeholder(1)}'
                await conn.execute(delete_query, context_id)

                insert_query = self._insert_tag_sql()
                for tag in normalized_tags:
                    await conn.execute(insert_query, context_id, tag)

            if txn:
                await _replace_tags_postgresql(cast('asyncpg.Connection', txn.connection))
            else:
                await self.backend.execute_write(cast(Any, _replace_tags_postgresql))
