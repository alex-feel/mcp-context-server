"""
Context repository for managing context entries.

This module handles all database operations related to context entries,
including CRUD operations and deduplication logic.
"""

import json
import logging
import sqlite3
from typing import Any

from app.db_manager import DatabaseConnectionManager
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ContextRepository(BaseRepository):
    """Repository for context entry operations.

    Handles storage, retrieval, search, and deletion of context entries
    with proper deduplication and transaction management.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        """Initialize context repository.

        Args:
            db_manager: Database connection manager for executing operations
        """
        super().__init__(db_manager)

    async def store_with_deduplication(
        self,
        thread_id: str,
        source: str,
        content_type: str,
        text_content: str,
        metadata: str | None = None,
    ) -> tuple[int, bool]:
        """Store context entry with deduplication logic.

        Checks if the latest entry has identical thread_id, source, and text_content.
        If found, updates the updated_at timestamp. Otherwise, inserts new entry.

        Args:
            thread_id: Thread identifier
            source: 'user' or 'agent'
            content_type: 'text' or 'multimodal'
            text_content: The actual text content
            metadata: JSON metadata string or None

        Returns:
            Tuple of (context_id, was_updated) where was_updated=True means
            an existing entry was updated, False means new entry was inserted.
        """
        def _store_with_deduplication(conn: sqlite3.Connection) -> tuple[int, bool]:
            cursor = conn.cursor()

            # Check if the LATEST entry (by id) for this thread_id and source has the same text_content
            # This ensures we only deduplicate consecutive duplicates, not all duplicates
            cursor.execute(
                '''
                SELECT id, text_content FROM context_entries
                WHERE thread_id = ? AND source = ?
                ORDER BY id DESC
                LIMIT 1
                ''',
                (thread_id, source),
            )

            latest_row = cursor.fetchone()

            if latest_row and latest_row['text_content'] == text_content:
                # The latest entry has identical text - update its timestamp
                existing_id = latest_row['id']
                cursor.execute(
                    '''
                    UPDATE context_entries
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    ''',
                    (existing_id,),
                )
                logger.debug(f'Updated existing context entry {existing_id} for thread {thread_id}')
                return existing_id, True  # (context_id, was_updated)
            # No duplicate - insert new entry as before
            cursor.execute(
                '''
                INSERT INTO context_entries
                (thread_id, source, content_type, text_content, metadata)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (thread_id, source, content_type, text_content, metadata),
            )
            id_result: int | None = cursor.lastrowid
            new_id = id_result if id_result is not None else 0
            logger.debug(f'Inserted new context entry {new_id} for thread {thread_id}')
            return new_id, False  # (context_id, was_updated)

        return await self.db_manager.execute_write(_store_with_deduplication)

    async def search_contexts(
        self,
        thread_id: str | None = None,
        source: str | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        """Search for context entries with filtering.

        Args:
            thread_id: Filter by thread ID
            source: Filter by source ('user' or 'agent')
            content_type: Filter by content type
            tags: Filter by tags (OR logic)
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of matching context entry rows
        """
        def _search(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            cursor = conn.cursor()

            # Build query with indexed fields first for optimization
            query = 'SELECT * FROM context_entries WHERE 1=1'
            params: list[str | int] = []

            # Thread filter (indexed)
            if thread_id:
                query += ' AND thread_id = ?'
                params.append(thread_id)

            # Source filter (indexed)
            if source:
                if source not in ['user', 'agent']:
                    return []
                query += ' AND source = ?'
                params.append(source)

            # Content type filter
            if content_type:
                query += ' AND content_type = ?'
                params.append(content_type)

            # Tag filter (uses subquery with indexed tag table)
            if tags:
                normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
                if normalized_tags:
                    placeholders = ','.join(['?' for _ in normalized_tags])
                    query += f'''
                        AND id IN (
                            SELECT DISTINCT context_entry_id
                            FROM tags
                            WHERE tag IN ({placeholders})
                        )
                    '''
                    params.extend(normalized_tags)

            # Order and pagination - use id as secondary sort for consistency
            query += ' ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?'
            params.extend((limit, offset))

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return list(rows)

        return await self.db_manager.execute_read(_search)

    async def get_by_ids(self, context_ids: list[int]) -> list[sqlite3.Row]:
        """Get context entries by their IDs.

        Args:
            context_ids: List of context entry IDs

        Returns:
            List of context entry rows
        """
        def _fetch(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in context_ids])
            query = f'''
                SELECT * FROM context_entries
                WHERE id IN ({placeholders})
                ORDER BY created_at DESC
            '''
            cursor.execute(query, tuple(context_ids))
            rows = cursor.fetchall()
            return list(rows)

        return await self.db_manager.execute_read(_fetch)

    async def delete_by_ids(self, context_ids: list[int]) -> int:
        """Delete context entries by their IDs.

        Args:
            context_ids: List of context entry IDs to delete

        Returns:
            Number of deleted entries
        """
        def _delete_by_ids(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in context_ids])
            cursor.execute(
                f'DELETE FROM context_entries WHERE id IN ({placeholders})',
                tuple(context_ids),
            )
            count: int = cursor.rowcount
            return count

        return await self.db_manager.execute_write(_delete_by_ids)

    async def delete_by_thread(self, thread_id: str) -> int:
        """Delete all context entries in a thread.

        Args:
            thread_id: Thread ID to delete entries from

        Returns:
            Number of deleted entries
        """
        def _delete_by_thread(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM context_entries WHERE thread_id = ?',
                (thread_id,),
            )
            count: int = cursor.rowcount
            return count

        return await self.db_manager.execute_write(_delete_by_thread)

    @staticmethod
    def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a dictionary.

        Args:
            row: SQLite Row object

        Returns:
            Dictionary representation of the row
        """
        entry = dict(row)

        # Parse JSON metadata if present
        metadata_raw = entry.get('metadata')
        if metadata_raw is not None and hasattr(metadata_raw, 'strip'):
            try:
                entry['metadata'] = json.loads(str(metadata_raw))
            except (json.JSONDecodeError, ValueError, AttributeError):
                entry['metadata'] = None

        return entry
