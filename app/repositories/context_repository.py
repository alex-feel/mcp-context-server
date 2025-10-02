"""
Context repository for managing context entries.

This module handles all database operations related to context entries,
including CRUD operations and deduplication logic.
"""

import json
import logging
import sqlite3
from typing import Any

from pydantic import ValidationError

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
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        limit: int = 50,
        offset: int = 0,
        explain_query: bool = False,
    ) -> tuple[list[sqlite3.Row], dict[str, Any]]:
        """Search for context entries with filtering including metadata.

        Args:
            thread_id: Filter by thread ID
            source: Filter by source ('user' or 'agent')
            content_type: Filter by content type
            tags: Filter by tags (OR logic)
            metadata: Simple metadata filters (key=value)
            metadata_filters: Advanced metadata filters with operators
            limit: Maximum number of results
            offset: Pagination offset
            explain_query: If True, include query execution plan

        Returns:
            Tuple of (matching rows, query statistics)
        """
        import time as time_module

        from app.metadata_types import MetadataFilter
        from app.query_builder import MetadataQueryBuilder

        def _search(conn: sqlite3.Connection) -> tuple[list[sqlite3.Row], dict[str, Any]]:
            start_time = time_module.time()
            cursor = conn.cursor()

            # Build query with indexed fields first for optimization
            query = 'SELECT * FROM context_entries WHERE 1=1'
            params: list[Any] = []

            # Thread filter (indexed)
            if thread_id:
                query += ' AND thread_id = ?'
                params.append(thread_id)

            # Source filter (indexed)
            if source:
                if source not in ['user', 'agent']:
                    return [], {'error': 'Invalid source type'}
                query += ' AND source = ?'
                params.append(source)

            # Content type filter
            if content_type:
                query += ' AND content_type = ?'
                params.append(content_type)

            # Add metadata filtering
            metadata_builder = MetadataQueryBuilder()

            # Simple metadata filters
            if metadata:
                for key, value in metadata.items():
                    metadata_builder.add_simple_filter(key, value)

            # Advanced metadata filters
            if metadata_filters:
                validation_errors: list[str] = []
                for filter_dict in metadata_filters:
                    try:
                        # Convert dict to MetadataFilter
                        filter_spec = MetadataFilter(**filter_dict)
                        metadata_builder.add_advanced_filter(filter_spec)
                    except ValidationError as e:
                        # Collect validation errors to return to user
                        error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                        validation_errors.append(error_msg)
                    except ValueError as e:
                        # Handle value errors (e.g., from field validators)
                        error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                        validation_errors.append(error_msg)
                    except Exception as e:
                        # Unexpected errors - still collect them
                        error_msg = f'Unexpected error in metadata filter {filter_dict}: {e}'
                        validation_errors.append(error_msg)
                        logger.error(f'Unexpected error processing metadata filter: {e}')

                # If there were validation errors, return them immediately
                if validation_errors:
                    error_response = {
                        'error': 'Metadata filter validation failed',
                        'validation_errors': validation_errors,
                        'execution_time_ms': 0.0,
                        'filters_applied': 0,
                        'rows_returned': 0,
                    }
                    return [], error_response

            # Add metadata conditions to query
            metadata_clause, metadata_params = metadata_builder.build_where_clause()
            if metadata_clause:
                query += f' AND {metadata_clause}'
                params.extend(metadata_params)

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

            # Calculate execution time
            execution_time_ms = (time_module.time() - start_time) * 1000

            # Build statistics
            stats: dict[str, Any] = {
                'execution_time_ms': round(execution_time_ms, 2),
                'filters_applied': metadata_builder.get_filter_count(),
                'rows_returned': len(rows),
            }

            # Get query plan if requested
            if explain_query:
                cursor.execute(f'EXPLAIN QUERY PLAN {query}', tuple(params))
                plan_rows = cursor.fetchall()
                # Convert sqlite3.Row objects to readable format
                plan_data: list[str] = []
                for row in plan_rows:
                    # Convert sqlite3.Row to dict to avoid <Row object> repr
                    row_dict = dict(row)
                    # SQLite EXPLAIN QUERY PLAN columns: id, parent, notused, detail
                    id_val = row_dict.get('id', '?')
                    parent_val = row_dict.get('parent', '?')
                    notused_val = row_dict.get('notused', '?')
                    detail_val = row_dict.get('detail', '?')
                    formatted = f'id:{id_val} parent:{parent_val} notused:{notused_val} detail:{detail_val}'
                    plan_data.append(formatted)
                stats['query_plan'] = '\n'.join(plan_data)

            # Always return tuple with rows and statistics
            rows_list: list[sqlite3.Row] = list(rows)
            result: tuple[list[sqlite3.Row], dict[str, Any]] = (rows_list, stats)
            return result

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

    async def update_context_entry(
        self,
        context_id: int,
        text_content: str | None = None,
        metadata: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Update text content and/or metadata of a context entry.

        Args:
            context_id: ID of the context entry to update
            text_content: New text content (if provided)
            metadata: New metadata JSON string (if provided)

        Returns:
            Tuple of (success, list_of_updated_fields)
        """
        def _update_entry(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
            cursor = conn.cursor()
            updated_fields: list[str] = []

            # First, check if the entry exists
            cursor.execute('SELECT id FROM context_entries WHERE id = ?', (context_id,))
            if not cursor.fetchone():
                return False, []

            # Build update query dynamically based on provided fields
            update_parts: list[str] = []
            params: list[Any] = []

            if text_content is not None:
                update_parts.append('text_content = ?')
                params.append(text_content)
                updated_fields.append('text_content')

            if metadata is not None:
                update_parts.append('metadata = ?')
                params.append(metadata)
                updated_fields.append('metadata')

            # If no fields to update, return early
            if not update_parts:
                return False, []

            # Always update the updated_at timestamp
            update_parts.append('updated_at = CURRENT_TIMESTAMP')

            # Execute update
            query = f"UPDATE context_entries SET {', '.join(update_parts)} WHERE id = ?"
            params.append(context_id)
            cursor.execute(query, tuple(params))

            # Check if any rows were affected
            if cursor.rowcount > 0:
                logger.debug(f'Updated context entry {context_id}, fields: {updated_fields}')
                return True, updated_fields

            return False, []

        return await self.db_manager.execute_write(_update_entry)

    async def check_entry_exists(self, context_id: int) -> bool:
        """Check if a context entry exists.

        Args:
            context_id: ID of the context entry

        Returns:
            True if the entry exists, False otherwise
        """
        def _check_exists(conn: sqlite3.Connection) -> bool:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM context_entries WHERE id = ? LIMIT 1', (context_id,))
            return cursor.fetchone() is not None

        return await self.db_manager.execute_read(_check_exists)

    async def get_content_type(self, context_id: int) -> str | None:
        """Get the content type of a context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            Content type ('text' or 'multimodal') or None if entry doesn't exist
        """
        def _get_content_type(conn: sqlite3.Connection) -> str | None:
            cursor = conn.cursor()
            cursor.execute('SELECT content_type FROM context_entries WHERE id = ?', (context_id,))
            row = cursor.fetchone()
            return row['content_type'] if row else None

        return await self.db_manager.execute_read(_get_content_type)

    async def update_content_type(self, context_id: int, content_type: str) -> bool:
        """Update the content type of a context entry.

        Args:
            context_id: ID of the context entry
            content_type: New content type ('text' or 'multimodal')

        Returns:
            True if updated successfully, False otherwise
        """
        def _update_content_type(conn: sqlite3.Connection) -> bool:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE context_entries SET content_type = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (content_type, context_id),
            )
            return cursor.rowcount > 0

        return await self.db_manager.execute_write(_update_content_type)

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
