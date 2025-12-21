"""
Context repository for managing context entries.

This module handles all database operations related to context entries,
including CRUD operations and deduplication logic.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pydantic import ValidationError

from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Explicit column list to avoid exposing internal database columns (e.g., text_search_vector)
# This constant is used in all SELECT queries that return context entries to ensure
# only the expected columns are returned, preventing internal PostgreSQL columns from
# leaking into API responses.
CONTEXT_ENTRY_COLUMNS = 'id, thread_id, source, content_type, text_content, metadata, created_at, updated_at'


class ContextRepository(BaseRepository):
    """Repository for context entry operations.

    Handles storage, retrieval, search, and deletion of context entries
    with proper deduplication and transaction management.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize context repository.

        Args:
            backend: Storage backend for executing database operations
        """
        super().__init__(backend)

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
        if self.backend.backend_type == 'sqlite':

            def _store_sqlite(conn: sqlite3.Connection) -> tuple[int, bool]:
                cursor = conn.cursor()

                # Check if the LATEST entry (by id) for this thread_id and source has the same text_content
                cursor.execute(
                    f'''
                    SELECT id, text_content FROM context_entries
                    WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
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
                        f'''
                        UPDATE context_entries
                        SET updated_at = CURRENT_TIMESTAMP
                        WHERE id = {self._placeholder(1)}
                        ''',
                        (existing_id,),
                    )
                    logger.debug(f'Updated existing context entry {existing_id} for thread {thread_id}')
                    return existing_id, True

                # No duplicate - insert new entry
                cursor.execute(
                    f'''
                    INSERT INTO context_entries
                    (thread_id, source, content_type, text_content, metadata)
                    VALUES ({self._placeholders(5)})
                    ''',
                    (thread_id, source, content_type, text_content, metadata),
                )
                new_id: int = cursor.lastrowid if cursor.lastrowid is not None else 0
                logger.debug(f'Inserted new context entry {new_id} for thread {thread_id}')
                return new_id, False

            return await self.backend.execute_write(_store_sqlite)

        # PostgreSQL
        # Note: TYPE_CHECKING ensures asyncpg.Connection type is only used during type checking
        async def _store_postgresql(conn: asyncpg.Connection) -> tuple[int, bool]:
            # Check latest entry
            latest_row = await conn.fetchrow(
                f'''
                    SELECT id, text_content FROM context_entries
                    WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                    ORDER BY id DESC
                    LIMIT 1
                    ''',
                thread_id,
                source,
            )

            if latest_row and latest_row['text_content'] == text_content:
                # Update timestamp
                existing_id = latest_row['id']
                await conn.execute(
                    f'''
                        UPDATE context_entries
                        SET updated_at = CURRENT_TIMESTAMP
                        WHERE id = {self._placeholder(1)}
                        ''',
                    existing_id,
                )
                logger.debug(f'Updated existing context entry {existing_id} for thread {thread_id}')
                return existing_id, True

            # Insert new entry with RETURNING clause
            new_id_result = await conn.fetchval(
                f'''
                    INSERT INTO context_entries
                    (thread_id, source, content_type, text_content, metadata)
                    VALUES ({self._placeholders(5)})
                    RETURNING id
                    ''',
                thread_id,
                source,
                content_type,
                text_content,
                metadata,
            )
            new_id = cast(int, new_id_result)
            logger.debug(f'Inserted new context entry {new_id} for thread {thread_id}')
            return new_id, False

        return await self.backend.execute_write(_store_postgresql)

    async def search_contexts(
        self,
        thread_id: str | None = None,
        source: str | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
        explain_query: bool = False,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Search for context entries with filtering including metadata and date range.

        Args:
            thread_id: Filter by thread ID
            source: Filter by source ('user' or 'agent')
            content_type: Filter by content type
            tags: Filter by tags (OR logic)
            metadata: Simple metadata filters (key=value)
            metadata_filters: Advanced metadata filters with operators
            start_date: Filter by created_at >= date (ISO 8601 format)
            end_date: Filter by created_at <= date (ISO 8601 format)
            limit: Maximum number of results
            offset: Pagination offset
            explain_query: If True, include query execution plan

        Returns:
            Tuple of (matching rows, query statistics)
            Note: Rows can be sqlite3.Row or asyncpg.Record depending on backend
        """
        import time as time_module

        from app.metadata_types import MetadataFilter
        from app.query_builder import MetadataQueryBuilder

        if self.backend.backend_type == 'sqlite':

            def _search_sqlite(conn: sqlite3.Connection) -> tuple[list[Any], dict[str, Any]]:
                start_time = time_module.time()
                cursor = conn.cursor()

                # Build query with indexed fields first for optimization
                # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
                query = f'SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries WHERE 1=1'
                params: list[Any] = []

                # Thread filter (indexed)
                if thread_id:
                    query += f' AND thread_id = {self._placeholder(len(params) + 1)}'
                    params.append(thread_id)

                # Source filter (indexed)
                if source:
                    query += f' AND source = {self._placeholder(len(params) + 1)}'
                    params.append(source)

                # Content type filter
                if content_type:
                    query += f' AND content_type = {self._placeholder(len(params) + 1)}'
                    params.append(content_type)

                # Date range filtering - Use datetime() to normalize ISO 8601 input
                # datetime() converts all ISO 8601 formats (T separator, Z suffix, timezone offsets)
                # to SQLite's space-separated format 'YYYY-MM-DD HH:MM:SS' for proper comparison.
                # Without datetime(), TEXT comparison fails because 'T' > ' ' in ASCII ordering.
                if start_date:
                    query += f' AND created_at >= datetime({self._placeholder(len(params) + 1)})'
                    params.append(start_date)

                if end_date:
                    query += f' AND created_at <= datetime({self._placeholder(len(params) + 1)})'
                    params.append(end_date)

                # Add metadata filtering
                metadata_builder = MetadataQueryBuilder(backend_type='sqlite')

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
                        tag_placeholders = ','.join([
                            self._placeholder(len(params) + i + 1) for i in range(len(normalized_tags))
                        ])
                        query += f'''
                            AND id IN (
                                SELECT DISTINCT context_entry_id
                                FROM tags
                                WHERE tag IN ({tag_placeholders})
                            )
                        '''
                        params.extend(normalized_tags)

                # Order and pagination - use id as secondary sort for consistency
                limit_placeholder = self._placeholder(len(params) + 1)
                offset_placeholder = self._placeholder(len(params) + 2)
                query += f' ORDER BY created_at DESC, id DESC LIMIT {limit_placeholder} OFFSET {offset_placeholder}'
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
                    'backend': 'sqlite',
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

                # Return list of rows and statistics
                return list(rows), stats

            return await self.backend.execute_read(_search_sqlite)

        # PostgreSQL
        async def _search_postgresql(conn: asyncpg.Connection) -> tuple[list[Any], dict[str, Any]]:
            start_time = time_module.time()

            # Build query with indexed fields first for optimization
            # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
            query = f'SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries WHERE 1=1'
            params: list[Any] = []

            # Thread filter (indexed)
            if thread_id:
                query += f' AND thread_id = {self._placeholder(len(params) + 1)}'
                params.append(thread_id)

            # Source filter (indexed)
            if source:
                query += f' AND source = {self._placeholder(len(params) + 1)}'
                params.append(source)

            # Content type filter
            if content_type:
                query += f' AND content_type = {self._placeholder(len(params) + 1)}'
                params.append(content_type)

            # Date range filtering - PostgreSQL uses TIMESTAMPTZ comparison
            # asyncpg requires Python datetime objects, not strings, for TIMESTAMPTZ parameters
            if start_date:
                query += f' AND created_at >= {self._placeholder(len(params) + 1)}'
                params.append(self._parse_date_for_postgresql(start_date))

            if end_date:
                query += f' AND created_at <= {self._placeholder(len(params) + 1)}'
                params.append(self._parse_date_for_postgresql(end_date))

            # Add metadata filtering
            # Pass param_offset so metadata builder knows current parameter position
            metadata_builder = MetadataQueryBuilder(backend_type='postgresql', param_offset=len(params))

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
                    tag_placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(normalized_tags))])
                    query += f'''
                        AND id IN (
                            SELECT DISTINCT context_entry_id
                            FROM tags
                            WHERE tag IN ({tag_placeholders})
                        )
                    '''
                    params.extend(normalized_tags)

            # Order and pagination - use id as secondary sort for consistency
            limit_placeholder = self._placeholder(len(params) + 1)
            offset_placeholder = self._placeholder(len(params) + 2)
            query += f' ORDER BY created_at DESC, id DESC LIMIT {limit_placeholder} OFFSET {offset_placeholder}'
            params.extend((limit, offset))

            rows = await conn.fetch(query, *params)

            # Calculate execution time
            execution_time_ms = (time_module.time() - start_time) * 1000

            # Build statistics
            stats: dict[str, Any] = {
                'execution_time_ms': round(execution_time_ms, 2),
                'filters_applied': metadata_builder.get_filter_count(),
                'rows_returned': len(rows),
                'backend': 'postgresql',
            }

            # Get query plan if requested (PostgreSQL EXPLAIN format)
            if explain_query:
                explain_result = await conn.fetch(f'EXPLAIN {query}', *params)
                plan_data: list[str] = [record['QUERY PLAN'] for record in explain_result]
                stats['query_plan'] = '\n'.join(plan_data)

            # Return list of rows and statistics
            return list(rows), stats

        return await self.backend.execute_read(_search_postgresql)

    async def get_by_ids(self, context_ids: list[int]) -> list[Any]:
        """Get context entries by their IDs.

        Args:
            context_ids: List of context entry IDs

        Returns:
            List of context entry rows (sqlite3.Row or asyncpg.Record depending on backend)
        """
        # Defensive check: return empty list if no IDs provided
        # Prevents SQL syntax errors when constructing IN clauses
        if not context_ids:
            return []

        if self.backend.backend_type == 'sqlite':

            def _fetch_sqlite(conn: sqlite3.Connection) -> list[Any]:
                cursor = conn.cursor()
                placeholders = ','.join([self._placeholder(i + 1) for i in range(len(context_ids))])
                # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
                query = f'''
                    SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries
                    WHERE id IN ({placeholders})
                    ORDER BY created_at DESC
                '''
                cursor.execute(query, tuple(context_ids))
                return list(cursor.fetchall())

            return await self.backend.execute_read(_fetch_sqlite)

        # PostgreSQL
        async def _fetch_postgresql(conn: asyncpg.Connection) -> list[Any]:
            placeholders = ','.join([self._placeholder(i + 1) for i in range(len(context_ids))])
            # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
            query = f'''
                SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries
                WHERE id IN ({placeholders})
                ORDER BY created_at DESC
            '''
            rows = await conn.fetch(query, *context_ids)
            return list(rows)

        return await self.backend.execute_read(_fetch_postgresql)

    async def delete_by_ids(self, context_ids: list[int]) -> int:
        """Delete context entries by their IDs.

        Args:
            context_ids: List of context entry IDs to delete

        Returns:
            Number of deleted entries
        """
        # Defensive check: return 0 if no IDs provided
        # Prevents SQL syntax errors when constructing IN clauses
        if not context_ids:
            return 0

        if self.backend.backend_type == 'sqlite':

            def _delete_by_ids_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                placeholders = ','.join([self._placeholder(i + 1) for i in range(len(context_ids))])
                cursor.execute(
                    f'DELETE FROM context_entries WHERE id IN ({placeholders})',
                    tuple(context_ids),
                )
                return cursor.rowcount

            return await self.backend.execute_write(_delete_by_ids_sqlite)

        # PostgreSQL
        async def _delete_by_ids_postgresql(conn: asyncpg.Connection) -> int:
            placeholders = ','.join([self._placeholder(i + 1) for i in range(len(context_ids))])
            result = await conn.execute(
                f'DELETE FROM context_entries WHERE id IN ({placeholders})',
                *context_ids,
            )
            # asyncpg returns "DELETE N" where N is the count
            return int(result.split()[-1]) if result else 0

        return await self.backend.execute_write(_delete_by_ids_postgresql)

    async def delete_by_thread(self, thread_id: str) -> int:
        """Delete all context entries in a thread.

        Args:
            thread_id: Thread ID to delete entries from

        Returns:
            Number of deleted entries
        """
        if self.backend.backend_type == 'sqlite':

            def _delete_by_thread_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                cursor.execute(
                    f'DELETE FROM context_entries WHERE thread_id = {self._placeholder(1)}',
                    (thread_id,),
                )
                return cursor.rowcount

            return await self.backend.execute_write(_delete_by_thread_sqlite)

        # PostgreSQL
        async def _delete_by_thread_postgresql(conn: asyncpg.Connection) -> int:
            result = await conn.execute(
                f'DELETE FROM context_entries WHERE thread_id = {self._placeholder(1)}',
                thread_id,
            )
            # asyncpg returns "DELETE N" where N is the count
            return int(result.split()[-1]) if result else 0

        return await self.backend.execute_write(_delete_by_thread_postgresql)

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
        if self.backend.backend_type == 'sqlite':

            def _update_entry_sqlite(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
                cursor = conn.cursor()
                updated_fields: list[str] = []

                # First, check if the entry exists
                cursor.execute(
                    f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                    (context_id,),
                )
                if not cursor.fetchone():
                    return False, []

                # Build update query dynamically based on provided fields
                update_parts: list[str] = []
                params: list[Any] = []

                if text_content is not None:
                    update_parts.append(f'text_content = {self._placeholder(len(params) + 1)}')
                    params.append(text_content)
                    updated_fields.append('text_content')

                if metadata is not None:
                    update_parts.append(f'metadata = {self._placeholder(len(params) + 1)}')
                    params.append(metadata)
                    updated_fields.append('metadata')

                # If no fields to update, return early
                if not update_parts:
                    return False, []

                # Always update the updated_at timestamp
                update_parts.append('updated_at = CURRENT_TIMESTAMP')

                # Execute update
                query = f"UPDATE context_entries SET {', '.join(update_parts)} WHERE id = {self._placeholder(len(params) + 1)}"
                params.append(context_id)
                cursor.execute(query, tuple(params))

                # Check if any rows were affected
                if cursor.rowcount > 0:
                    logger.debug(f'Updated context entry {context_id}, fields: {updated_fields}')
                    return True, updated_fields

                return False, []

            return await self.backend.execute_write(_update_entry_sqlite)

        # PostgreSQL
        async def _update_entry_postgresql(conn: asyncpg.Connection) -> tuple[bool, list[str]]:
            updated_fields: list[str] = []

            # First, check if the entry exists
            row = await conn.fetchrow(
                f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                context_id,
            )
            if not row:
                return False, []

            # Build update query dynamically based on provided fields
            update_parts: list[str] = []
            params: list[Any] = []

            if text_content is not None:
                update_parts.append(f'text_content = {self._placeholder(len(params) + 1)}')
                params.append(text_content)
                updated_fields.append('text_content')

            if metadata is not None:
                update_parts.append(f'metadata = {self._placeholder(len(params) + 1)}')
                params.append(metadata)
                updated_fields.append('metadata')

            # If no fields to update, return early
            if not update_parts:
                return False, []

            # Always update the updated_at timestamp
            update_parts.append('updated_at = CURRENT_TIMESTAMP')

            # Execute update
            query = f"UPDATE context_entries SET {', '.join(update_parts)} WHERE id = {self._placeholder(len(params) + 1)}"
            params.append(context_id)
            result = await conn.execute(query, *params)

            # Check if any rows were affected (asyncpg returns "UPDATE N")
            rows_affected = int(result.split()[-1]) if result else 0
            if rows_affected > 0:
                logger.debug(f'Updated context entry {context_id}, fields: {updated_fields}')
                return True, updated_fields

            return False, []

        return await self.backend.execute_write(_update_entry_postgresql)

    async def check_entry_exists(self, context_id: int) -> bool:
        """Check if a context entry exists.

        Args:
            context_id: ID of the context entry

        Returns:
            True if the entry exists, False otherwise
        """
        if self.backend.backend_type == 'sqlite':

            def _check_exists_sqlite(conn: sqlite3.Connection) -> bool:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT 1 FROM context_entries WHERE id = {self._placeholder(1)} LIMIT 1',
                    (context_id,),
                )
                return cursor.fetchone() is not None

            return await self.backend.execute_read(_check_exists_sqlite)

        # PostgreSQL
        async def _check_exists_postgresql(conn: asyncpg.Connection) -> bool:
            row = await conn.fetchrow(
                f'SELECT 1 FROM context_entries WHERE id = {self._placeholder(1)} LIMIT 1',
                context_id,
            )
            return row is not None

        return await self.backend.execute_read(_check_exists_postgresql)

    async def get_content_type(self, context_id: int) -> str | None:
        """Get the content type of a context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            Content type ('text' or 'multimodal') or None if entry doesn't exist
        """
        if self.backend.backend_type == 'sqlite':

            def _get_content_type_sqlite(conn: sqlite3.Connection) -> str | None:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT content_type FROM context_entries WHERE id = {self._placeholder(1)}',
                    (context_id,),
                )
                row = cursor.fetchone()
                return row['content_type'] if row else None

            return await self.backend.execute_read(_get_content_type_sqlite)

        # PostgreSQL
        async def _get_content_type_postgresql(conn: asyncpg.Connection) -> str | None:
            row = await conn.fetchrow(
                f'SELECT content_type FROM context_entries WHERE id = {self._placeholder(1)}',
                context_id,
            )
            return row['content_type'] if row else None

        return await self.backend.execute_read(_get_content_type_postgresql)

    async def update_content_type(self, context_id: int, content_type: str) -> bool:
        """Update the content type of a context entry.

        Args:
            context_id: ID of the context entry
            content_type: New content type ('text' or 'multimodal')

        Returns:
            True if updated successfully, False otherwise
        """
        if self.backend.backend_type == 'sqlite':

            def _update_content_type_sqlite(conn: sqlite3.Connection) -> bool:
                cursor = conn.cursor()
                content_type_placeholder = self._placeholder(1)
                id_placeholder = self._placeholder(2)
                query = (
                    f'UPDATE context_entries SET content_type = {content_type_placeholder}, '
                    f'updated_at = CURRENT_TIMESTAMP WHERE id = {id_placeholder}'
                )
                cursor.execute(query, (content_type, context_id))
                return cursor.rowcount > 0

            return await self.backend.execute_write(_update_content_type_sqlite)

        # PostgreSQL
        async def _update_content_type_postgresql(conn: asyncpg.Connection) -> bool:
            content_type_placeholder = self._placeholder(1)
            id_placeholder = self._placeholder(2)
            query = (
                f'UPDATE context_entries SET content_type = {content_type_placeholder}, '
                f'updated_at = CURRENT_TIMESTAMP WHERE id = {id_placeholder}'
            )
            result = await conn.execute(query, content_type, context_id)
            # asyncpg returns "UPDATE N" where N is the count
            return int(result.split()[-1]) > 0 if result else False

        return await self.backend.execute_write(_update_content_type_postgresql)

    async def patch_metadata(
        self,
        context_id: int,
        patch: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Apply RFC 7396 JSON Merge Patch to metadata atomically.

        This method performs a partial update of the metadata field using database-native
        JSON patching functions for atomic, race-condition-free operations.

        RFC 7396 JSON Merge Patch Semantics:
        - New keys in patch are ADDED to existing metadata
        - Existing keys are REPLACED with new values
        - Keys with null values are DELETED from metadata

        IMPORTANT LIMITATIONS (RFC 7396):
        - Cannot set a value to null: null always means DELETE. If you need to store
          null values, use the full metadata replacement (metadata parameter) instead.
        - Array operations are replace-only: Arrays are replaced entirely, not merged.
          Individual array elements cannot be added, removed, or modified - the entire
          array is replaced with the new value.
        - Empty patch {} is a no-op for data but still updates the updated_at timestamp.

        Backend-specific implementation:
        - SQLite: Uses json_patch() function (available in SQLite 3.38.0+)
        - PostgreSQL: Uses custom jsonb_merge_patch() function for TRUE recursive deep merge.
          The function is created by migration app/migrations/add_jsonb_merge_patch_postgresql.sql
          and provides identical RFC 7396 semantics to SQLite's json_patch().

        Args:
            context_id: ID of the context entry to update
            patch: Dictionary containing the merge patch to apply

        Returns:
            Tuple of (success, list_of_updated_fields).
            Updated fields will include 'metadata' if successful.
        """
        # Convert patch dict to JSON string for database operations
        patch_json = json.dumps(patch, ensure_ascii=False)

        if self.backend.backend_type == 'sqlite':

            def _patch_metadata_sqlite(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
                cursor = conn.cursor()

                # Verify entry exists before attempting update
                cursor.execute(
                    f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                    (context_id,),
                )
                if not cursor.fetchone():
                    return False, []

                # Apply JSON Merge Patch using SQLite's json_patch() function
                # json_patch() implements RFC 7396 semantics:
                # - COALESCE ensures null metadata is treated as empty object '{}'
                # - json_patch(target, patch) merges patch into target
                # - null values in patch DELETE keys from result
                cursor.execute(
                    f'''
                    UPDATE context_entries
                    SET metadata = json_patch(COALESCE(metadata, '{{}}'), {self._placeholder(1)}),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = {self._placeholder(2)}
                    ''',
                    (patch_json, context_id),
                )

                if cursor.rowcount > 0:
                    logger.debug(f'Patched metadata for context entry {context_id}')
                    return True, ['metadata']

                return False, []

            return await self.backend.execute_write(_patch_metadata_sqlite)

        # PostgreSQL implementation - RFC 7396 compliant using jsonb_merge_patch() function
        async def _patch_metadata_postgresql(conn: asyncpg.Connection) -> tuple[bool, list[str]]:
            # Verify entry exists before attempting update
            row = await conn.fetchrow(
                f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                context_id,
            )
            if not row:
                return False, []

            # RFC 7396 JSON Merge Patch Implementation for PostgreSQL
            #
            # Uses the custom jsonb_merge_patch() function that implements TRUE recursive
            # deep merge semantics as specified in RFC 7396:
            # - New keys in patch are ADDED to existing metadata
            # - Existing keys are REPLACED with new values from patch
            # - Keys with null values are DELETED from metadata
            # - Nested objects are RECURSIVELY merged (not replaced like || operator)
            #
            # The jsonb_merge_patch() function is created by the migration file:
            # app/migrations/add_jsonb_merge_patch_postgresql.sql
            #
            # This approach provides identical behavior to SQLite's json_patch() function,
            # ensuring consistent RFC 7396 semantics across both backends.
            p1 = self._placeholder(1)
            p2 = self._placeholder(2)
            result = await conn.execute(
                f'''
                UPDATE context_entries
                SET metadata = jsonb_merge_patch(COALESCE(metadata, '{{}}'::jsonb), {p1}::jsonb),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = {p2}
                ''',
                patch_json,
                context_id,
            )

            # asyncpg returns "UPDATE N" where N is the count
            rows_affected = int(result.split()[-1]) if result else 0
            if rows_affected > 0:
                logger.debug(f'Patched metadata for context entry {context_id}')
                return True, ['metadata']

            return False, []

        return await self.backend.execute_write(_patch_metadata_postgresql)

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

    async def store_contexts_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> list[tuple[int, int | None, str | None]]:
        """Store multiple context entries in a single transaction.

        Each entry is processed with deduplication logic: if an entry with
        identical thread_id, source, and text_content exists, it is updated
        rather than creating a duplicate.

        Args:
            entries: List of entry dictionaries with keys:
                - thread_id: str
                - source: str
                - text_content: str
                - metadata: str | None (JSON string)
                - content_type: str ('text' or 'multimodal')

        Returns:
            List of tuples: (index, context_id or None, error or None)
            On success: (0, 123, None)
            On failure: (0, None, 'Error message')
        """
        if self.backend.backend_type == 'sqlite':

            def _store_batch_sqlite(conn: sqlite3.Connection) -> list[tuple[int, int | None, str | None]]:
                cursor = conn.cursor()
                results: list[tuple[int, int | None, str | None]] = []

                for idx, entry in enumerate(entries):
                    try:
                        thread_id = entry['thread_id']
                        source = entry['source']
                        text_content = entry['text_content']
                        metadata = entry.get('metadata')
                        content_type = entry.get('content_type', 'text')

                        # Check for deduplication - find latest entry with same thread_id, source, text_content
                        cursor.execute(
                            f'''
                            SELECT id FROM context_entries
                            WHERE thread_id = {self._placeholder(1)}
                            AND source = {self._placeholder(2)}
                            AND text_content = {self._placeholder(3)}
                            ORDER BY id DESC
                            LIMIT 1
                            ''',
                            (thread_id, source, text_content),
                        )
                        existing = cursor.fetchone()

                        if existing:
                            # Update existing entry
                            existing_id = existing['id']
                            cursor.execute(
                                f'''
                                UPDATE context_entries
                                SET metadata = {self._placeholder(1)},
                                    content_type = {self._placeholder(2)},
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE id = {self._placeholder(3)}
                                ''',
                                (metadata, content_type, existing_id),
                            )
                            results.append((idx, existing_id, None))
                            logger.debug(f'Batch: updated existing context entry {existing_id}')
                        else:
                            # Insert new entry
                            cursor.execute(
                                f'''
                                INSERT INTO context_entries
                                (thread_id, source, content_type, text_content, metadata)
                                VALUES ({self._placeholders(5)})
                                ''',
                                (thread_id, source, content_type, text_content, metadata),
                            )
                            new_id = cursor.lastrowid or 0
                            results.append((idx, new_id, None))
                            logger.debug(f'Batch: inserted new context entry {new_id}')

                    except Exception as e:
                        results.append((idx, None, str(e)))
                        logger.warning(f'Batch store failed for entry {idx}: {e}')

                return results

            return await self.backend.execute_write(_store_batch_sqlite)

        # PostgreSQL
        async def _store_batch_postgresql(conn: asyncpg.Connection) -> list[tuple[int, int | None, str | None]]:
            results: list[tuple[int, int | None, str | None]] = []

            for idx, entry in enumerate(entries):
                try:
                    thread_id = entry['thread_id']
                    source = entry['source']
                    text_content = entry['text_content']
                    metadata = entry.get('metadata')
                    content_type = entry.get('content_type', 'text')

                    # Check for deduplication
                    existing = await conn.fetchrow(
                        f'''
                        SELECT id FROM context_entries
                        WHERE thread_id = {self._placeholder(1)}
                        AND source = {self._placeholder(2)}
                        AND text_content = {self._placeholder(3)}
                        ORDER BY id DESC
                        LIMIT 1
                        ''',
                        thread_id,
                        source,
                        text_content,
                    )

                    if existing:
                        # Update existing entry
                        existing_id = existing['id']
                        await conn.execute(
                            f'''
                            UPDATE context_entries
                            SET metadata = {self._placeholder(1)},
                                content_type = {self._placeholder(2)},
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = {self._placeholder(3)}
                            ''',
                            metadata,
                            content_type,
                            existing_id,
                        )
                        results.append((idx, existing_id, None))
                        logger.debug(f'Batch: updated existing context entry {existing_id}')
                    else:
                        # Insert new entry
                        new_id_result = await conn.fetchval(
                            f'''
                            INSERT INTO context_entries
                            (thread_id, source, content_type, text_content, metadata)
                            VALUES ({self._placeholders(5)})
                            RETURNING id
                            ''',
                            thread_id,
                            source,
                            content_type,
                            text_content,
                            metadata,
                        )
                        new_id = cast(int, new_id_result)
                        results.append((idx, new_id, None))
                        logger.debug(f'Batch: inserted new context entry {new_id}')

                except Exception as e:
                    results.append((idx, None, str(e)))
                    logger.warning(f'Batch store failed for entry {idx}: {e}')

            return results

        return await self.backend.execute_write(_store_batch_postgresql)

    async def update_contexts_batch(
        self,
        updates: list[dict[str, Any]],
    ) -> list[tuple[int, int, list[str] | None, str | None]]:
        """Update multiple context entries in a single transaction.

        Args:
            updates: List of update dictionaries with keys:
                - context_id: int (required)
                - text_content: str | None (optional)
                - metadata: str | None (JSON string, full replacement)
                - content_type: str | None (optional)

        Returns:
            List of tuples: (index, context_id, updated_fields or None, error or None)
        """
        if self.backend.backend_type == 'sqlite':

            def _update_batch_sqlite(conn: sqlite3.Connection) -> list[tuple[int, int, list[str] | None, str | None]]:
                cursor = conn.cursor()
                results: list[tuple[int, int, list[str] | None, str | None]] = []

                for idx, update in enumerate(updates):
                    try:
                        context_id = update['context_id']

                        # Check if entry exists
                        cursor.execute(
                            f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                            (context_id,),
                        )
                        if not cursor.fetchone():
                            results.append((idx, context_id, None, f'Context entry {context_id} not found'))
                            continue

                        # Build dynamic update
                        update_parts: list[str] = []
                        params: list[Any] = []
                        updated_fields: list[str] = []

                        if 'text_content' in update and update['text_content'] is not None:
                            update_parts.append(f'text_content = {self._placeholder(len(params) + 1)}')
                            params.append(update['text_content'])
                            updated_fields.append('text_content')

                        if 'metadata' in update:
                            update_parts.append(f'metadata = {self._placeholder(len(params) + 1)}')
                            params.append(update['metadata'])
                            updated_fields.append('metadata')

                        if 'content_type' in update and update['content_type'] is not None:
                            update_parts.append(f'content_type = {self._placeholder(len(params) + 1)}')
                            params.append(update['content_type'])
                            updated_fields.append('content_type')

                        if not update_parts:
                            results.append((idx, context_id, None, 'No fields to update'))
                            continue

                        # Always update timestamp
                        update_parts.append('updated_at = CURRENT_TIMESTAMP')

                        id_placeholder = self._placeholder(len(params) + 1)
                        query = f"UPDATE context_entries SET {', '.join(update_parts)} WHERE id = {id_placeholder}"
                        params.append(context_id)
                        cursor.execute(query, tuple(params))

                        if cursor.rowcount > 0:
                            results.append((idx, context_id, updated_fields, None))
                            logger.debug(f'Batch: updated context entry {context_id}, fields: {updated_fields}')
                        else:
                            results.append((idx, context_id, None, 'Update had no effect'))

                    except Exception as e:
                        results.append((idx, update.get('context_id', 0), None, str(e)))
                        logger.warning(f'Batch update failed for entry {idx}: {e}')

                return results

            return await self.backend.execute_write(_update_batch_sqlite)

        # PostgreSQL
        async def _update_batch_postgresql(
            conn: asyncpg.Connection,
        ) -> list[tuple[int, int, list[str] | None, str | None]]:
            results: list[tuple[int, int, list[str] | None, str | None]] = []

            for idx, update in enumerate(updates):
                try:
                    context_id = update['context_id']

                    # Check if entry exists
                    row = await conn.fetchrow(
                        f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                        context_id,
                    )
                    if not row:
                        results.append((idx, context_id, None, f'Context entry {context_id} not found'))
                        continue

                    # Build dynamic update
                    update_parts: list[str] = []
                    params: list[Any] = []
                    updated_fields: list[str] = []

                    if 'text_content' in update and update['text_content'] is not None:
                        update_parts.append(f'text_content = {self._placeholder(len(params) + 1)}')
                        params.append(update['text_content'])
                        updated_fields.append('text_content')

                    if 'metadata' in update:
                        update_parts.append(f'metadata = {self._placeholder(len(params) + 1)}')
                        params.append(update['metadata'])
                        updated_fields.append('metadata')

                    if 'content_type' in update and update['content_type'] is not None:
                        update_parts.append(f'content_type = {self._placeholder(len(params) + 1)}')
                        params.append(update['content_type'])
                        updated_fields.append('content_type')

                    if not update_parts:
                        results.append((idx, context_id, None, 'No fields to update'))
                        continue

                    # Always update timestamp
                    update_parts.append('updated_at = CURRENT_TIMESTAMP')

                    id_placeholder = self._placeholder(len(params) + 1)
                    set_clause = ', '.join(update_parts)
                    query = f'UPDATE context_entries SET {set_clause} WHERE id = {id_placeholder}'
                    params.append(context_id)
                    result = await conn.execute(query, *params)

                    # asyncpg returns "UPDATE N" where N is the count
                    rows_affected = int(result.split()[-1]) if result else 0
                    if rows_affected > 0:
                        results.append((idx, context_id, updated_fields, None))
                        logger.debug(f'Batch: updated context entry {context_id}, fields: {updated_fields}')
                    else:
                        results.append((idx, context_id, None, 'Update had no effect'))

                except Exception as e:
                    results.append((idx, update.get('context_id', 0), None, str(e)))
                    logger.warning(f'Batch update failed for entry {idx}: {e}')

            return results

        return await self.backend.execute_write(_update_batch_postgresql)

    async def delete_contexts_batch(
        self,
        context_ids: list[int] | None = None,
        thread_ids: list[str] | None = None,
        source: str | None = None,
        older_than_days: int | None = None,
    ) -> tuple[int, list[str]]:
        """Delete multiple context entries by various criteria.

        At least one criterion must be provided. Criteria can be combined
        for more targeted deletion. Cascading delete removes associated
        tags, images, and embeddings.

        Args:
            context_ids: Specific context entry IDs to delete
            thread_ids: Delete all entries in these threads
            source: Filter by source ('user' or 'agent') - combine with other criteria
            older_than_days: Delete entries older than N days

        Returns:
            Tuple of (deleted_count, list_of_criteria_used)
        """
        criteria_used: list[str] = []

        if self.backend.backend_type == 'sqlite':

            def _delete_batch_sqlite(conn: sqlite3.Connection) -> tuple[int, list[str]]:
                cursor = conn.cursor()
                conditions: list[str] = []
                params: list[Any] = []

                if context_ids:
                    placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(context_ids))])
                    conditions.append(f'id IN ({placeholders})')
                    params.extend(context_ids)
                    criteria_used.append(f'context_ids: {len(context_ids)} IDs')

                if thread_ids:
                    placeholders = ','.join([
                        self._placeholder(len(params) + i + 1) for i in range(len(thread_ids))
                    ])
                    conditions.append(f'thread_id IN ({placeholders})')
                    params.extend(thread_ids)
                    criteria_used.append(f'thread_ids: {len(thread_ids)} threads')

                if source:
                    conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                    params.append(source)
                    criteria_used.append(f'source: {source}')

                if older_than_days is not None:
                    conditions.append(
                        f"created_at < datetime('now', {self._placeholder(len(params) + 1)})",
                    )
                    params.append(f'-{older_than_days} days')
                    criteria_used.append(f'older_than_days: {older_than_days}')

                if not conditions:
                    return 0, criteria_used

                where_clause = ' AND '.join(conditions)
                query = f'DELETE FROM context_entries WHERE {where_clause}'
                cursor.execute(query, tuple(params))

                deleted_count = cursor.rowcount
                logger.info(f'Batch delete: removed {deleted_count} entries using criteria: {criteria_used}')
                return deleted_count, criteria_used

            return await self.backend.execute_write(_delete_batch_sqlite)

        # PostgreSQL
        async def _delete_batch_postgresql(conn: asyncpg.Connection) -> tuple[int, list[str]]:
            conditions: list[str] = []
            params: list[Any] = []

            if context_ids:
                placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(context_ids))])
                conditions.append(f'id IN ({placeholders})')
                params.extend(context_ids)
                criteria_used.append(f'context_ids: {len(context_ids)} IDs')

            if thread_ids:
                placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(thread_ids))])
                conditions.append(f'thread_id IN ({placeholders})')
                params.extend(thread_ids)
                criteria_used.append(f'thread_ids: {len(thread_ids)} threads')

            if source:
                conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                params.append(source)
                criteria_used.append(f'source: {source}')

            if older_than_days is not None:
                conditions.append(
                    f"created_at < (NOW() - INTERVAL '{older_than_days} days')",
                )
                criteria_used.append(f'older_than_days: {older_than_days}')

            if not conditions:
                return 0, criteria_used

            where_clause = ' AND '.join(conditions)
            query = f'DELETE FROM context_entries WHERE {where_clause}'
            result = await conn.execute(query, *params)

            # asyncpg returns "DELETE N" where N is the count
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f'Batch delete: removed {deleted_count} entries using criteria: {criteria_used}')
            return deleted_count, criteria_used

        return await self.backend.execute_write(_delete_batch_postgresql)
