"""
Repository for vector embeddings supporting both sqlite-vec and pgvector.

This module provides data access for semantic search embeddings,
handling storage, retrieval, and search operations on vector embeddings
across both SQLite (sqlite-vec) and PostgreSQL (pgvector) backends.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

from app.backends.base import StorageBackend
from app.logger_config import config_logger
from app.repositories.base import BaseRepository
from app.settings import get_settings

if TYPE_CHECKING:
    import asyncpg

# Get settings
settings = get_settings()
# Configure logging
config_logger(settings.log_level)
logger = logging.getLogger(__name__)


class MetadataFilterValidationError(Exception):
    """Exception raised when metadata filters fail validation.

    This exception enables unified error handling between search_context
    and semantic_search_context tools.
    """

    def __init__(self, message: str, validation_errors: list[str]) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            validation_errors: List of validation error messages
        """
        super().__init__(message)
        self.message = message
        self.validation_errors = validation_errors


class EmbeddingRepository(BaseRepository):
    """Repository for vector embeddings supporting both sqlite-vec and pgvector.

    This repository handles all database operations for semantic search embeddings,
    using either sqlite-vec extension (SQLite) or pgvector extension (PostgreSQL)
    depending on the configured storage backend.

    Supported backends:
    - SQLite: Uses sqlite-vec with BLOB storage and vec_distance_l2()
    - PostgreSQL: Uses pgvector with native vector type and <-> operator
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize the embedding repository.

        Args:
            backend: Storage backend for all database operations
        """
        super().__init__(backend)

    async def store(
        self,
        context_id: int,
        embedding: list[float],
        model: str = 'embeddinggemma:latest',
    ) -> None:
        """Store embedding for a context entry.

        Args:
            context_id: ID of the context entry
            embedding: 768-dimensional embedding vector
            model: Model identifier
        """
        if self.backend.backend_type == 'sqlite':

            def _store_sqlite(conn: sqlite3.Connection) -> None:
                try:
                    import sqlite_vec
                except ImportError as e:
                    raise RuntimeError(
                        'sqlite_vec package is required for semantic search. Install with: uv sync --extra semantic-search',
                    ) from e

                embedding_blob = sqlite_vec.serialize_float32(embedding)
                query1 = (
                    f'INSERT INTO vec_context_embeddings(rowid, embedding) '
                    f'VALUES ({self._placeholder(1)}, {self._placeholder(2)})'
                )
                conn.execute(query1, (context_id, embedding_blob))

                query2 = (
                    f'INSERT INTO embedding_metadata (context_id, model_name, dimensions, created_at, updated_at) '
                    f'VALUES ({self._placeholder(1)}, {self._placeholder(2)}, {self._placeholder(3)}, '
                    f'CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)'
                )
                conn.execute(query2, (context_id, model, len(embedding)))

            await self.backend.execute_write(_store_sqlite)
            logger.debug(f'Stored embedding for context {context_id} (SQLite)')

        else:  # postgresql

            async def _store_postgresql(conn: asyncpg.Connection) -> None:
                # Insert into vec_context_embeddings
                query1 = (
                    f'INSERT INTO vec_context_embeddings(context_id, embedding) '
                    f'VALUES ({self._placeholder(1)}, {self._placeholder(2)})'
                )
                await conn.execute(query1, context_id, embedding)

                # Insert into embedding_metadata
                query2 = (
                    f'INSERT INTO embedding_metadata (context_id, model_name, dimensions, created_at, updated_at) '
                    f'VALUES ({self._placeholder(1)}, {self._placeholder(2)}, {self._placeholder(3)}, '
                    f'CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)'
                )
                await conn.execute(query2, context_id, model, len(embedding))
                return

            await self.backend.execute_write(cast(Any, _store_postgresql))
            logger.debug(f'Stored embedding for context {context_id} (PostgreSQL)')

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 20,
        offset: int = 0,
        thread_id: str | None = None,
        source: Literal['user', 'agent'] | None = None,
        content_type: Literal['text', 'multimodal'] | None = None,
        tags: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        explain_query: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """KNN search with optional filters including date range and metadata.

        SQLite: Uses CTE-based pre-filtering with vec_distance_l2() function
        PostgreSQL: Uses direct JOIN with <-> operator for L2 distance

        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results to return
            offset: Number of results to skip (pagination)
            thread_id: Optional filter by thread
            source: Optional filter by source type
            content_type: Filter by content type (text or multimodal)
            tags: Filter by any of these tags (OR logic)
            start_date: Filter by created_at >= date (ISO 8601 format)
            end_date: Filter by created_at <= date (ISO 8601 format)
            metadata: Simple metadata filters (key=value equality)
            metadata_filters: Advanced metadata filters with operators
            explain_query: If True, include query execution plan in stats

        Returns:
            Tuple of (search results list, statistics dictionary)
        """
        if self.backend.backend_type == 'sqlite':

            def _search_sqlite(
                conn: sqlite3.Connection,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                import time as time_module

                start_time = time_module.time()

                try:
                    import sqlite_vec
                except ImportError as e:
                    raise RuntimeError(
                        'sqlite_vec package is required for semantic search. Install with: uv sync --extra semantic-search',
                    ) from e

                query_blob = sqlite_vec.serialize_float32(query_embedding)

                filter_conditions = []
                filter_params: list[Any] = []

                # Count filters applied
                filter_count = 0

                if thread_id:
                    filter_conditions.append('thread_id = ?')
                    filter_params.append(thread_id)
                    filter_count += 1

                if source:
                    filter_conditions.append('source = ?')
                    filter_params.append(source)
                    filter_count += 1

                if content_type:
                    filter_conditions.append('content_type = ?')
                    filter_params.append(content_type)
                    filter_count += 1

                # Tag filter (uses subquery with indexed tag table)
                if tags:
                    normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
                    if normalized_tags:
                        tag_placeholders = ','.join(['?' for _ in normalized_tags])
                        filter_conditions.append(f'''
                            id IN (
                                SELECT DISTINCT context_entry_id
                                FROM tags
                                WHERE tag IN ({tag_placeholders})
                            )
                        ''')
                        filter_params.extend(normalized_tags)
                        filter_count += 1

                # Date range filtering - Use datetime() to normalize ISO 8601 input
                # datetime() converts all ISO 8601 formats (T separator, Z suffix, timezone offsets)
                # to SQLite's space-separated format 'YYYY-MM-DD HH:MM:SS' for proper comparison.
                # Without datetime(), TEXT comparison fails because 'T' > ' ' in ASCII ordering.
                if start_date:
                    filter_conditions.append('created_at >= datetime(?)')
                    filter_params.append(start_date)
                    filter_count += 1

                if end_date:
                    filter_conditions.append('created_at <= datetime(?)')
                    filter_params.append(end_date)
                    filter_count += 1

                # Metadata filtering using MetadataQueryBuilder
                metadata_filter_count = 0
                if metadata or metadata_filters:
                    from pydantic import ValidationError

                    from app.metadata_types import MetadataFilter
                    from app.query_builder import MetadataQueryBuilder

                    metadata_builder = MetadataQueryBuilder(backend_type='sqlite')

                    # Simple metadata filters (key=value equality)
                    if metadata:
                        for key, value in metadata.items():
                            try:
                                metadata_builder.add_simple_filter(key, value)
                                metadata_filter_count += 1
                            except ValueError as e:
                                logger.warning(f'Invalid simple metadata filter key={key}: {e}')

                    # Advanced metadata filters with operators
                    if metadata_filters:
                        validation_errors: list[str] = []
                        for filter_dict in metadata_filters:
                            try:
                                filter_spec = MetadataFilter(**filter_dict)
                                metadata_builder.add_advanced_filter(filter_spec)
                                metadata_filter_count += 1
                            except ValidationError as e:
                                error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                                validation_errors.append(error_msg)
                            except ValueError as e:
                                error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                                validation_errors.append(error_msg)
                            except Exception as e:
                                error_msg = f'Unexpected error in metadata filter {filter_dict}: {e}'
                                validation_errors.append(error_msg)
                                logger.error(f'Unexpected error processing metadata filter: {e}')

                        # Raise exception if validation fails (unified with search_context behavior)
                        if validation_errors:
                            raise MetadataFilterValidationError(
                                'Metadata filter validation failed',
                                validation_errors,
                            )

                    # Add metadata conditions to filter
                    metadata_clause, metadata_params = metadata_builder.build_where_clause()
                    if metadata_clause:
                        filter_conditions.append(metadata_clause)
                        filter_params.extend(metadata_params)
                        filter_count += metadata_filter_count

                where_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ''

                query = f'''
                    WITH filtered_contexts AS (
                        SELECT id
                        FROM context_entries
                        {where_clause}
                    )
                    SELECT
                        ce.id,
                        ce.thread_id,
                        ce.source,
                        ce.content_type,
                        ce.text_content,
                        ce.metadata,
                        ce.created_at,
                        ce.updated_at,
                        vec_distance_l2(?, ve.embedding) as distance
                    FROM filtered_contexts fc
                    JOIN context_entries ce ON ce.id = fc.id
                    JOIN vec_context_embeddings ve ON ve.rowid = fc.id
                    ORDER BY distance
                    LIMIT ? OFFSET ?
                '''

                params = filter_params + [query_blob, limit, offset]

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]

                # Calculate execution time and build stats
                execution_time_ms = (time_module.time() - start_time) * 1000
                stats: dict[str, Any] = {
                    'execution_time_ms': round(execution_time_ms, 2),
                    'filters_applied': filter_count,
                    'rows_returned': len(results),
                    'backend': 'sqlite',
                }

                # Get query plan if requested
                if explain_query:
                    cursor = conn.execute(f'EXPLAIN QUERY PLAN {query}', params)
                    plan_rows = cursor.fetchall()
                    plan_data: list[str] = []
                    for row in plan_rows:
                        row_dict = dict(row)
                        id_val = row_dict.get('id', '?')
                        parent_val = row_dict.get('parent', '?')
                        notused_val = row_dict.get('notused', '?')
                        detail_val = row_dict.get('detail', '?')
                        formatted = f'id:{id_val} parent:{parent_val} notused:{notused_val} detail:{detail_val}'
                        plan_data.append(formatted)
                    stats['query_plan'] = '\n'.join(plan_data)

                return results, stats

            return await self.backend.execute_read(_search_sqlite)

        # postgresql
        async def _search_postgresql(
            conn: asyncpg.Connection,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            import time as time_module

            start_time = time_module.time()

            filter_conditions = ['1=1']  # Always true, makes building easier
            filter_params: list[Any] = [query_embedding]
            param_position = 2  # Start at 2 because $1 is embedding

            # Count filters applied
            filter_count = 0

            if thread_id:
                filter_conditions.append(f'ce.thread_id = {self._placeholder(param_position)}')
                filter_params.append(thread_id)
                param_position += 1
                filter_count += 1

            if source:
                filter_conditions.append(f'ce.source = {self._placeholder(param_position)}')
                filter_params.append(source)
                param_position += 1
                filter_count += 1

            if content_type:
                filter_conditions.append(f'ce.content_type = {self._placeholder(param_position)}')
                filter_params.append(content_type)
                param_position += 1
                filter_count += 1

            # Tag filter (uses subquery with indexed tag table)
            if tags:
                normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
                if normalized_tags:
                    tag_placeholders = ','.join([
                        self._placeholder(param_position + i) for i in range(len(normalized_tags))
                    ])
                    filter_conditions.append(f'''
                        ce.id IN (
                            SELECT DISTINCT context_entry_id
                            FROM tags
                            WHERE tag IN ({tag_placeholders})
                        )
                    ''')
                    filter_params.extend(normalized_tags)
                    param_position += len(normalized_tags)
                    filter_count += 1

            # Date range filtering - PostgreSQL uses TIMESTAMPTZ comparison
            # asyncpg requires Python datetime objects, not strings, for TIMESTAMPTZ parameters
            if start_date:
                filter_conditions.append(f'ce.created_at >= {self._placeholder(param_position)}')
                filter_params.append(self._parse_date_for_postgresql(start_date))
                param_position += 1
                filter_count += 1

            if end_date:
                filter_conditions.append(f'ce.created_at <= {self._placeholder(param_position)}')
                filter_params.append(self._parse_date_for_postgresql(end_date))
                param_position += 1
                filter_count += 1

            # Metadata filtering using MetadataQueryBuilder
            metadata_filter_count = 0
            if metadata or metadata_filters:
                from pydantic import ValidationError

                from app.metadata_types import MetadataFilter
                from app.query_builder import MetadataQueryBuilder

                # param_offset is the current number of params minus 1 because MetadataQueryBuilder
                # uses 1-based indexing and we need to continue from the current position
                metadata_builder = MetadataQueryBuilder(
                    backend_type='postgresql',
                    param_offset=len(filter_params),
                )

                # Simple metadata filters (key=value equality)
                if metadata:
                    for key, value in metadata.items():
                        try:
                            metadata_builder.add_simple_filter(key, value)
                            metadata_filter_count += 1
                        except ValueError as e:
                            logger.warning(f'Invalid simple metadata filter key={key}: {e}')

                # Advanced metadata filters with operators
                if metadata_filters:
                    validation_errors: list[str] = []
                    for filter_dict in metadata_filters:
                        try:
                            filter_spec = MetadataFilter(**filter_dict)
                            metadata_builder.add_advanced_filter(filter_spec)
                            metadata_filter_count += 1
                        except ValidationError as e:
                            error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                        except ValueError as e:
                            error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                        except Exception as e:
                            error_msg = f'Unexpected error in metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                            logger.error(f'Unexpected error processing metadata filter: {e}')

                    # Raise exception if validation fails (unified with search_context behavior)
                    if validation_errors:
                        raise MetadataFilterValidationError(
                            'Metadata filter validation failed',
                            validation_errors,
                        )

                # Add metadata conditions to filter with 'ce.' table alias prefix
                metadata_clause, metadata_params = metadata_builder.build_where_clause()
                if metadata_clause:
                    # Prefix metadata conditions with 'ce.' table alias for the context_entries table
                    metadata_clause_with_alias = metadata_clause.replace('metadata', 'ce.metadata')
                    filter_conditions.append(metadata_clause_with_alias)
                    filter_params.extend(metadata_params)
                    param_position += len(metadata_params)
                    filter_count += metadata_filter_count

            where_clause = ' AND '.join(filter_conditions)

            # Use <-> operator for L2 distance (Euclidean)
            query = f'''
                    SELECT
                        ce.id,
                        ce.thread_id,
                        ce.source,
                        ce.content_type,
                        ce.text_content,
                        ce.metadata,
                        ce.created_at,
                        ce.updated_at,
                        ve.embedding <-> {self._placeholder(1)} as distance
                    FROM context_entries ce
                    JOIN vec_context_embeddings ve ON ve.context_id = ce.id
                    WHERE {where_clause}
                    ORDER BY ve.embedding <-> {self._placeholder(1)}
                    LIMIT {self._placeholder(param_position)} OFFSET {self._placeholder(param_position + 1)}
                '''

            filter_params.extend([limit, offset])

            rows = await conn.fetch(query, *filter_params)
            results = [dict(row) for row in rows]

            # Calculate execution time and build stats
            execution_time_ms = (time_module.time() - start_time) * 1000
            stats: dict[str, Any] = {
                'execution_time_ms': round(execution_time_ms, 2),
                'filters_applied': filter_count,
                'rows_returned': len(results),
                'backend': 'postgresql',
            }

            # Get query plan if requested
            if explain_query:
                plan_result = await conn.fetch(f'EXPLAIN {query}', *filter_params)
                plan_data = [str(row[0]) for row in plan_result]
                stats['query_plan'] = '\n'.join(plan_data)

            return results, stats

        return await self.backend.execute_read(_search_postgresql)

    async def update(self, context_id: int, embedding: list[float]) -> None:
        """Update embedding for a context entry.

        Args:
            context_id: ID of the context entry
            embedding: New embedding vector
        """
        if self.backend.backend_type == 'sqlite':

            def _update_sqlite(conn: sqlite3.Connection) -> None:
                try:
                    import sqlite_vec
                except ImportError as e:
                    raise RuntimeError(
                        'sqlite_vec package is required for semantic search. Install with: uv sync --extra semantic-search',
                    ) from e

                embedding_blob = sqlite_vec.serialize_float32(embedding)
                query1 = (
                    f'UPDATE vec_context_embeddings SET embedding = {self._placeholder(1)} '
                    f'WHERE rowid = {self._placeholder(2)}'
                )
                conn.execute(query1, (embedding_blob, context_id))

                query2 = (
                    f'UPDATE embedding_metadata SET updated_at = CURRENT_TIMESTAMP WHERE context_id = {self._placeholder(1)}'
                )
                conn.execute(query2, (context_id,))

            await self.backend.execute_write(_update_sqlite)
            logger.debug(f'Updated embedding for context {context_id} (SQLite)')

        else:  # postgresql

            async def _update_postgresql(conn: asyncpg.Connection) -> None:
                # Update vec_context_embeddings
                query1 = (
                    f'UPDATE vec_context_embeddings SET embedding = {self._placeholder(1)} '
                    f'WHERE context_id = {self._placeholder(2)}'
                )
                await conn.execute(query1, embedding, context_id)

                # Update timestamp in embedding_metadata (trigger handles updated_at)
                query2 = (
                    f'UPDATE embedding_metadata SET updated_at = CURRENT_TIMESTAMP WHERE context_id = {self._placeholder(1)}'
                )
                await conn.execute(query2, context_id)

            await self.backend.execute_write(cast(Any, _update_postgresql))
            logger.debug(f'Updated embedding for context {context_id} (PostgreSQL)')

    async def delete(self, context_id: int) -> None:
        """Delete embedding for a context entry.

        Args:
            context_id: ID of the context entry
        """
        if self.backend.backend_type == 'sqlite':

            def _delete_sqlite(conn: sqlite3.Connection) -> None:
                query1 = f'DELETE FROM vec_context_embeddings WHERE rowid = {self._placeholder(1)}'
                conn.execute(query1, (context_id,))

                query2 = f'DELETE FROM embedding_metadata WHERE context_id = {self._placeholder(1)}'
                conn.execute(query2, (context_id,))

            await self.backend.execute_write(_delete_sqlite)
            logger.debug(f'Deleted embedding for context {context_id} (SQLite)')

        else:  # postgresql

            async def _delete_postgresql(conn: asyncpg.Connection) -> None:
                # Delete from vec_context_embeddings (CASCADE will delete from embedding_metadata)
                query = f'DELETE FROM vec_context_embeddings WHERE context_id = {self._placeholder(1)}'
                await conn.execute(query, context_id)

            await self.backend.execute_write(cast(Any, _delete_postgresql))
            logger.debug(f'Deleted embedding for context {context_id} (PostgreSQL)')

    async def exists(self, context_id: int) -> bool:
        """Check if embedding exists for context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            True if embedding exists, False otherwise
        """
        if self.backend.backend_type == 'sqlite':

            def _exists_sqlite(conn: sqlite3.Connection) -> bool:
                query = f'SELECT 1 FROM embedding_metadata WHERE context_id = {self._placeholder(1)} LIMIT 1'
                cursor = conn.execute(query, (context_id,))
                return cursor.fetchone() is not None

            return await self.backend.execute_read(_exists_sqlite)

        # postgresql
        async def _exists_postgresql(conn: asyncpg.Connection) -> bool:
            query = f'SELECT 1 FROM embedding_metadata WHERE context_id = {self._placeholder(1)} LIMIT 1'
            row = await conn.fetchrow(query, context_id)
            return row is not None

        return await self.backend.execute_read(_exists_postgresql)

    async def get_statistics(self, thread_id: str | None = None) -> dict[str, Any]:
        """Get embedding statistics.

        Args:
            thread_id: Optional filter by thread

        Returns:
            Dictionary with statistics (count, coverage, etc.)
        """
        if self.backend.backend_type == 'sqlite':

            def _get_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                if thread_id:
                    query1 = f'SELECT COUNT(*) FROM context_entries WHERE thread_id = {self._placeholder(1)}'
                    cursor = conn.execute(query1, (thread_id,))
                else:
                    cursor = conn.execute('SELECT COUNT(*) FROM context_entries')

                total_entries = cursor.fetchone()[0]

                if thread_id:
                    query2 = f'''
                        SELECT COUNT(*)
                        FROM embedding_metadata em
                        JOIN context_entries ce ON em.context_id = ce.id
                        WHERE ce.thread_id = {self._placeholder(1)}
                    '''
                    cursor = conn.execute(query2, (thread_id,))
                else:
                    cursor = conn.execute('SELECT COUNT(*) FROM embedding_metadata')

                embedding_count = cursor.fetchone()[0]

                coverage_percentage = (embedding_count / total_entries * 100) if total_entries > 0 else 0.0

                return {
                    'total_embeddings': embedding_count,
                    'total_entries': total_entries,
                    'coverage_percentage': round(coverage_percentage, 2),
                    'backend': 'sqlite',
                }

            return await self.backend.execute_read(_get_stats_sqlite)

        # postgresql
        async def _get_stats_postgresql(conn: asyncpg.Connection) -> dict[str, Any]:
            if thread_id:
                query1 = f'SELECT COUNT(*) FROM context_entries WHERE thread_id = {self._placeholder(1)}'
                total_entries = await conn.fetchval(query1, thread_id)
            else:
                total_entries = await conn.fetchval('SELECT COUNT(*) FROM context_entries')

            if thread_id:
                query2 = f'''
                        SELECT COUNT(*)
                        FROM embedding_metadata em
                        JOIN context_entries ce ON em.context_id = ce.id
                        WHERE ce.thread_id = {self._placeholder(1)}
                    '''
                embedding_count = await conn.fetchval(query2, thread_id)
            else:
                embedding_count = await conn.fetchval('SELECT COUNT(*) FROM embedding_metadata')

            coverage_percentage = (embedding_count / total_entries * 100) if total_entries > 0 else 0.0

            return {
                'total_embeddings': embedding_count,
                'total_entries': total_entries,
                'coverage_percentage': round(coverage_percentage, 2),
                'backend': 'postgresql',
            }

        return await self.backend.execute_read(_get_stats_postgresql)

    async def get_table_dimension(self) -> int | None:
        """Get the dimension of the existing vector table.

        This is useful for diagnostics and validation to check if the configured
        EMBEDDING_DIM matches the actual table dimension.

        Returns:
            Dimension of existing embeddings, or None if no embeddings exist
        """
        if self.backend.backend_type == 'sqlite':

            def _get_dimension_sqlite(conn: sqlite3.Connection) -> int | None:
                cursor = conn.execute('SELECT dimensions FROM embedding_metadata LIMIT 1')
                row = cursor.fetchone()
                return row[0] if row else None

            return await self.backend.execute_read(_get_dimension_sqlite)

        # postgresql
        async def _get_dimension_postgresql(conn: asyncpg.Connection) -> int | None:
            row = await conn.fetchrow('SELECT dimensions FROM embedding_metadata LIMIT 1')
            return row['dimensions'] if row else None

        return await self.backend.execute_read(_get_dimension_postgresql)
