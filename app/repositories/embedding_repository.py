"""
Repository for vector embeddings supporting both sqlite-vec and pgvector.

This module provides data access for semantic search embeddings,
handling storage, retrieval, and search operations on vector embeddings
across both SQLite (sqlite-vec) and PostgreSQL (pgvector) backends.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext

logger = logging.getLogger(__name__)


@dataclass
class ChunkEmbedding:
    """Embedding data for a single chunk with boundary information.

    This dataclass bundles embedding vector with its character boundaries
    in the original document, enabling chunk-aware reranking.

    Attributes:
        embedding: The embedding vector for this chunk.
        start_index: Character offset where chunk starts in original document.
        end_index: Character offset where chunk ends in original document.

    Example:
        >>> chunk_emb = ChunkEmbedding(
        ...     embedding=[0.1, 0.2, 0.3],
        ...     start_index=0,
        ...     end_index=100
        ... )
    """

    embedding: list[float]
    start_index: int
    end_index: int


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
        model: str,
        *,
        start_index: int = 0,
        end_index: int = 0,
    ) -> None:
        """Store embedding for a context entry.

        This is a convenience method for storing a single embedding. It uses the chunked
        storage architecture internally, creating a single-chunk entry for compatibility
        with the 1:N embedding schema.

        Args:
            context_id: ID of the context entry
            embedding: Embedding vector (dimension depends on provider/model configuration)
            model: Model identifier (from settings.embedding.model)
            start_index: Character offset where text starts (default: 0 for full document)
            end_index: Character offset where text ends (default: 0 for legacy/unknown)
        """
        # Delegate to store_chunked with single embedding for unified storage logic
        chunk_emb = ChunkEmbedding(embedding=embedding, start_index=start_index, end_index=end_index)
        await self.store_chunked(context_id, [chunk_emb], model)
        logger.debug(f'Stored embedding for context {context_id}')

    async def store_chunked(
        self,
        context_id: int,
        chunk_embeddings: list[ChunkEmbedding],
        model: str,
        txn: TransactionContext | None = None,
        *,
        upsert: bool = False,
    ) -> None:
        """Store multiple chunk embeddings with boundaries for a context entry atomically.

        This method replaces store() for chunked content. All embeddings are
        stored in a single transaction - either all succeed or all fail.
        Chunk boundaries are stored for chunk-aware reranking.

        Args:
            context_id: ID of the context entry
            chunk_embeddings: List of ChunkEmbedding objects (embedding + boundaries)
            model: Model identifier (from settings.embedding.model)
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.
            upsert: If True, delete existing embeddings before storing new ones.
                Use this for defense-in-depth when deduplication is possible.
                Default is False for backward compatibility.

        Raises:
            ValueError: If chunk_embeddings list is empty
        """
        if not chunk_embeddings:
            raise ValueError('chunk_embeddings list cannot be empty')

        # Defense-in-depth: if upsert enabled, delete existing embeddings first
        # This ensures idempotency - calling multiple times produces same result
        if upsert:
            deleted_count = await self.delete_all_chunks(context_id, txn=txn)
            if deleted_count > 0:
                logger.debug(
                    f'UPSERT mode: deleted {deleted_count} existing chunk embeddings '
                    f'for context {context_id} before storing new ones',
                )

        chunk_count = len(chunk_embeddings)
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _store_chunked_sqlite(conn: sqlite3.Connection) -> None:
                try:
                    import sqlite_vec
                except ImportError as e:
                    raise RuntimeError(
                        'sqlite_vec package is required for semantic search. '
                        'Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)',
                    ) from e

                # Step 1: Get next available rowid for vec0 virtual table
                cursor = conn.execute('SELECT COALESCE(MAX(rowid), 0) + 1 FROM vec_context_embeddings')
                next_rowid = cursor.fetchone()[0]

                vec_rowids: list[int] = []
                for i, chunk_emb in enumerate(chunk_embeddings):
                    vec_rowid = next_rowid + i
                    embedding_blob: bytes = cast(Any, sqlite_vec).serialize_float32(chunk_emb.embedding)
                    conn.execute(
                        'INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (?, ?)',
                        (vec_rowid, embedding_blob),
                    )
                    vec_rowids.append(vec_rowid)

                # Step 2: Insert mapping records into embedding_chunks WITH BOUNDARIES
                for i, vec_rowid in enumerate(vec_rowids):
                    chunk_emb = chunk_embeddings[i]
                    conn.execute(
                        'INSERT INTO embedding_chunks(context_id, vec_rowid, start_index, end_index) VALUES (?, ?, ?, ?)',
                        (context_id, vec_rowid, chunk_emb.start_index, chunk_emb.end_index),
                    )

                # Step 3: Insert embedding_metadata with chunk_count
                conn.execute(
                    '''INSERT INTO embedding_metadata (context_id, model_name, dimensions, chunk_count, created_at, updated_at)
                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)''',
                    (context_id, model, len(chunk_embeddings[0].embedding), chunk_count),
                )

            if txn:
                _store_chunked_sqlite(cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_store_chunked_sqlite)
            logger.debug(f'Stored {chunk_count} chunk embeddings for context {context_id} (SQLite)')

        else:  # postgresql

            async def _store_chunked_postgresql(conn: asyncpg.Connection) -> None:
                # Step 1: Insert all embeddings into vec_context_embeddings WITH BOUNDARIES
                # PostgreSQL uses id BIGSERIAL, context_id can repeat (1:N)
                for chunk_emb in chunk_embeddings:
                    await conn.execute(
                        '''INSERT INTO vec_context_embeddings(context_id, embedding, start_index, end_index)
                           VALUES ($1, $2, $3, $4)''',
                        context_id, chunk_emb.embedding, chunk_emb.start_index, chunk_emb.end_index,
                    )

                # Step 2: Insert embedding_metadata with chunk_count
                await conn.execute(
                    '''INSERT INTO embedding_metadata (context_id, model_name, dimensions, chunk_count, created_at, updated_at)
                       VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)''',
                    context_id, model, len(chunk_embeddings[0].embedding), chunk_count,
                )

            if txn:
                await _store_chunked_postgresql(cast('asyncpg.Connection', txn.connection))
            else:
                await self.backend.execute_write(cast(Any, _store_chunked_postgresql))
            logger.debug(f'Stored {chunk_count} chunk embeddings for context {context_id} (PostgreSQL)')

    async def delete_all_chunks(
        self,
        context_id: int,
        txn: TransactionContext | None = None,
    ) -> int:
        """Delete all chunk embeddings for a context entry.

        Used before re-embedding when content is updated.
        For SQLite, also cleans up embedding_chunks mapping table.

        Args:
            context_id: ID of the context entry
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Number of chunk embeddings deleted
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _delete_all_chunks_sqlite(conn: sqlite3.Connection) -> int:
                # Step 1: Get vec_rowids from embedding_chunks
                cursor = conn.execute(
                    'SELECT vec_rowid FROM embedding_chunks WHERE context_id = ?',
                    (context_id,),
                )
                vec_rowids = [row[0] for row in cursor.fetchall()]

                if not vec_rowids:
                    return 0

                # Step 2: Delete from vec_context_embeddings (virtual table)
                for vec_rowid in vec_rowids:
                    conn.execute(
                        'DELETE FROM vec_context_embeddings WHERE rowid = ?',
                        (vec_rowid,),
                    )

                # Step 3: Delete from embedding_chunks
                conn.execute(
                    'DELETE FROM embedding_chunks WHERE context_id = ?',
                    (context_id,),
                )

                # Step 4: Delete from embedding_metadata
                conn.execute(
                    'DELETE FROM embedding_metadata WHERE context_id = ?',
                    (context_id,),
                )

                return len(vec_rowids)

            if txn:
                deleted_count = _delete_all_chunks_sqlite(cast(sqlite3.Connection, txn.connection))
            else:
                deleted_count = await self.backend.execute_write(_delete_all_chunks_sqlite)
            logger.debug(f'Deleted {deleted_count} chunk embeddings for context {context_id} (SQLite)')
            return deleted_count

        # postgresql

        async def _delete_all_chunks_postgresql(conn: asyncpg.Connection) -> int:
            # Step 1: Count chunks before delete
            count: int = await conn.fetchval(
                'SELECT COUNT(*) FROM vec_context_embeddings WHERE context_id = $1',
                context_id,
            )

            if count == 0:
                return 0

            # Step 2: Delete from vec_context_embeddings
            await conn.execute(
                'DELETE FROM vec_context_embeddings WHERE context_id = $1',
                context_id,
            )

            # Step 3: Delete from embedding_metadata
            await conn.execute(
                'DELETE FROM embedding_metadata WHERE context_id = $1',
                context_id,
            )

            return count

        if txn:
            deleted_count = await _delete_all_chunks_postgresql(cast('asyncpg.Connection', txn.connection))
        else:
            deleted_count = await self.backend.execute_write(cast(Any, _delete_all_chunks_postgresql))
        logger.debug(f'Deleted {deleted_count} chunk embeddings for context {context_id} (PostgreSQL)')
        return deleted_count

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
                        'sqlite_vec package is required for semantic search. '
                        'Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)',
                    ) from e

                query_blob: bytes = cast(Any, sqlite_vec).serialize_float32(query_embedding)

                filter_conditions: list[str] = []
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

                # Use CTE with deduplication by context_id - preserves best chunk boundaries
                # Uses subquery JOIN to identify which chunk had MIN(distance)
                query = f'''
                    WITH filtered_contexts AS (
                        SELECT id
                        FROM context_entries
                        {where_clause}
                    ),
                    chunk_distances AS (
                        SELECT
                            ec.context_id,
                            ec.start_index,
                            ec.end_index,
                            vec_distance_l2(?, ve.embedding) as distance
                        FROM filtered_contexts fc
                        JOIN embedding_chunks ec ON ec.context_id = fc.id
                        JOIN vec_context_embeddings ve ON ve.rowid = ec.vec_rowid
                    ),
                    best_chunks AS (
                        SELECT
                            cd.context_id,
                            cd.start_index,
                            cd.end_index,
                            cd.distance as best_distance
                        FROM chunk_distances cd
                        INNER JOIN (
                            SELECT context_id, MIN(distance) as min_distance
                            FROM chunk_distances
                            GROUP BY context_id
                        ) min_cd ON cd.context_id = min_cd.context_id
                                AND cd.distance = min_cd.min_distance
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
                        bc.best_distance as distance,
                        bc.start_index as matched_chunk_start,
                        bc.end_index as matched_chunk_end
                    FROM best_chunks bc
                    JOIN context_entries ce ON ce.id = bc.context_id
                    ORDER BY bc.best_distance
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

            # Use CTE with DISTINCT ON to preserve best chunk boundaries
            # DISTINCT ON selects first row per context_id when ordered by distance
            query = f'''
                    WITH chunk_distances AS (
                        SELECT
                            ve.context_id,
                            ve.start_index,
                            ve.end_index,
                            ve.embedding <-> {self._placeholder(1)} as distance
                        FROM vec_context_embeddings ve
                        JOIN context_entries ce ON ce.id = ve.context_id
                        WHERE {where_clause}
                    ),
                    best_chunks AS (
                        SELECT DISTINCT ON (context_id)
                            context_id,
                            start_index,
                            end_index,
                            distance as best_distance
                        FROM chunk_distances
                        ORDER BY context_id, distance
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
                        bc.best_distance as distance,
                        bc.start_index as matched_chunk_start,
                        bc.end_index as matched_chunk_end
                    FROM best_chunks bc
                    JOIN context_entries ce ON ce.id = bc.context_id
                    ORDER BY bc.best_distance
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

    async def update(
        self,
        context_id: int,
        chunk_embeddings: list[ChunkEmbedding],
        model: str,
    ) -> None:
        """Update embeddings for a context entry (delete old, store new).

        This method replaces all existing chunk embeddings with new ones atomically.
        Uses delete-all + store-chunked pattern for consistency.

        Args:
            context_id: ID of the context entry
            chunk_embeddings: New ChunkEmbedding objects (embedding + boundaries)
            model: Model identifier
        """
        # Delete existing chunks
        await self.delete_all_chunks(context_id)

        # Store new chunks
        await self.store_chunked(context_id, chunk_embeddings, model)

        logger.debug(f'Updated {len(chunk_embeddings)} embeddings for context {context_id}')

    async def delete(self, context_id: int) -> None:
        """Delete all embeddings for a context entry.

        Delegates to delete_all_chunks() for proper cleanup of chunked embeddings.

        Args:
            context_id: ID of the context entry
        """
        await self.delete_all_chunks(context_id)

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
        """Get embedding statistics including chunk information.

        Args:
            thread_id: Optional filter by thread

        Returns:
            Dictionary with statistics (count, coverage, chunk info, etc.)
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

                # Get total chunk count from embedding_chunks table
                if thread_id:
                    query3 = f'''
                        SELECT COUNT(*)
                        FROM embedding_chunks ec
                        JOIN context_entries ce ON ec.context_id = ce.id
                        WHERE ce.thread_id = {self._placeholder(1)}
                    '''
                    cursor = conn.execute(query3, (thread_id,))
                else:
                    cursor = conn.execute('SELECT COUNT(*) FROM embedding_chunks')

                total_chunks = cursor.fetchone()[0]

                coverage_percentage = (embedding_count / total_entries * 100) if total_entries > 0 else 0.0
                average_chunks = round(total_chunks / embedding_count, 2) if embedding_count > 0 else 0.0

                return {
                    'total_embeddings': embedding_count,
                    'total_entries': total_entries,
                    'total_chunks': total_chunks,
                    'average_chunks_per_entry': average_chunks,
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

            # Get total chunk count from vec_context_embeddings (1:N relationship)
            if thread_id:
                query3 = f'''
                        SELECT COUNT(*)
                        FROM vec_context_embeddings ve
                        JOIN context_entries ce ON ve.context_id = ce.id
                        WHERE ce.thread_id = {self._placeholder(1)}
                    '''
                total_chunks = await conn.fetchval(query3, thread_id)
            else:
                total_chunks = await conn.fetchval('SELECT COUNT(*) FROM vec_context_embeddings')

            coverage_percentage = (embedding_count / total_entries * 100) if total_entries > 0 else 0.0
            average_chunks = round(total_chunks / embedding_count, 2) if embedding_count > 0 else 0.0

            return {
                'total_embeddings': embedding_count,
                'total_entries': total_entries,
                'total_chunks': total_chunks,
                'average_chunks_per_entry': average_chunks,
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
