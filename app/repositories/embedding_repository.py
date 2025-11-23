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


class EmbeddingRepository(BaseRepository):
    """Repository for vector embeddings supporting both sqlite-vec and pgvector.

    This repository handles all database operations for semantic search embeddings,
    using either sqlite-vec extension (SQLite) or pgvector extension (PostgreSQL/Supabase)
    depending on the configured storage backend.

    Supported backends:
    - SQLite: Uses sqlite-vec with BLOB storage and vec_distance_l2()
    - PostgreSQL: Uses pgvector with native vector type and <-> operator
    - Supabase: Uses pgvector (same as PostgreSQL)
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

        else:  # postgresql, supabase

            async def _store_postgresql(conn: asyncpg.Connection) -> None:
                try:
                    import numpy as np
                except ImportError as e:
                    raise RuntimeError(
                        'numpy package is required for semantic search. Install with: uv sync --extra semantic-search',
                    ) from e

                embedding_array = np.array(embedding, dtype=np.float32)

                # Insert into vec_context_embeddings
                query1 = (
                    f'INSERT INTO vec_context_embeddings(context_id, embedding) '
                    f'VALUES ({self._placeholder(1)}, {self._placeholder(2)})'
                )
                await conn.execute(query1, context_id, embedding_array)

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
        thread_id: str | None = None,
        source: Literal['user', 'agent'] | None = None,
    ) -> list[dict[str, Any]]:
        """KNN search with optional filters.

        SQLite: Uses CTE-based pre-filtering with vec_distance_l2() function
        PostgreSQL: Uses direct JOIN with <-> operator for L2 distance

        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results to return
            thread_id: Optional filter by thread
            source: Optional filter by source type

        Returns:
            List of search results with context and similarity scores
        """
        if self.backend.backend_type == 'sqlite':

            def _search_sqlite(conn: sqlite3.Connection) -> list[dict[str, Any]]:
                try:
                    import sqlite_vec
                except ImportError as e:
                    raise RuntimeError(
                        'sqlite_vec package is required for semantic search. Install with: uv sync --extra semantic-search',
                    ) from e

                query_blob = sqlite_vec.serialize_float32(query_embedding)

                filter_conditions = []
                filter_params: list[Any] = []
                param_position = 1

                if thread_id:
                    filter_conditions.append(f'thread_id = {self._placeholder(param_position)}')
                    filter_params.append(thread_id)
                    param_position += 1

                if source:
                    filter_conditions.append(f'source = {self._placeholder(param_position)}')
                    filter_params.append(source)
                    param_position += 1

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
                        vec_distance_l2({self._placeholder(param_position)}, ve.embedding) as distance
                    FROM filtered_contexts fc
                    JOIN context_entries ce ON ce.id = fc.id
                    JOIN vec_context_embeddings ve ON ve.rowid = fc.id
                    ORDER BY distance
                    LIMIT {self._placeholder(param_position + 1)}
                '''

                params = filter_params + [query_blob, limit]

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

            return await self.backend.execute_read(_search_sqlite)

        # postgresql, supabase
        async def _search_postgresql(conn: asyncpg.Connection) -> list[dict[str, Any]]:
            try:
                import numpy as np
            except ImportError as e:
                raise RuntimeError(
                    'numpy package is required for semantic search. Install with: uv sync --extra semantic-search',
                ) from e

            embedding_array = np.array(query_embedding, dtype=np.float32)

            filter_conditions = ['1=1']  # Always true, makes building easier
            filter_params: list[Any] = [embedding_array]
            param_position = 2  # Start at 2 because $1 is embedding

            if thread_id:
                filter_conditions.append(f'ce.thread_id = {self._placeholder(param_position)}')
                filter_params.append(thread_id)
                param_position += 1

            if source:
                filter_conditions.append(f'ce.source = {self._placeholder(param_position)}')
                filter_params.append(source)
                param_position += 1

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
                    LIMIT {self._placeholder(param_position)}
                '''

            filter_params.append(limit)

            rows = await conn.fetch(query, *filter_params)

            return [dict(row) for row in rows]

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

        else:  # postgresql, supabase

            async def _update_postgresql(conn: asyncpg.Connection) -> None:
                try:
                    import numpy as np
                except ImportError as e:
                    raise RuntimeError(
                        'numpy package is required for semantic search. Install with: uv sync --extra semantic-search',
                    ) from e

                embedding_array = np.array(embedding, dtype=np.float32)

                # Update vec_context_embeddings
                query1 = (
                    f'UPDATE vec_context_embeddings SET embedding = {self._placeholder(1)} '
                    f'WHERE context_id = {self._placeholder(2)}'
                )
                await conn.execute(query1, embedding_array, context_id)

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

        else:  # postgresql, supabase

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

        # postgresql, supabase
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
                }

            return await self.backend.execute_read(_get_stats_sqlite)

        # postgresql, supabase
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

        # postgresql, supabase
        async def _get_dimension_postgresql(conn: asyncpg.Connection) -> int | None:
            row = await conn.fetchrow('SELECT dimensions FROM embedding_metadata LIMIT 1')
            return row['dimensions'] if row else None

        return await self.backend.execute_read(_get_dimension_postgresql)
